"""Native-async Microsoft Entra ID adapter for Azure AI Foundry."""

from __future__ import annotations

import asyncio
import logging
import os
import weakref
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional

logger = logging.getLogger(__name__)

SCOPE_AI_AZURE_DEFAULT = "https://ai.azure.com/.default"

try:
    import azure.identity.aio as _azure_identity
    _azure_identity_import_error: Exception | None = None
except Exception as exc:
    _azure_identity = None
    _azure_identity_import_error = exc

_credential_caches: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[EntraIdentityConfig, Any]]" = (
    weakref.WeakKeyDictionary()
)
_credential_leases: dict[int, int] = {}
_issued_providers: "weakref.WeakSet[Any]" = weakref.WeakSet()


def has_azure_identity_installed() -> bool:
    """Return whether the optional async Azure Identity SDK is installed."""
    return _azure_identity is not None


def _require_azure_identity():
    if _azure_identity is None:
        if _azure_identity_import_error is not None and not isinstance(
            _azure_identity_import_error,
            ImportError,
        ):
            raise _azure_identity_import_error
        raise ImportError(
            "The 'azure-identity' package is required for Azure AI Foundry "
            "Entra ID authentication. Install it with: "
            "pip install 'async-hermes-agent[azure-identity]'"
        ) from _azure_identity_import_error
    return _azure_identity


@dataclass(frozen=True)
class EntraIdentityConfig:
    """Serializable configuration for the async Azure credential chain."""

    scope: str = SCOPE_AI_AZURE_DEFAULT
    exclude_interactive_browser: bool = True

    def __post_init__(self) -> None:
        scope = str(self.scope or "").strip() or SCOPE_AI_AZURE_DEFAULT
        object.__setattr__(self, "scope", scope)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scope": self.scope,
            "exclude_interactive_browser": self.exclude_interactive_browser,
        }

    @classmethod
    def from_dict(
        cls,
        data: Optional[Dict[str, Any]],
        *,
        default_scope: Optional[str] = None,
    ) -> "EntraIdentityConfig":
        data = data or {}
        return cls(
            scope=(
                str(data.get("scope") or "").strip()
                or default_scope
                or SCOPE_AI_AZURE_DEFAULT
            ),
            exclude_interactive_browser=bool(
                data.get("exclude_interactive_browser", True)
            ),
        )


async def build_credential(config: EntraIdentityConfig) -> Any:
    """Return the event-loop-local async ``DefaultAzureCredential``."""
    loop = asyncio.get_running_loop()
    cache = _credential_caches.setdefault(loop, {})
    credential = cache.get(config)
    if credential is not None:
        return credential

    identity = _require_azure_identity()
    kwargs: Dict[str, Any] = {}
    if not config.exclude_interactive_browser:
        kwargs["exclude_interactive_browser_credential"] = False
    credential = identity.DefaultAzureCredential(**kwargs)
    cache[config] = credential
    return credential


async def reset_credential_cache() -> None:
    """Close and clear cached asynchronous Azure credential chains."""
    credentials = [
        credential
        for cache in tuple(_credential_caches.values())
        for credential in cache.values()
    ]
    _credential_caches.clear()
    _credential_leases.clear()
    for provider in tuple(_issued_providers):
        provider._hermes_released = True
    _issued_providers.clear()
    for credential in credentials:
        close = getattr(credential, "close", None)
        if close is not None:
            await close()


async def build_token_provider(
    scope: Optional[str] = None,
    *,
    config: Optional[EntraIdentityConfig] = None,
    base_url: Optional[str] = None,
    exclude_interactive_browser: bool = True,
) -> Callable[[], Awaitable[str]]:
    """Return Microsoft's native coroutine bearer-token provider."""
    del base_url
    identity = _require_azure_identity()
    if config is None:
        config = EntraIdentityConfig(
            scope=scope or SCOPE_AI_AZURE_DEFAULT,
            exclude_interactive_browser=exclude_interactive_browser,
        )
    credential = await build_credential(config)
    provider = identity.get_bearer_token_provider(credential, config.scope)
    provider._hermes_credential = credential
    provider._hermes_released = False
    _issued_providers.add(provider)
    _credential_leases[id(credential)] = _credential_leases.get(id(credential), 0) + 1
    return provider


async def _release_token_provider(value: Any) -> None:
    """Release one provider lease and close its credential after the last use."""
    credential = getattr(value, "_hermes_credential", None)
    if credential is None or getattr(value, "_hermes_released", False):
        return
    value._hermes_released = True
    _issued_providers.discard(value)
    credential_id = id(credential)
    remaining = _credential_leases.get(credential_id, 1) - 1
    if remaining > 0:
        _credential_leases[credential_id] = remaining
        return
    _credential_leases.pop(credential_id, None)
    for cache in tuple(_credential_caches.values()):
        for config, cached in tuple(cache.items()):
            if cached is credential:
                cache.pop(config, None)
    close = getattr(credential, "close", None)
    if close is not None:
        await close()


def is_token_provider(value: Any) -> bool:
    return callable(value) and not isinstance(value, str)


async def materialize_bearer_for_http(value: Any) -> str:
    """Resolve a fresh token for a manual async HTTP request."""
    if is_token_provider(value):
        token = await value()
        if isinstance(token, str) and token:
            return token
        raise ValueError("token provider returned empty value")
    if isinstance(value, str) and value:
        return value
    raise ValueError("no usable api_key / token provider")


async def build_bearer_http_client(
    token_provider: Callable[[], Awaitable[str]],
    **httpx_kwargs: Any,
) -> Any:
    """Return an ``httpx.AsyncClient`` with per-request Entra bearer refresh."""
    if not is_token_provider(token_provider):
        raise ValueError(
            "build_bearer_http_client requires a zero-arg coroutine token provider"
        )

    import httpx
    from agent.ssl_verify import _create_httpx_client

    async def inject_bearer(request: httpx.Request) -> None:
        token = await materialize_bearer_for_http(token_provider)
        for header_name in (
            "Authorization",
            "Api-Key",
            "X-Api-Key",
        ):
            request.headers.pop(header_name, None)
        request.headers["Authorization"] = f"Bearer {token}"

    client = await _create_httpx_client(
        event_hooks={"request": [inject_bearer]},
        **httpx_kwargs,
    )
    client._hermes_token_provider = token_provider
    return client


async def has_azure_identity_credentials(
    scope: Optional[str] = None,
    *,
    config: Optional[EntraIdentityConfig] = None,
    timeout_seconds: float = 10.0,
    allow_install: bool = True,
    **overrides: Any,
) -> bool:
    """Return whether the async credential chain can mint a token in time."""
    del allow_install
    if not has_azure_identity_installed():
        return False
    if config is None:
        config = EntraIdentityConfig(
            scope=(scope or "").strip() or SCOPE_AI_AZURE_DEFAULT,
            **overrides,
        )
    try:
        credential = await build_credential(config)
        async with asyncio.timeout(max(0.01, timeout_seconds)):
            token = await credential.get_token(config.scope)
        return bool(getattr(token, "token", None))
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.debug("Entra credential probe failed: %s", exc)
        return False


async def describe_active_credential(
    config: Optional[EntraIdentityConfig] = None,
    *,
    scope: Optional[str] = None,
    timeout_seconds: float = 10.0,
    allow_install: bool = True,
    **overrides: Any,
) -> Dict[str, Any]:
    """Return non-secret diagnostics for the active async identity chain."""
    del allow_install
    if config is None:
        config = EntraIdentityConfig(
            scope=(scope or "").strip() or SCOPE_AI_AZURE_DEFAULT,
            **overrides,
        )
    info: Dict[str, Any] = {"ok": False, "scope": config.scope}
    if not has_azure_identity_installed():
        info["error"] = "azure-identity not installed"
        info["hint"] = "pip install 'async-hermes-agent[azure-identity]'"
        return info

    if os.environ.get("AZURE_TENANT_ID", "").strip():
        info["tenant_id_env"] = os.environ["AZURE_TENANT_ID"].strip()
    env_sources = []
    if os.environ.get("AZURE_FEDERATED_TOKEN_FILE", "").strip():
        env_sources.append("WorkloadIdentityCredential (AZURE_FEDERATED_TOKEN_FILE)")
    if all(
        os.environ.get(name, "").strip()
        for name in ("AZURE_CLIENT_ID", "AZURE_CLIENT_SECRET", "AZURE_TENANT_ID")
    ):
        env_sources.append("EnvironmentCredential (client secret)")
    if os.environ.get("IDENTITY_ENDPOINT", "").strip() or os.environ.get(
        "MSI_ENDPOINT", ""
    ).strip():
        env_sources.append("ManagedIdentityCredential (IDENTITY_ENDPOINT)")
    info["env_sources"] = env_sources

    try:
        credential = await build_credential(config)
        async with asyncio.timeout(max(0.01, timeout_seconds)):
            token = await credential.get_token(config.scope)
    except asyncio.CancelledError:
        raise
    except TimeoutError:
        info["error"] = f"Token probe timed out after {timeout_seconds:.0f}s"
        return info
    except Exception as exc:
        info["error"] = str(exc)
        return info

    info["ok"] = True
    info["expires_on"] = getattr(token, "expires_on", None)
    return info


__all__ = [
    "EntraIdentityConfig",
    "SCOPE_AI_AZURE_DEFAULT",
    "build_bearer_http_client",
    "build_credential",
    "build_token_provider",
    "describe_active_credential",
    "has_azure_identity_credentials",
    "has_azure_identity_installed",
    "is_token_provider",
    "materialize_bearer_for_http",
    "reset_credential_cache",
]
