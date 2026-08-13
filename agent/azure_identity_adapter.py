"""Native-async Microsoft Entra ID adapter for Azure AI Foundry."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import threading
import weakref
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional

import aiofiles.os as _aiofiles_os

from agent.secret_scope import (
    UnscopedSecretError as _UnscopedSecretError,
    current_secret_scope as _current_secret_scope,
    is_multiplex_active as _is_multiplex_active,
)
from hermes_constants import get_hermes_home as _get_hermes_home

logger = logging.getLogger(__name__)

SCOPE_AI_AZURE_DEFAULT = "https://ai.azure.com/.default"

try:
    import azure.identity.aio as _azure_identity_module

    _azure_identity: Any = _azure_identity_module
    _azure_identity_import_error: Exception | None = None
except Exception as exc:
    _azure_identity = None
    _azure_identity_import_error = exc

_AZURE_PUBLIC_CLOUD_AUTHORITY = "login.microsoftonline.com"
_AZURE_IDENTITY_ENV_NAMES = (
    "AZURE_AUTHORITY_HOST",
    "AZURE_CLIENT_CERTIFICATE_PASSWORD",
    "AZURE_CLIENT_CERTIFICATE_PATH",
    "AZURE_CLIENT_ID",
    "AZURE_CLIENT_SECRET",
    "AZURE_CLIENT_SEND_CERTIFICATE_CHAIN",
    "AZURE_FEDERATED_TOKEN_FILE",
    "AZURE_TENANT_ID",
    "AZURE_TOKEN_CREDENTIALS",
    "AZURE_USERNAME",
)
_UNSUPPORTED_SCOPED_IDENTITY_ENV_NAMES = (
    "AZURE_CLIENT_CERTIFICATE_PASSWORD",
    "AZURE_CLIENT_CERTIFICATE_PATH",
    "AZURE_CLIENT_SECRET",
    "AZURE_CLIENT_SEND_CERTIFICATE_CHAIN",
    "AZURE_FEDERATED_TOKEN_FILE",
    "AZURE_TOKEN_CREDENTIALS",
    "AZURE_USERNAME",
)
_CLIENT_SECRET_ENV_NAMES = (
    "AZURE_CLIENT_ID",
    "AZURE_CLIENT_SECRET",
    "AZURE_TENANT_ID",
)
_CERTIFICATE_ENV_NAMES = (
    "AZURE_CLIENT_ID",
    "AZURE_CLIENT_CERTIFICATE_PATH",
    "AZURE_TENANT_ID",
)
_UNSUPPORTED_TOKEN_CREDENTIALS = frozenset(
    {
        "dev",
        "prod",
        "azureclicredential",
        "azuredeveloperclicredential",
        "azurepowershellcredential",
        "visualstudiocodecredential",
        "workloadidentitycredential",
    }
)
_DEFAULT_CREDENTIAL_EXCLUDES = (
    "exclude_environment_credential",
    "exclude_workload_identity_credential",
    "exclude_managed_identity_credential",
    "exclude_shared_token_cache_credential",
    "exclude_visual_studio_code_credential",
    "exclude_cli_credential",
    "exclude_developer_cli_credential",
    "exclude_powershell_credential",
)

_CredentialCacheKey = tuple[str, "EntraIdentityConfig", str]
_credential_caches: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[_CredentialCacheKey, Any]]" = (
    weakref.WeakKeyDictionary()
)
_credential_leases: dict[int, int] = {}
_issued_providers: "weakref.WeakSet[Any]" = weakref.WeakSet()
_credential_cache_guard = threading.RLock()


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


def _identity_environment() -> tuple[dict[str, str], bool]:
    """Return the environment visible to this credential construction."""
    multiplexed = _is_multiplex_active()
    if not multiplexed:
        return (
            {
                name: value
                for name in _AZURE_IDENTITY_ENV_NAMES
                if (value := os.environ.get(name)) is not None
            },
            False,
        )

    scope = _current_secret_scope()
    if scope is None:
        raise _UnscopedSecretError(
            "Azure identity credential construction requires an active profile "
            "secret scope while multiplexing is enabled"
        )
    return (
        {
            name: str(value)
            for name in _AZURE_IDENTITY_ENV_NAMES
            if (value := scope.get(name)) is not None
        },
        True,
    )


def _credential_kwargs(
    config: EntraIdentityConfig,
    settings: dict[str, str],
    *,
    multiplexed: bool,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    if not config.exclude_interactive_browser:
        raise RuntimeError(
            "InteractiveBrowserCredential is unavailable in azure.identity.aio; "
            "native-async Azure identity cannot enable interactive browser auth"
        )
    if multiplexed:
        unsupported = sorted(
            name
            for name in _UNSUPPORTED_SCOPED_IDENTITY_ENV_NAMES
            if settings.get(name, "").strip()
        )
        if unsupported:
            names = ", ".join(unsupported)
            raise RuntimeError(
                "Profile-scoped Azure identity settings cannot be passed safely to "
                f"DefaultAzureCredential: {names}. azure-identity reads these values "
                "only from process-global os.environ; use a non-environment "
                "credential in multiplex mode or isolate this profile in its own "
                "process."
            )
    process_selector = os.environ.get("AZURE_TOKEN_CREDENTIALS", "").strip()
    if multiplexed and process_selector:
        raise RuntimeError(
            "Process-global AZURE_TOKEN_CREDENTIALS cannot safely select an "
            "Azure identity chain while profile multiplexing is enabled; unset "
            "it or isolate this profile in its own process."
        )

    selector = settings.get("AZURE_TOKEN_CREDENTIALS", "").strip().lower()
    if selector in _UNSUPPORTED_TOKEN_CREDENTIALS:
        raise RuntimeError(
            f"AZURE_TOKEN_CREDENTIALS={selector!r} selects an Azure credential "
            "that is not native-async and profile-safe in azure.identity.aio"
        )

    client_secret_configured = all(
        settings.get(name, "").strip() for name in _CLIENT_SECRET_ENV_NAMES
    )
    certificate_configured = all(
        settings.get(name, "").strip() for name in _CERTIFICATE_ENV_NAMES
    )
    if selector == "managedidentitycredential":
        selected = "exclude_managed_identity_credential"
    elif selector == "environmentcredential":
        if certificate_configured and not client_secret_configured:
            raise RuntimeError(
                "CertificateCredential performs blocking certificate-file reads "
                "and cannot be used by the native-async adapter"
            )
        selected = "exclude_environment_credential"
    elif selector:
        # Preserve azure-identity's actionable validation for unknown selector
        # values. It validates AZURE_TOKEN_CREDENTIALS before applying these
        # explicit exclusion flags.
        selected = "exclude_managed_identity_credential"
    elif client_secret_configured:
        selected = "exclude_environment_credential"
    elif certificate_configured:
        raise RuntimeError(
            "CertificateCredential performs blocking certificate-file reads and "
            "cannot be used by the native-async adapter"
        )
    elif settings.get("AZURE_FEDERATED_TOKEN_FILE", "").strip():
        raise RuntimeError(
            "WorkloadIdentityCredential in azure.identity.aio performs blocking "
            "token-file reads and cannot be used by the native-async adapter"
        )
    elif settings.get("AZURE_USERNAME", "").strip():
        raise RuntimeError(
            "SharedTokenCacheCredential performs blocking persistent-cache I/O "
            "and cannot be used by the native-async adapter"
        )
    else:
        selected = "exclude_managed_identity_credential"

    # The pinned async SDK's default chain contains blocking file/cache paths,
    # process-global developer identities, and omits upstream's broker entry.
    # Select only the one verified native credential rather than silently
    # falling through to an unsafe credential later in get_token().
    kwargs.update({name: name != selected for name in _DEFAULT_CREDENTIAL_EXCLUDES})
    if multiplexed:
        client_id = settings.get("AZURE_CLIENT_ID") or None
        kwargs.update(
            authority=settings.get("AZURE_AUTHORITY_HOST")
            or _AZURE_PUBLIC_CLOUD_AUTHORITY,
            managed_identity_client_id=client_id,
        )
    return kwargs


def _credential_fingerprint(
    settings: dict[str, str],
    kwargs: Dict[str, Any],
) -> str:
    digest = hashlib.sha256()
    for name in _AZURE_IDENTITY_ENV_NAMES:
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(settings.get(name, "").encode())
        digest.update(b"\0")
    for name, value in sorted(kwargs.items()):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(repr(value).encode())
        digest.update(b"\0")
    return digest.hexdigest()


async def _canonical_credential_home() -> str:
    expanduser = _aiofiles_os.wrap(os.path.expanduser)
    expanded = await expanduser(os.fspath(_get_hermes_home()))
    realpath = _aiofiles_os.wrap(os.path.realpath)
    return os.path.normcase(await realpath(expanded))


async def _finish_owned_task(task: asyncio.Task[Any]) -> Any:
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError as exc:  # noqa: ASYNC103 - re-raised below
            if task.cancelled():
                raise
            if cancellation is None:
                cancellation = exc
        except Exception as exc:
            if cancellation is not None:
                raise cancellation from exc
            raise
    if cancellation is not None:
        raise cancellation
    return result


async def _close_credentials(credentials: list[Any]) -> None:
    async def close_all() -> None:
        async def close_one(credential: Any) -> None:
            close = getattr(credential, "close", None)
            if close is not None:
                await close()

        results = await asyncio.gather(
            *(close_one(credential) for credential in credentials),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result

    if credentials:
        await _finish_owned_task(asyncio.create_task(close_all()))


async def build_credential(config: EntraIdentityConfig) -> Any:
    """Return the loop/profile-local async ``DefaultAzureCredential``."""
    loop = asyncio.get_running_loop()
    settings, multiplexed = _identity_environment()
    kwargs = _credential_kwargs(config, settings, multiplexed=multiplexed)
    key = (
        await _canonical_credential_home(),
        config,
        _credential_fingerprint(settings, kwargs),
    )
    with _credential_cache_guard:
        cache = _credential_caches.setdefault(loop, {})
        credential = cache.get(key)
        if credential is not None:
            return credential

    identity = _require_azure_identity()
    credential = identity.DefaultAzureCredential(**kwargs)
    with _credential_cache_guard:
        cache[key] = credential
    return credential


async def reset_credential_cache() -> None:
    """Close and clear this loop/profile's Azure credential chains."""
    loop = asyncio.get_running_loop()
    profile_home = await _canonical_credential_home()
    with _credential_cache_guard:
        cache = _credential_caches.get(loop, {})
        credentials = []
        seen: set[int] = set()
        for key, credential in tuple(cache.items()):
            if key[0] != profile_home:
                continue
            cache.pop(key, None)
            credential_id = id(credential)
            if credential_id not in seen:
                credentials.append(credential)
                seen.add(credential_id)
        if not cache:
            _credential_caches.pop(loop, None)

        for provider in tuple(_issued_providers):
            if id(getattr(provider, "_hermes_credential", None)) in seen:
                provider._hermes_released = True
                provider._hermes_credential_loop = None
                provider._hermes_credential = None
                provider._hermes_sdk_provider_cell[0] = None
                _issued_providers.discard(provider)
        for credential_id in seen:
            _credential_leases.pop(credential_id, None)
    await _close_credentials(credentials)


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
    owner_loop = asyncio.get_running_loop()
    owner_loop_ref = weakref.ref(owner_loop)
    sdk_provider = identity.get_bearer_token_provider(credential, config.scope)
    sdk_provider_cell: list[Any] = [sdk_provider]

    async def provider() -> str:
        if asyncio.get_running_loop() is not owner_loop_ref():
            raise RuntimeError(
                "Azure token providers must be used on the event loop that "
                "created them"
            )
        active_provider = sdk_provider_cell[0]
        if active_provider is None:
            raise RuntimeError("Azure token provider has been released")
        return await active_provider()

    managed_provider: Any = provider
    setattr(managed_provider, "_hermes_credential", credential)
    setattr(managed_provider, "_hermes_credential_loop", owner_loop)
    setattr(managed_provider, "_hermes_sdk_provider_cell", sdk_provider_cell)
    setattr(managed_provider, "_hermes_released", False)
    with _credential_cache_guard:
        _issued_providers.add(managed_provider)
        _credential_leases[id(credential)] = (
            _credential_leases.get(id(credential), 0) + 1
        )
    return managed_provider


async def _release_token_provider(value: Any) -> None:
    """Release one provider lease and close its credential after the last use."""
    credential = getattr(value, "_hermes_credential", None)
    owner_loop = getattr(value, "_hermes_credential_loop", None)
    with _credential_cache_guard:
        if credential is None or getattr(value, "_hermes_released", False):
            return
        if owner_loop is not asyncio.get_running_loop():
            raise RuntimeError(
                "Azure token providers must be released on the event loop that "
                "created them"
            )
        value._hermes_released = True
        value._hermes_credential_loop = None
        value._hermes_credential = None
        value._hermes_sdk_provider_cell[0] = None
        _issued_providers.discard(value)
        credential_id = id(credential)
        remaining = _credential_leases.get(credential_id, 1) - 1
        if remaining > 0:
            _credential_leases[credential_id] = remaining
            return
        _credential_leases.pop(credential_id, None)
        for loop, cache in tuple(_credential_caches.items()):
            for key, cached in tuple(cache.items()):
                if cached is credential:
                    cache.pop(key, None)
            if not cache:
                _credential_caches.pop(loop, None)
    await _close_credentials([credential])


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

    try:
        settings, _multiplexed = _identity_environment()
    except _UnscopedSecretError as exc:
        info["error"] = str(exc)
        return info

    if settings.get("AZURE_TENANT_ID", "").strip():
        info["tenant_id_env"] = settings["AZURE_TENANT_ID"].strip()
    env_sources = []
    if settings.get("AZURE_FEDERATED_TOKEN_FILE", "").strip():
        env_sources.append("WorkloadIdentityCredential (AZURE_FEDERATED_TOKEN_FILE)")
    if all(
        settings.get(name, "").strip()
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
