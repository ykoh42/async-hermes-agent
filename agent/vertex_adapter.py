"""Vertex AI (Google Cloud) adapter for Hermes Agent.

Provides authentication and configuration for Vertex AI's OpenAI-compatible
endpoint. This allows Hermes to use Gemini models via Google Cloud with
enterprise-grade rate limits and quotas.

Requires: pip install google-auth

Environment variables honored (all optional):
  GOOGLE_APPLICATION_CREDENTIALS — path to a service account JSON file (secret).
  VERTEX_CREDENTIALS_PATH        — alias, takes precedence if set (secret).
  VERTEX_PROJECT_ID              — override the project_id embedded in creds.
  VERTEX_REGION                  — override default region ("global" unless set).

Non-secret routing settings (project_id, region) also live in config.yaml
under the ``vertex:`` section; env vars take precedence over config.yaml.
"""

import asyncio
import json
import logging
import os
import time
import weakref
from typing import Optional, Tuple

import aiofiles
import aiofiles.os
import httpx

from agent.secret_scope import get_secret as _get_secret, is_multiplex_active

try:
    from google.auth import _cloud_sdk
    from google.auth.transport import aiohttp_requests
    from google.oauth2 import _credentials_async as user_credentials
    from google.oauth2 import _service_account_async as service_account
except ImportError:
    _cloud_sdk = None  # type: ignore[assignment]
    aiohttp_requests = None  # type: ignore[assignment]
    user_credentials = None  # type: ignore[assignment]
    service_account = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

DEFAULT_REGION = "global"

_creds_cache: dict = {}
_cache_locks: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock]" = (
    weakref.WeakKeyDictionary()
)


async def _finish_process_wait(process: asyncio.subprocess.Process) -> int:
    """Reap one owned gcloud process before propagating cancellation."""
    wait_task = asyncio.create_task(process.wait())
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            return_code = await asyncio.shield(wait_task)
            break
        except asyncio.CancelledError as exc:  # noqa: ASYNC103 - re-raised below
            if wait_task.cancelled():
                raise
            if cancellation is None:
                cancellation = exc
        except Exception as exc:
            if cancellation is not None:
                raise cancellation from exc
            raise
    if cancellation is not None:
        raise cancellation
    return return_code


async def _finish_process_communicate(
    process: asyncio.subprocess.Process,
    communicate_task: asyncio.Task[tuple[bytes | None, bytes | None]],
) -> tuple[bytes | None, bytes | None]:
    """Drain pipes and reap one owned gcloud process."""
    async def drain_or_wait() -> tuple[bytes | None, bytes | None]:
        try:
            return await communicate_task
        except BaseException:
            await _finish_process_wait(process)
            raise

    cleanup_task = asyncio.create_task(drain_or_wait())
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            output = await asyncio.shield(cleanup_task)
            break
        except asyncio.CancelledError as exc:  # noqa: ASYNC103 - re-raised below
            if cleanup_task.cancelled():
                raise
            if cancellation is None:
                cancellation = exc
        except Exception as exc:
            if cancellation is not None:
                raise cancellation from exc
            raise
    if cancellation is not None:
        raise cancellation
    return output


def _cache_lock() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    lock = _cache_locks.get(loop)
    if lock is None:
        lock = asyncio.Lock()
        _cache_locks[loop] = lock
    return lock


async def _vertex_config() -> dict:
    """Return the ``vertex:`` section of config.yaml, or {} on any failure.

    Non-secret routing settings (project_id, region) live in config.yaml per
    the .env-secrets-only rule. Env vars still take precedence — they are read
    directly at the call sites below, with config.yaml as the fallback.
    """
    try:
        from hermes_cli.config import load_config_readonly

        section = (await load_config_readonly()).get("vertex")
        return section if isinstance(section, dict) else {}
    except Exception:
        return {}


async def _resolve_region(explicit: Optional[str] = None) -> str:
    """Region precedence: explicit arg > VERTEX_REGION env > config.yaml > default."""
    if explicit:
        return explicit
    env_region = (_get_secret("VERTEX_REGION") or "").strip()
    if env_region:
        return env_region
    cfg_region = str((await _vertex_config()).get("region") or "").strip()
    return cfg_region or DEFAULT_REGION


async def _resolve_project_override() -> Optional[str]:
    """Project-ID override precedence: VERTEX_PROJECT_ID env > config.yaml.

    Returns None when neither is set (the credentials' embedded project_id
    is used in that case).
    """
    env_project = (_get_secret("VERTEX_PROJECT_ID") or "").strip()
    if env_project:
        return env_project
    cfg_project = str((await _vertex_config()).get("project_id") or "").strip()
    return cfg_project or None


async def _resolve_credentials_path(explicit: Optional[str]) -> Optional[str]:
    if explicit and await aiofiles.os.path.exists(explicit):
        return explicit
    # Routed through get_secret (not a raw os.environ read): in a multiplex
    # gateway serving several profiles from one process, os.environ reflects
    # whichever profile's .env happened to be loaded at boot, not the profile
    # the current turn belongs to. Reading it directly here would let one
    # profile mint Vertex tokens from — and get billed against — a different
    # profile's service-account file. See agent/secret_scope.py.
    for env_var in ("VERTEX_CREDENTIALS_PATH", "GOOGLE_APPLICATION_CREDENTIALS"):
        path = _get_secret(env_var)
        if path and await aiofiles.os.path.exists(path):
            return path
    return None


async def _refresh_credentials(creds) -> None:
    auth_req = aiohttp_requests.Request()
    try:
        await creds.refresh(auth_req)
    finally:
        await auth_req.close()


async def _load_credentials_file(path: str):
    async with aiofiles.open(path, encoding="utf-8") as credentials_file:
        info = json.loads(await credentials_file.read())

    credential_type = info.get("type")
    if credential_type == "service_account":
        creds = service_account.Credentials.from_service_account_info(
            info,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        return creds, info.get("project_id")
    if credential_type == "authorized_user":
        creds = user_credentials.Credentials.from_authorized_user_info(
            info,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        return creds, None
    raise ValueError(
        "Vertex async credentials must be a service_account or authorized_user JSON file"
    )


async def _gcloud_project_id() -> Optional[str]:
    project_id = (
        os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GCLOUD_PROJECT")
        or ""
    ).strip()
    if project_id:
        return project_id
    try:
        process = await asyncio.create_subprocess_exec(
            "gcloud.cmd" if os.name == "nt" else "gcloud",
            "config",
            "get",
            "project",
            "--format=value(core.project)",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        communicate_task = asyncio.create_task(process.communicate())
        try:
            stdout, _ = await asyncio.wait_for(
                asyncio.shield(communicate_task), timeout=5.0
            )
        except asyncio.CancelledError:
            if process.returncode is None:
                process.kill()
            await _finish_process_communicate(process, communicate_task)
            raise
        except TimeoutError:
            if process.returncode is None:
                process.kill()
            await _finish_process_communicate(process, communicate_task)
            return None
    except (FileNotFoundError, OSError):
        return None
    if process.returncode != 0:
        return None
    return stdout.decode("utf-8", errors="replace").strip() or None


async def _metadata_credentials() -> Tuple[Optional[str], Optional[str]]:
    metadata_host = os.getenv("GCE_METADATA_HOST", "metadata.google.internal").strip()
    base_url = f"http://{metadata_host}/computeMetadata/v1"
    headers = {"Metadata-Flavor": "Google"}
    timeout = httpx.Timeout(3.0)
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        project_response, token_response = await asyncio.gather(
            client.get(f"{base_url}/project/project-id", headers=headers),
            client.get(
                f"{base_url}/instance/service-accounts/default/token",
                headers=headers,
            ),
        )
        project_response.raise_for_status()
        token_response.raise_for_status()
        token_data = token_response.json()
    token = str(token_data.get("access_token") or "").strip()
    project_id = project_response.text.strip()
    if not token or not project_id:
        return None, None
    expires_in = max(int(token_data.get("expires_in") or 0), 0)
    _creds_cache["__metadata__"] = {
        "token": token,
        "project_id": project_id,
        "expires_at": time.time() + expires_in,
    }
    return token, project_id


async def get_vertex_credentials(credentials_path: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """Return a (fresh access_token, project_id) pair or (None, None) on failure.

    Caches the underlying Credentials object and refreshes it when within
    5 minutes of expiry, so repeated calls don't thrash the token endpoint.
    """
    if service_account is None:
        logger.warning("google-auth package not installed. Cannot use Vertex AI.")
        return None, None

    resolved_path = await _resolve_credentials_path(credentials_path)
    cache_key = resolved_path

    try:
        async with _cache_lock():
            if resolved_path is None:
                # google.auth.default() reads GOOGLE_APPLICATION_CREDENTIALS
                # straight from os.environ internally — it has no notion of
                # the profile secret scope. _resolve_credentials_path already
                # confirmed (via get_secret) that *this* profile doesn't
                # define the var, but python-dotenv's load_dotenv() mutates
                # os.environ at boot for whichever profile happened to load
                # first, so a raw os.environ read here can still pick up a
                # different profile's service-account path. Refuse rather
                # than silently authenticating under a stranger's identity.
                if is_multiplex_active() and os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
                    logger.warning(
                        "Vertex ADC skipped for this profile: "
                        "GOOGLE_APPLICATION_CREDENTIALS is set in the process "
                        "environment (from another profile's .env) but not in "
                        "this profile's own config. Set VERTEX_CREDENTIALS_PATH "
                        "in this profile's .env instead of relying on ADC."
                    )
                    return None, None
                adc_path = _cloud_sdk.get_application_default_credentials_path()
                if await aiofiles.os.path.isfile(adc_path):
                    resolved_path = adc_path
                    cache_key = resolved_path
                else:
                    cached_metadata = _creds_cache.get("__metadata__")
                    if (
                        isinstance(cached_metadata, dict)
                        and cached_metadata.get("expires_at", 0) - time.time() >= 300
                    ):
                        token = cached_metadata.get("token")
                        project_id = cached_metadata.get("project_id")
                    else:
                        token, project_id = await _metadata_credentials()
                    override_project = await _resolve_project_override()
                    return token, override_project or project_id

            cached = _creds_cache.get(cache_key)
            if cached is None:
                creds, project_id = await _load_credentials_file(resolved_path)
                if project_id is None and resolved_path == _cloud_sdk.get_application_default_credentials_path():
                    project_id = await _gcloud_project_id()
                _creds_cache[cache_key] = (creds, project_id)
            else:
                creds, project_id = cached

            needs_refresh = (
                not getattr(creds, "token", None)
                or getattr(creds, "expired", False)
                or (
                    getattr(creds, "expiry", None) is not None
                    and (creds.expiry.timestamp() - time.time()) < 300
                )
            )
            if needs_refresh:
                await _refresh_credentials(creds)

        override_project = await _resolve_project_override()
        if override_project:
            project_id = override_project

        return creds.token, project_id
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.error(f"Failed to resolve Vertex AI credentials: {e}")
        _creds_cache.pop(cache_key, None)

        # If ADC failed (e.g. expired refresh token), try the SA file
        # before giving up — it may have been added after initial startup.
        if credentials_path is None:
            sa_path = await _resolve_credentials_path(credentials_path)
            if sa_path and sa_path != cache_key:
                logger.info("ADC failed, retrying with service account: %s", sa_path)
                return await get_vertex_credentials(sa_path)

        return None, None


def build_vertex_base_url(project_id: str, region: str = DEFAULT_REGION) -> str:
    """Build the OpenAI-compatible base URL for Vertex AI.

    The `global` location uses a bare `aiplatform.googleapis.com` hostname,
    while regional locations use `{region}-aiplatform.googleapis.com`.
    Gemini 3.x preview models are only served via the global endpoint at
    the time of writing.
    """
    host = "aiplatform.googleapis.com" if region == "global" else f"{region}-aiplatform.googleapis.com"
    return f"https://{host}/v1beta1/projects/{project_id}/locations/{region}/endpoints/openapi"


async def get_vertex_config(
    credentials_path: Optional[str] = None,
    region: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve (access_token, base_url) for Vertex AI, or (None, None) on failure."""
    token, project_id = await get_vertex_credentials(credentials_path)
    if not token or not project_id:
        return None, None

    effective_region = await _resolve_region(region)
    base_url = build_vertex_base_url(project_id, effective_region)
    return token, base_url


async def has_vertex_credentials() -> bool:
    """Fast check for whether Vertex credentials appear configured.

    No network calls and no google-auth import — safe for provider
    auto-detection and setup-status display. True when either a service
    account JSON path is resolvable, or an explicit project ID is configured
    (env or config.yaml, implying ADC is intended).
    """
    if await _resolve_credentials_path(None):
        return True
    if await _resolve_project_override():
        return True
    return False
