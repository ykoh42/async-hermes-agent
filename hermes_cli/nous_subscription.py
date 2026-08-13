"""Helpers for Nous subscription managed-tool capabilities.

Only the feature-state surface consumed by the retained agent prompt lives in
this library build.  Upstream's interactive setup/default mutation commands
belong to the deliberately removed CLI product surface.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable

import aiofiles.os

from hermes_cli.config import (
    get_env_value_prefer_dotenv,
    load_config_readonly as load_config,
)
from hermes_cli.nous_account import (
    NousPortalAccountInfo,
    get_nous_portal_account_info,
)
from tools.managed_tool_gateway import is_managed_tool_gateway_ready
from utils import is_truthy_value
from tools.tool_backend_helpers import (
    fal_key_is_configured,
    has_direct_modal_credentials,
    normalize_browser_cloud_provider,
    normalize_modal_mode,
    resolve_modal_backend_state,
    resolve_openai_audio_api_key,
)


_DEFAULT_PLATFORM_TOOLSETS = {
    "cli": "hermes-cli",
}


async def _get_env_value(key: str) -> str | None:
    """Read a credential in upstream scope order without blocking on dotenv."""
    try:
        from agent.secret_scope import UnscopedSecretError, get_secret
    except Exception:
        import os

        value = os.environ.get(key)
    else:
        try:
            value = get_secret(key)
        except UnscopedSecretError:
            raise
        except Exception:
            import os

            value = os.environ.get(key)
    if value is not None:
        return value
    return await get_env_value_prefer_dotenv(key)

def _uses_gateway(section: object) -> bool:
    """Return True when a config section explicitly opts into the gateway."""
    if not isinstance(section, dict):
        return False
    return is_truthy_value(section.get("use_gateway"), default=False)


@dataclass(frozen=True)
class NousFeatureState:
    key: str
    label: str
    included_by_default: bool
    available: bool
    active: bool
    managed_by_nous: bool
    direct_override: bool
    toolset_enabled: bool
    current_provider: str = ""
    explicit_configured: bool = False


@dataclass(frozen=True)
class NousSubscriptionFeatures:
    subscribed: bool
    nous_auth_present: bool
    provider_is_nous: bool
    features: dict[str, NousFeatureState]
    account_info: NousPortalAccountInfo | None = None

    @property
    def web(self) -> NousFeatureState:
        return self.features["web"]

    @property
    def image_gen(self) -> NousFeatureState:
        return self.features["image_gen"]

    @property
    def tts(self) -> NousFeatureState:
        return self.features["tts"]

    @property
    def stt(self) -> NousFeatureState:
        return self.features["stt"]

    @property
    def browser(self) -> NousFeatureState:
        return self.features["browser"]

    @property
    def video_gen(self) -> NousFeatureState:
        return self.features["video_gen"]

    @property
    def modal(self) -> NousFeatureState:
        return self.features["modal"]

    def items(self) -> Iterable[NousFeatureState]:
        ordered = ("web", "image_gen", "video_gen", "tts", "stt", "browser", "modal")
        for key in ordered:
            yield self.features[key]


def _model_config_dict(config: dict[str, object]) -> dict[str, object]:
    model_cfg = config.get("model")
    if isinstance(model_cfg, dict):
        return dict(model_cfg)
    if isinstance(model_cfg, str) and model_cfg.strip():
        return {"default": model_cfg.strip()}
    return {}


def _toolset_enabled(config: dict[str, object], toolset_key: str) -> bool:
    from toolsets import resolve_toolset

    platform_toolsets = config.get("platform_toolsets")
    if not isinstance(platform_toolsets, dict) or not platform_toolsets:
        platform_toolsets = {"cli": [_DEFAULT_PLATFORM_TOOLSETS["cli"]]}

    target_tools = set(resolve_toolset(toolset_key))
    if not target_tools:
        return False

    for platform, raw_toolsets in platform_toolsets.items():
        if isinstance(raw_toolsets, list):
            toolset_names = list(raw_toolsets)
        else:
            default_toolset = _DEFAULT_PLATFORM_TOOLSETS.get(platform)
            toolset_names = [default_toolset] if default_toolset else []
        if not toolset_names:
            default_toolset = _DEFAULT_PLATFORM_TOOLSETS.get(platform)
            if default_toolset:
                toolset_names = [default_toolset]

        available_tools: set[str] = set()
        for toolset_name in toolset_names:
            if not isinstance(toolset_name, str) or not toolset_name:
                continue
            try:
                available_tools.update(resolve_toolset(toolset_name))
            except Exception:
                continue

        if target_tools and target_tools.issubset(available_tools):
            return True

    return False


async def _has_agent_browser() -> bool:
    import shutil

    from hermes_constants import agent_browser_runnable, with_hermes_node_path

    # Validate the resolved binary actually runs — a dangling global symlink
    # (issue #48521) is reported by ``which`` but fails at exec. Fall through to
    # the local node_modules copy, which the validator also checks.
    which = aiofiles.os.wrap(shutil.which)
    if await agent_browser_runnable(await which("agent-browser")):
        return True

    # Hermes-managed Node dirs (Windows installer / POSIX $HERMES_HOME/node)
    # are prepended to PATH at runtime but usually absent from the *probe*
    # process's PATH — the same rung `_find_agent_browser` searches. Without
    # it a successful install keeps reporting "needs setup" on Windows.
    managed_path = (await with_hermes_node_path()).get("PATH", "")
    if managed_path:
        managed_hit = await which("agent-browser", path=managed_path)
        if managed_hit and await agent_browser_runnable(managed_hit):
            return True

    # Local node_modules/.bin: resolve via PATHEXT-aware ``shutil.which`` so
    # Windows picks the executable ``.cmd`` shim. Probing the extensionless
    # POSIX shim directly fails exec (WinError 193) even right after a
    # successful ``npm install`` — the bug that pinned every browser row on
    # "Setup required" in the desktop GUI.
    local_bin_dir = Path(__file__).parent.parent / "node_modules" / ".bin"
    if await aiofiles.os.path.isdir(local_bin_dir):
        local_which = await which("agent-browser", path=str(local_bin_dir))
        if local_which and await agent_browser_runnable(local_which):
            return True
    return False


async def _local_browser_runnable() -> bool:
    """Return True when the *local* browser backend would actually start.

    The ``agent-browser`` CLI being present is necessary but not sufficient for
    local mode: agent-browser also needs a Chromium build on disk (without one
    it hangs on first use until the command timeout fires), unless the
    Lightpanda engine is selected — text-only navigation needs no Chromium.

    This mirrors the local-mode tail of
    :func:`tools.browser_tool.check_browser_requirements`, so the setup/status
    surfaces advertise local browser readiness only when the runtime would
    actually run it. Cloud providers (Browserbase, Browser Use, Firecrawl) host
    their own Chromium and therefore gate on :func:`_has_agent_browser` alone.
    """
    if not await _has_agent_browser():
        return False
    try:
        from tools.browser_tool import _chromium_installed, _using_lightpanda_engine
    except Exception:
        # If the runtime probe can't be imported, fall back to binary presence
        # (prior behaviour) rather than crashing the setup/status surface.
        return True
    if await _using_lightpanda_engine():
        return True
    return await _chromium_installed()


def _browser_label(current_provider: str) -> str:
    mapping = {
        "browserbase": "Browserbase",
        "browser-use": "Browser Use",
        "firecrawl": "Firecrawl",
        "camofox": "Camofox",
        "local": "Local browser",
    }
    return mapping.get(current_provider or "local", current_provider or "Local browser")


def _tts_label(current_provider: str) -> str:
    mapping = {
        "openai": "OpenAI TTS",
        "elevenlabs": "ElevenLabs",
        "edge": "Edge TTS",
        "xai": "xAI TTS",
        "mistral": "Mistral Voxtral TTS",
        "neutts": "NeuTTS",
    }
    return mapping.get(current_provider or "edge", current_provider or "Edge TTS")


def _stt_label(current_provider: str) -> str:
    mapping = {
        "openai": "OpenAI Whisper",
        "groq": "Groq Whisper",
        "mistral": "Mistral Voxtral Transcribe",
        "local": "Local faster-whisper",
    }
    return mapping.get(current_provider or "local", current_provider or "Local faster-whisper")


def _resolve_browser_feature_state(
    *,
    browser_tool_enabled: bool,
    browser_provider: str,
    browser_provider_explicit: bool,
    browser_local_available: bool,
    browser_local_runnable: bool,
    direct_camofox: bool,
    direct_browserbase: bool,
    direct_browser_use: bool,
    direct_firecrawl: bool,
    managed_browser_available: bool,
) -> tuple[str, bool, bool, bool]:
    """Resolve browser availability using the same precedence as runtime.

    ``browser_local_available`` means "the agent-browser CLI is present" — the
    only local requirement for cloud providers, which host their own Chromium.
    ``browser_local_runnable`` additionally requires a usable local Chromium
    build (or the Lightpanda engine), mirroring the local-mode tail of
    :func:`tools.browser_tool.check_browser_requirements`. Local mode must gate
    on the latter, or setup/status advertise a browser that fails on first use
    when Chromium is missing.
    """
    if direct_camofox:
        return "camofox", True, bool(browser_tool_enabled), False

    if browser_provider_explicit:
        current_provider = browser_provider or "local"
        if current_provider == "browserbase":
            available = bool(browser_local_available and direct_browserbase)
            active = bool(browser_tool_enabled and available)
            return current_provider, available, active, False
        if current_provider == "browser-use":
            provider_available = managed_browser_available or direct_browser_use
            available = bool(browser_local_available and provider_available)
            managed = bool(
                browser_tool_enabled
                and browser_local_available
                and managed_browser_available
                and not direct_browser_use
            )
            active = bool(browser_tool_enabled and available)
            return current_provider, available, active, managed
        if current_provider == "firecrawl":
            available = bool(browser_local_available and direct_firecrawl)
            active = bool(browser_tool_enabled and available)
            return current_provider, available, active, False
        if current_provider == "camofox":
            return current_provider, False, False, False

        current_provider = "local"
        available = bool(browser_local_runnable)
        active = bool(browser_tool_enabled and available)
        return current_provider, available, active, False

    if managed_browser_available or direct_browser_use:
        available = bool(browser_local_available)
        managed = bool(
            browser_tool_enabled
            and browser_local_available
            and managed_browser_available
            and not direct_browser_use
        )
        active = bool(browser_tool_enabled and available)
        return "browser-use", available, active, managed

    if direct_browserbase:
        available = bool(browser_local_available)
        active = bool(browser_tool_enabled and available)
        return "browserbase", available, active, False

    available = bool(browser_local_runnable)
    active = bool(browser_tool_enabled and available)
    return "local", available, active, False


async def get_nous_subscription_features(
    config: dict[str, object] | None = None,
    *,
    force_fresh: bool = False,
) -> NousSubscriptionFeatures:
    if config is None:
        config = await load_config() or {}
    config = dict(config)
    model_cfg = _model_config_dict(config)
    provider_is_nous = str(model_cfg.get("provider") or "").strip().lower() == "nous"

    try:
        if force_fresh:
            account_info = await get_nous_portal_account_info(force_fresh=True)
        else:
            account_info = await get_nous_portal_account_info()
    except Exception:
        account_info = None

    # Coarse "entitled to any managed tool" gate: paid access OR a live free
    # tool pool. Per-backend availability is then narrowed by coverage below
    # (the pool funds image but not video, etc.).
    managed_tools_flag = bool(
        account_info
        and account_info.logged_in
        and account_info.tool_gateway_entitled
    )
    nous_auth_present = bool(account_info and account_info.logged_in)

    def _entitled_for(category: str) -> bool:
        return bool(account_info and account_info.tool_gateway_entitled_for(category))
    subscribed = provider_is_nous or nous_auth_present

    web_tool_enabled = _toolset_enabled(config, "web")
    image_tool_enabled = _toolset_enabled(config, "image_gen")
    video_tool_enabled = _toolset_enabled(config, "video_gen")
    tts_tool_enabled = _toolset_enabled(config, "tts")
    browser_tool_enabled = _toolset_enabled(config, "browser")
    modal_tool_enabled = _toolset_enabled(config, "terminal")

    web_cfg = config.get("web") if isinstance(config.get("web"), dict) else {}
    tts_cfg = config.get("tts") if isinstance(config.get("tts"), dict) else {}
    stt_cfg = config.get("stt") if isinstance(config.get("stt"), dict) else {}
    browser_cfg = config.get("browser") if isinstance(config.get("browser"), dict) else {}
    terminal_cfg = config.get("terminal") if isinstance(config.get("terminal"), dict) else {}

    web_backend = str(web_cfg.get("backend") or "").strip().lower()
    # Per-capability overrides: if set, they determine which backend is active for
    # search/extract independently of web.backend.
    web_search_backend = str(web_cfg.get("search_backend") or "").strip().lower()
    tts_provider = str(tts_cfg.get("provider") or "edge").strip().lower()
    # STT default is "local" (faster-whisper) per DEFAULT_CONFIG, which
    # requires `pip install faster-whisper`. For Nous subscribers we'd
    # rather route through the managed OpenAI audio gateway — see
    # apply_nous_managed_defaults below.
    stt_provider = str(stt_cfg.get("provider") or "local").strip().lower()
    browser_provider_explicit = "cloud_provider" in browser_cfg
    browser_provider = normalize_browser_cloud_provider(
        browser_cfg.get("cloud_provider") if browser_provider_explicit else None
    )
    terminal_backend = (
        str(terminal_cfg.get("backend") or "local").strip().lower()
    )
    modal_mode = normalize_modal_mode(
        terminal_cfg.get("modal_mode")
    )

    # use_gateway flags — when True, the user explicitly opted into the
    # Tool Gateway via `hermes model`, so direct credentials should NOT
    # prevent gateway routing.
    web_use_gateway = _uses_gateway(web_cfg)
    tts_use_gateway = _uses_gateway(tts_cfg)
    stt_use_gateway = _uses_gateway(stt_cfg)
    browser_use_gateway = _uses_gateway(browser_cfg)
    image_gen_cfg = config.get("image_gen") if isinstance(config.get("image_gen"), dict) else {}
    image_use_gateway = _uses_gateway(image_gen_cfg)
    video_gen_cfg = config.get("video_gen") if isinstance(config.get("video_gen"), dict) else {}
    video_use_gateway = _uses_gateway(video_gen_cfg)

    direct_exa = bool(await _get_env_value("EXA_API_KEY"))
    direct_firecrawl = bool(
        await _get_env_value("FIRECRAWL_API_KEY")
        or await _get_env_value("FIRECRAWL_API_URL")
    )
    direct_parallel = bool(await _get_env_value("PARALLEL_API_KEY"))
    direct_tavily = bool(await _get_env_value("TAVILY_API_KEY"))
    direct_searxng = bool(await _get_env_value("SEARXNG_URL"))
    direct_fal = await fal_key_is_configured()
    direct_fal_video = direct_fal  # same FAL_KEY; separate var so use_gateway is independent
    direct_openai_tts = bool(await resolve_openai_audio_api_key())
    direct_elevenlabs = bool(await _get_env_value("ELEVENLABS_API_KEY"))
    direct_camofox = bool(await _get_env_value("CAMOFOX_URL"))
    direct_browserbase = bool(
        await _get_env_value("BROWSERBASE_API_KEY")
        and await _get_env_value("BROWSERBASE_PROJECT_ID")
    )
    direct_browser_use = bool(await _get_env_value("BROWSER_USE_API_KEY"))
    direct_modal = await has_direct_modal_credentials()

    # STT direct providers. OpenAI Whisper reuses the same audio key as
    # OpenAI TTS — resolve_openai_audio_api_key() reads VOICE_TOOLS_OPENAI_KEY
    # and falls back to OPENAI_API_KEY. The local provider's "direct"
    # signal is whether faster-whisper is importable; we lazy-import so
    # this module stays cheap on the happy path.
    direct_openai_stt = bool(await resolve_openai_audio_api_key())
    direct_groq_stt = bool(await _get_env_value("GROQ_API_KEY"))
    direct_mistral_stt = bool(await _get_env_value("MISTRAL_API_KEY"))
    transcription_tools = sys.modules.get("tools.transcription_tools")
    local_stt_available = bool(
        getattr(transcription_tools, "_HAS_FASTER_WHISPER", False)
    ) or bool(await _get_env_value("HERMES_LOCAL_STT_COMMAND"))

    # When use_gateway is set, suppress direct credentials for managed detection
    if web_use_gateway:
        direct_firecrawl = False
        direct_exa = False
        direct_parallel = False
        direct_tavily = False
    if image_use_gateway:
        direct_fal = False
    if video_use_gateway:
        direct_fal_video = False
    if tts_use_gateway:
        direct_openai_tts = False
        direct_elevenlabs = False
    if stt_use_gateway:
        direct_openai_stt = False
        direct_groq_stt = False
        direct_mistral_stt = False
        local_stt_available = False
    if browser_use_gateway:
        direct_browser_use = False
        direct_browserbase = False

    managed_web_available = (
        managed_tools_flag
        and nous_auth_present
        and await is_managed_tool_gateway_ready("firecrawl")
        and _entitled_for("firecrawl")
    )
    managed_image_available = (
        managed_tools_flag
        and nous_auth_present
        and await is_managed_tool_gateway_ready("fal-queue")
        and _entitled_for("fal")
    )
    # Video gen rides the same fal-queue gateway as image gen, but the free tool
    # pool funds image and NOT video — so gate it on its own coverage category
    # rather than aliasing it to image. (Paid users are entitled to both.)
    managed_video_available = (
        managed_tools_flag
        and nous_auth_present
        and await is_managed_tool_gateway_ready("fal-queue")
        and _entitled_for("fal-video")
    )
    managed_tts_available = (
        managed_tools_flag
        and nous_auth_present
        and await is_managed_tool_gateway_ready("openai-audio")
        and _entitled_for("openai-audio")
    )
    # STT and TTS share the same managed gateway endpoint ("openai-audio")
    # because the OpenAI audio API covers both /audio/speech (TTS) and
    # /audio/transcriptions (STT). One probe (and one entitlement), used by both.
    managed_stt_available = managed_tts_available
    managed_browser_available = (
        managed_tools_flag
        and nous_auth_present
        and await is_managed_tool_gateway_ready("browser-use")
        and _entitled_for("browser-use")
    )
    managed_modal_available = (
        managed_tools_flag
        and nous_auth_present
        and await is_managed_tool_gateway_ready("modal")
        and _entitled_for("modal")
    )
    modal_state = resolve_modal_backend_state(
        modal_mode,
        has_direct=direct_modal,
        managed_ready=managed_modal_available,
        managed_enabled=managed_tools_flag,
    )

    web_managed = web_backend == "firecrawl" and managed_web_available and not direct_firecrawl
    web_active = bool(
        web_tool_enabled
        and (
            web_managed
            or (web_backend == "exa" and direct_exa)
            or (web_backend == "firecrawl" and direct_firecrawl)
            or (web_backend == "parallel" and direct_parallel)
            or (web_backend == "tavily" and direct_tavily)
            or (web_backend == "searxng" and direct_searxng)
            # Per-capability overrides: search_backend or extract_backend may be set
            # without web.backend (using the new split config from #20061)
            or (web_search_backend == "searxng" and direct_searxng)
            or (web_search_backend == "exa" and direct_exa)
            or (web_search_backend == "firecrawl" and direct_firecrawl)
            or (web_search_backend == "parallel" and direct_parallel)
            or (web_search_backend == "tavily" and direct_tavily)
        )
    )
    web_available = bool(
        managed_web_available or direct_exa or direct_firecrawl or direct_parallel or direct_tavily or direct_searxng
    )

    image_managed = image_tool_enabled and managed_image_available and not direct_fal
    image_active = bool(image_tool_enabled and (image_managed or direct_fal))
    image_available = bool(managed_image_available or direct_fal)

    video_managed = video_tool_enabled and managed_video_available and not direct_fal_video
    video_active = bool(video_tool_enabled and (video_managed or direct_fal_video))
    video_available = bool(managed_video_available or direct_fal_video)

    tts_current_provider = tts_provider or "edge"
    tts_managed = (
        tts_tool_enabled
        and tts_current_provider == "openai"
        and managed_tts_available
        and not direct_openai_tts
    )
    tts_available = bool(
        tts_current_provider in {"edge", "neutts"}
        or (tts_current_provider == "openai" and (managed_tts_available or direct_openai_tts))
        or (tts_current_provider == "elevenlabs" and direct_elevenlabs)
        or (
            tts_current_provider == "mistral"
            and bool(await _get_env_value("MISTRAL_API_KEY"))
        )
    )
    tts_active = bool(tts_tool_enabled and tts_available)

    # STT availability per provider. Unlike TTS, STT isn't a model-callable
    # tool — the gateway voice middleware calls it on every inbound voice
    # message — so toolset_enabled is N/A and we treat stt as always
    # "enabled" if a usable provider is configured.
    stt_current_provider = stt_provider or "local"
    stt_managed = (
        stt_current_provider == "openai"
        and managed_stt_available
        and not direct_openai_stt
    )
    stt_available = bool(
        (stt_current_provider == "local" and local_stt_available)
        or (stt_current_provider == "openai" and (managed_stt_available or direct_openai_stt))
        or (stt_current_provider == "groq" and direct_groq_stt)
        or (stt_current_provider == "mistral" and direct_mistral_stt)
    )
    stt_active = stt_available

    browser_local_available = await _has_agent_browser()
    browser_local_runnable = await _local_browser_runnable()
    (
        browser_current_provider,
        browser_available,
        browser_active,
        browser_managed,
    ) = _resolve_browser_feature_state(
        browser_tool_enabled=browser_tool_enabled,
        browser_provider=browser_provider,
        browser_provider_explicit=browser_provider_explicit,
        browser_local_available=browser_local_available,
        browser_local_runnable=browser_local_runnable,
        direct_camofox=direct_camofox,
        direct_browserbase=direct_browserbase,
        direct_browser_use=direct_browser_use,
        direct_firecrawl=direct_firecrawl,
        managed_browser_available=managed_browser_available,
    )

    if terminal_backend != "modal":
        modal_managed = False
        modal_available = True
        modal_active = bool(modal_tool_enabled)
        modal_direct_override = False
    elif modal_state["selected_backend"] == "managed":
        modal_managed = bool(modal_tool_enabled)
        modal_available = True
        modal_active = bool(modal_tool_enabled)
        modal_direct_override = False
    elif modal_state["selected_backend"] == "direct":
        modal_managed = False
        modal_available = True
        modal_active = bool(modal_tool_enabled)
        modal_direct_override = bool(modal_tool_enabled)
    elif modal_mode == "managed":
        modal_managed = False
        modal_available = bool(managed_modal_available)
        modal_active = False
        modal_direct_override = False
    elif modal_mode == "direct":
        modal_managed = False
        modal_available = bool(direct_modal)
        modal_active = False
        modal_direct_override = False
    else:
        modal_managed = False
        modal_available = bool(managed_modal_available or direct_modal)
        modal_active = False
        modal_direct_override = False

    tts_explicit_configured = False
    raw_tts_cfg = config.get("tts")
    if isinstance(raw_tts_cfg, dict) and "provider" in raw_tts_cfg:
        tts_explicit_configured = tts_provider not in {"", "edge"}

    # STT considers any non-default provider explicit. "local" is the
    # DEFAULT_CONFIG seed, so seeing it doesn't mean the user picked it.
    stt_explicit_configured = False
    raw_stt_cfg = config.get("stt")
    if isinstance(raw_stt_cfg, dict) and "provider" in raw_stt_cfg:
        stt_explicit_configured = stt_provider not in {"", "local"}

    features = {
        "web": NousFeatureState(
            key="web",
            label="Web tools",
            included_by_default=True,
            available=web_available,
            active=web_active,
            managed_by_nous=web_managed,
            direct_override=web_active and not web_managed,
            toolset_enabled=web_tool_enabled,
            current_provider=web_backend or web_search_backend or "",
            explicit_configured=bool(web_backend or web_search_backend),
        ),
        "image_gen": NousFeatureState(
            key="image_gen",
            label="Image generation",
            included_by_default=True,
            available=image_available,
            active=image_active,
            managed_by_nous=image_managed,
            direct_override=image_active and not image_managed,
            toolset_enabled=image_tool_enabled,
            current_provider="FAL" if direct_fal else ("Nous Subscription" if image_managed else ""),
            explicit_configured=direct_fal,
        ),
        "video_gen": NousFeatureState(
            key="video_gen",
            label="Video generation",
            included_by_default=False,
            available=video_available,
            active=video_active,
            managed_by_nous=video_managed,
            direct_override=video_active and not video_managed,
            toolset_enabled=video_tool_enabled,
            current_provider="FAL" if direct_fal_video else ("Nous Subscription" if video_managed else ""),
            explicit_configured=direct_fal_video,
        ),
        "tts": NousFeatureState(
            key="tts",
            label="OpenAI TTS",
            included_by_default=True,
            available=tts_available,
            active=tts_active,
            managed_by_nous=tts_managed,
            direct_override=tts_active and not tts_managed,
            toolset_enabled=tts_tool_enabled,
            current_provider=_tts_label(tts_current_provider),
            explicit_configured=tts_explicit_configured,
        ),
        "stt": NousFeatureState(
            key="stt",
            label="Speech-to-text",
            included_by_default=True,
            available=stt_available,
            active=stt_active,
            managed_by_nous=stt_managed,
            direct_override=stt_active and not stt_managed,
            # STT isn't toolset-gated (gateway middleware calls it
            # unconditionally on inbound voice), so report True so the
            # status display doesn't flag it as "tool disabled".
            toolset_enabled=True,
            current_provider=_stt_label(stt_current_provider),
            explicit_configured=stt_explicit_configured,
        ),
        "browser": NousFeatureState(
            key="browser",
            label="Browser automation",
            included_by_default=True,
            available=browser_available,
            active=browser_active,
            managed_by_nous=browser_managed,
            direct_override=browser_active and not browser_managed,
            toolset_enabled=browser_tool_enabled,
            current_provider=_browser_label(browser_current_provider),
            explicit_configured=browser_provider_explicit,
        ),
        "modal": NousFeatureState(
            key="modal",
            label="Modal execution",
            included_by_default=False,
            available=modal_available,
            active=modal_active,
            managed_by_nous=modal_managed,
            direct_override=terminal_backend == "modal" and modal_direct_override,
            toolset_enabled=modal_tool_enabled,
            current_provider="Modal" if terminal_backend == "modal" else terminal_backend or "local",
            explicit_configured=terminal_backend == "modal",
        ),
    }

    return NousSubscriptionFeatures(
        subscribed=subscribed,
        nous_auth_present=nous_auth_present,
        provider_is_nous=provider_is_nous,
        features=features,
        account_info=account_info,
    )
