"""Meta Model API (Muse Spark) provider profile."""

from __future__ import annotations

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


def _resolve_effort(reasoning_config: dict | None) -> str:
    """Map Hermes reasoning settings to Meta's accepted effort values."""
    config = reasoning_config or {}
    if config.get("enabled") is False:
        return "minimal"
    effort = str(config.get("effort") or "").strip().lower()
    if effort == "none":
        return "minimal"
    if effort in {"low", "medium", "high"}:
        return effort
    if effort in {"max", "xhigh", "ultra"}:
        return "xhigh"
    return "medium"


class MetaAIProfile(ProviderProfile):
    """Meta Model API profile with top-level reasoning_effort."""

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        supports_reasoning: bool = False,  # noqa: ARG002
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return {}, {"reasoning_effort": _resolve_effort(reasoning_config)}


meta_ai = MetaAIProfile(
    name="meta-ai",
    aliases=("meta", "muse", "muse-spark", "model-api", "msl"),
    display_name="Meta Model API",
    description="Meta Muse Spark family (Meta Superintelligence Labs)",
    signup_url="https://developer.meta.com/ai/",
    env_vars=("MODEL_API_KEY", "META_API_KEY", "META_MODEL_API_KEY", "META_BASE_URL"),
    # Do not read META_BASE_URL at import time: provider discovery can run in a
    # multiplexed profile without a bound secret scope. Runtime config still
    # resolves the declared env var for the selected profile.
    base_url="https://api.meta.ai/v1",
    auth_type="api_key",
    api_mode="codex_responses",
    supports_vision=True,
    default_aux_model="muse-spark-1.2-contributor",
    default_max_tokens=16384,
    fallback_models=("muse-spark-1.2", "muse-spark-1.2-contributor"),
)

register_provider(meta_ai)
