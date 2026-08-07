"""Native Anthropic provider profile."""

import logging

import httpx

from providers import register_provider
from providers.base import ProviderProfile

logger = logging.getLogger(__name__)


class AnthropicProfile(ProviderProfile):
    """Native Anthropic — uses x-api-key header, not Bearer."""

    async def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Anthropic uses x-api-key header and anthropic-version."""
        if not api_key:
            return None

        from hermes_cli.urllib_security import open_credentialed_url

        try:
            response = await open_credentialed_url(
                httpx.Request(
                    "GET",
                    "https://api.anthropic.com/v1/models",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "Accept": "application/json",
                    },
                ),
                timeout=timeout,
            )
            data = response.json()
            return [
                model["id"]
                for model in data.get("data", [])
                if isinstance(model, dict) and "id" in model
            ]
        except Exception as exc:
            logger.debug("fetch_models(anthropic): %s", exc)
            return None


anthropic = AnthropicProfile(
    name="anthropic",
    aliases=("claude", "claude-oauth", "claude-code"),
    api_mode="anthropic_messages",
    env_vars=("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"),
    base_url="https://api.anthropic.com",
    auth_type="api_key",
    default_aux_model="claude-haiku-4-5-20251001",
)

register_provider(anthropic)
