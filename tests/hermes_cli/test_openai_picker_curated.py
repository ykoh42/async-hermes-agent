"""Regression tests for two OpenAI/OpenRouter model-picker bugs.

Bug 1 — OpenAI picker dumped the raw ``/v1/models`` catalog
    ``provider_model_ids("openai")`` hit ``api.openai.com/v1/models`` and
    returned the full 120+ entry catalog (embeddings, whisper, tts, dall-e,
    moderation, gpt-3.5, …). The ``hermes model`` CLI shows only the curated
    agentic list. The picker now intersects the live default-endpoint catalog
    with the curated list (preserving curated order) so both surfaces match.
    Custom OpenAI-compatible endpoints (proxies, gateways) keep the live list
    verbatim so discovery still works.

Bug 2 — OpenRouter appeared authenticated whenever OPENAI_API_KEY was set
    OpenRouter's HermesOverlay carried ``extra_env_vars=("OPENAI_API_KEY",)``.
    ``list_authenticated_providers`` reads ``extra_env_vars`` to decide whether
    a provider has credentials, so any OpenAI user saw a phantom OpenRouter
    row. The overlay entry is removed; runtime credential resolution still
    falls back to OPENAI_API_KEY for explicitly-selected OpenRouter (handled
    in runtime_provider.py, independent of the overlay).
"""

from hermes_cli.providers import HERMES_OVERLAYS


# --- Bug 2: overlay no longer lists OPENAI_API_KEY --------------------------

def test_openrouter_overlay_does_not_list_openai_api_key():
    overlay = HERMES_OVERLAYS["openrouter"]
    assert "OPENAI_API_KEY" not in overlay.extra_env_vars
