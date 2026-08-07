"""``canonical_custom_identity`` must return the durable config-key identity.

A keyed ``providers:`` entry's identity is its config key, not its display
name — ``custom_provider_slug`` encodes that, and the endpoint- and
model-based recovery sources both honour it. The configured-provider fallback
built its slug from whatever string the caller had, so a display name that
differs from its key healed to ``custom:<display-name>``: a second identity
for the same endpoint that no longer matches what persistence and routing
store.

Both spellings match the entry (``_get_named_custom_provider`` accepts
either), so the test asserts they converge on one identity rather than
asserting any particular spelling is rejected.
"""

from __future__ import annotations

import pytest

from hermes_cli import runtime_provider as rp

PROVIDER_KEY = "my-endpoint"
DISPLAY_NAME = "My Endpoint Display"
BASE_URL = "https://example.invalid/v1"
MODEL = "cool-model-1"

CANONICAL = f"custom:{PROVIDER_KEY}"


@pytest.fixture
def keyed_provider_config():
    """A ``providers:`` entry whose display name differs from its config key."""
    config = {
        "providers": {
            PROVIDER_KEY: {
                "name": DISPLAY_NAME,
                "api": BASE_URL,
                "api_key": "sk-test",
                "default_model": MODEL,
                "models": [MODEL],
            }
        }
    }
    return config


@pytest.fixture
def use_config(monkeypatch):
    def configure(config):
        async def load_config():
            return config

        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly",
            load_config,
        )

    return configure


@pytest.mark.asyncio
async def test_display_name_heals_to_the_config_key_identity(
    keyed_provider_config,
    use_config,
):
    """The regression: the display-name spelling must not mint a second identity."""
    use_config(keyed_provider_config)
    assert await rp.canonical_custom_identity(
        config_provider=DISPLAY_NAME
    ) == CANONICAL


@pytest.mark.asyncio
async def test_config_model_provider_display_name_heals_too(
    keyed_provider_config,
    use_config,
):
    """Same path reached through ``config.model.provider`` rather than an argument."""
    keyed_provider_config["model"] = {"provider": DISPLAY_NAME}
    use_config(keyed_provider_config)
    assert await rp.canonical_custom_identity() == CANONICAL


@pytest.mark.asyncio
async def test_config_key_spelling_still_resolves(
    keyed_provider_config,
    use_config,
):
    """The spelling that already worked keeps working."""
    use_config(keyed_provider_config)
    assert await rp.canonical_custom_identity(
        config_provider=PROVIDER_KEY
    ) == CANONICAL


@pytest.mark.asyncio
async def test_all_recovery_sources_agree_on_one_identity(
    keyed_provider_config,
    use_config,
):
    """Endpoint, model and configured-provider recovery must not disagree.

    Three sources feeding the same session-identity slot is only safe while
    they agree; a divergent one silently splits an endpoint in two.
    """
    use_config(keyed_provider_config)
    by_url = await rp.canonical_custom_identity(base_url=BASE_URL)
    by_model = await rp.canonical_custom_identity(model=MODEL)
    by_config = await rp.canonical_custom_identity(config_provider=DISPLAY_NAME)

    assert {by_url, by_model, by_config} == {CANONICAL}


@pytest.mark.asyncio
async def test_unconfigured_candidate_still_returns_none(
    keyed_provider_config,
    use_config,
):
    """Fail-closed contract: never invent an identity resolution can't honour."""
    use_config(keyed_provider_config)
    assert await rp.canonical_custom_identity(
        config_provider="not-a-configured-entry"
    ) is None


@pytest.mark.asyncio
async def test_legacy_unkeyed_entry_keeps_its_name_identity(use_config):
    """``custom_providers:`` entries have no key, so the name stays the identity."""
    config = {
        "custom_providers": [
            {
                "name": "Legacy Endpoint",
                "base_url": "https://legacy.invalid/v1",
                "api_key": "sk-legacy",
                "models": ["legacy-model"],
            }
        ]
    }
    use_config(config)
    assert await rp.canonical_custom_identity(
        config_provider="Legacy Endpoint"
    ) == "custom:legacy-endpoint"
