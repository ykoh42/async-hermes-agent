import os

import pytest

from tools import tts_tool


def test_builtin_names_cannot_be_command_providers():
    config = {"providers": {"openai": {"type": "command", "command": "x"}}}
    for name in tts_tool.BUILTIN_TTS_PROVIDERS:
        assert tts_tool._resolve_command_provider_config(name, config) is None


def test_command_lookup_and_defaults():
    config = {
        "providers": {
            "voice-cli": {"type": "command", "command": "voice {input_path}"}
        }
    }
    assert tts_tool._resolve_command_provider_config("VOICE-CLI", config)
    provider = config["providers"]["voice-cli"]
    assert tts_tool._get_command_tts_timeout(provider) == 120
    assert tts_tool._get_command_tts_output_format(provider) == "mp3"
    assert tts_tool._resolve_max_text_length("voice-cli", config) == 5000


def test_template_quotes_injected_values():
    rendered = tts_tool._render_command_tts_template(
        "voice {input_path} {output_path}",
        {"input_path": "a; touch bad", "output_path": "out file.mp3"},
    )
    if os.name != "nt":
        assert "'a; touch bad'" in rendered
        assert "'out file.mp3'" in rendered


@pytest.mark.asyncio
async def test_configured_command_provider_satisfies_requirements(monkeypatch):
    async def config():
        return {
            "provider": "voice-cli",
            "providers": {
                "voice-cli": {"type": "command", "command": "voice"}
            },
        }

    monkeypatch.setattr(tts_tool, "_load_tts_config", config)
    assert await tts_tool.check_tts_requirements() is True
