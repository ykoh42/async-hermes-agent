import pytest

from agent import tts_registry
from agent.tts_provider import TTSProvider
from tools import tts_tool


class Provider(TTSProvider):
    name = "cartesia"

    async def synthesize(self, text, output_path, **kwargs):
        del text, kwargs
        return output_path


@pytest.fixture(autouse=True)
def registry_reset():
    tts_registry._reset_for_tests()
    yield
    tts_registry._reset_for_tests()


@pytest.mark.asyncio
async def test_registered_plugin_is_dispatched(monkeypatch, tmp_path):
    async def discovered(*args, **kwargs):
        del args, kwargs

    monkeypatch.setattr("hermes_cli.plugins._ensure_plugins_discovered", discovered)
    provider = Provider()
    tts_registry.register_provider(provider)
    output = str(tmp_path / "voice.mp3")
    assert await tts_tool._dispatch_to_plugin_provider(
        "hello", output, "cartesia", {}
    ) == output


@pytest.mark.asyncio
async def test_builtins_and_command_config_short_circuit():
    assert await tts_tool._dispatch_to_plugin_provider(
        "hello", "out.mp3", "openai", {}
    ) is None
    assert await tts_tool._dispatch_to_plugin_provider(
        "hello",
        "out.mp3",
        "cartesia",
        {"providers": {"cartesia": {"type": "command", "command": "x"}}},
    ) is None
