from tools import tts_tool


def test_elevenlabs_default_uses_sdk_environment():
    assert tts_tool._elevenlabs_environment_kwargs({}) == {}
