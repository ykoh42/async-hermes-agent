"""Speaker playback belongs to the intentionally removed CLI/gateway surface."""

from tools import tts_tool


def test_speaker_streaming_surface_is_not_shipped():
    assert not hasattr(tts_tool, "stream_tts_to_speaker")
