import pytest

from tools import tts_tool


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        ("edge", 5000),
        ("openai", 4096),
        ("xai", 15000),
        ("minimax", 10000),
        ("mistral", 4000),
        ("gemini", 32000),
        ("unknown", 4000),
        ("", 4000),
    ],
)
def test_provider_limits(provider, expected):
    assert tts_tool._resolve_max_text_length(provider, {}) == expected


def test_user_override_wins():
    assert tts_tool._resolve_max_text_length(
        "openai", {"openai": {"max_text_length": 123}}
    ) == 123
