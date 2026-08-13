"""Parity tests for the shared TTS text normalization helper."""

from tools.tts_text_normalize import (
    flatten_newlines_for_payload,
    prepare_spoken_text,
    strip_nonspoken_blocks,
)


def test_prepare_spoken_text_expands_weather_units():
    raw = "## Weather\n14°C, wind 9 km/h, rain 1.3 mm, range 11–17°C"
    spoken = prepare_spoken_text(raw)
    assert "14 degrees Celsius" in spoken
    assert "9 kilometres per hour" in spoken
    assert "1.3 millimetres" in spoken
    assert "11 to 17 degrees Celsius" in spoken


def test_prepare_spoken_text_polishes_edge_cases():
    assert prepare_spoken_text("## Weather\nIt will be sunny") == (
        "Weather, It will be sunny."
    )
    assert "300 US dollars" in prepare_spoken_text("US$300, next")
    assert "5 dollars per month" in prepare_spoken_text("$5/month")
    assert "2026/06/02" in prepare_spoken_text("due 2026/06/02 ok")


def test_reasoning_and_verifier_blocks_are_not_spoken():
    footer = (
        "⚠️ File-mutation verifier: 1 file(s) were NOT modified this turn "
        "despite any wording above that may suggest otherwise."
    )
    spoken = prepare_spoken_text(
        "<think budget=high>secret reasoning</think>The answer is 42.\n" + footer
    )
    assert "secret reasoning" not in spoken
    assert "File-mutation verifier" not in spoken
    assert "42" in spoken


def test_unterminated_think_block_is_removed():
    spoken = prepare_spoken_text("Answer first. <think>truncated reasoning")
    assert "truncated reasoning" not in spoken
    assert "Answer first" in spoken


def test_multiple_think_blocks_are_removed():
    spoken = strip_nonspoken_blocks("<think>a</think>one<think>b</think> two")
    assert spoken.strip() == "one  two"


def test_emoji_and_markdown_are_removed():
    spoken = prepare_spoken_text("**Done!** 🎉🚀 All tests pass ✅")
    assert "**" not in spoken
    assert "🎉" not in spoken
    assert "🚀" not in spoken
    assert "✅" not in spoken
    assert "All tests pass" in spoken


def test_payload_newlines_are_flattened_without_double_punctuation():
    spoken = flatten_newlines_for_payload("Alpha.\nBeta!\n\nGamma")
    assert "\n" not in spoken
    assert ".." not in spoken
    assert "Alpha." in spoken
    assert "Beta!" in spoken


def test_max_chars_is_applied_after_cleanup():
    assert prepare_spoken_text("**abcdef**", max_chars=4) == "abcd"
