"""Regression test for upstream microsecond TTS output timestamps."""

import datetime
import inspect
import re

from tools import tts_tool


def test_default_output_path_uses_microsecond_timestamp():
    source = inspect.getsource(tts_tool.text_to_speech_tool)
    assert "%Y%m%d_%H%M%S_%f" in source


def test_timestamp_component_is_filename_safe():
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    assert re.fullmatch(r"\d{8}_\d{6}_\d{6}", stamp)
