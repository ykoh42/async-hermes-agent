"""LRU bound on the retained Piper/KittenTTS cache contract."""

import tools.tts_tool as tts


def test_loads_on_miss_and_serves_from_cache_on_hit():
    cache = {}
    calls = []

    def load():
        calls.append(1)
        return "model"

    assert tts._tts_cache_get_or_load(cache, "a", load) == "model"
    assert tts._tts_cache_get_or_load(cache, "a", load) == "model"
    assert len(calls) == 1


def test_hit_refreshes_recency(monkeypatch):
    monkeypatch.setattr(tts, "_TTS_MODEL_CACHE_MAX", 2)
    cache = {}
    tts._tts_cache_get_or_load(cache, "a", lambda: "a")
    tts._tts_cache_get_or_load(cache, "b", lambda: "b")
    tts._tts_cache_get_or_load(cache, "a", lambda: "a")
    tts._tts_cache_get_or_load(cache, "c", lambda: "c")
    assert "b" not in cache
    assert set(cache) == {"a", "c"}
