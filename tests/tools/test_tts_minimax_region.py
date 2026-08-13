"""Async parity tests for MiniMax TTS region and credential selection."""

import pytest

from tools import tts_tool


GLOBAL_KEY = "FAKE_GLOBAL_CREDENTIAL"
CN_KEY = "FAKE_CN_CREDENTIAL"


@pytest.fixture
def fake_credentials(monkeypatch):
    values = {}

    async def resolve(env_var, provider_id):
        del provider_id
        return values.get(env_var, "")

    monkeypatch.setattr(tts_tool, "_resolve_provider_key", resolve)
    return values


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("config", "credentials", "expected"),
    [
        (
            {},
            {"MINIMAX_API_KEY": GLOBAL_KEY},
            ("global", tts_tool.DEFAULT_MINIMAX_BASE_URL, "MINIMAX_API_KEY", GLOBAL_KEY),
        ),
        (
            {},
            {"MINIMAX_CN_API_KEY": CN_KEY},
            ("cn", tts_tool.DEFAULT_MINIMAX_CN_BASE_URL, "MINIMAX_CN_API_KEY", CN_KEY),
        ),
        (
            {},
            {"MINIMAX_API_KEY": GLOBAL_KEY, "MINIMAX_CN_API_KEY": CN_KEY},
            ("global", tts_tool.DEFAULT_MINIMAX_BASE_URL, "MINIMAX_API_KEY", GLOBAL_KEY),
        ),
        (
            {"minimax": {"region": "cn"}},
            {"MINIMAX_API_KEY": GLOBAL_KEY, "MINIMAX_CN_API_KEY": CN_KEY},
            ("cn", tts_tool.DEFAULT_MINIMAX_CN_BASE_URL, "MINIMAX_CN_API_KEY", CN_KEY),
        ),
    ],
)
async def test_runtime_selection_matrix(
    fake_credentials,
    config,
    credentials,
    expected,
):
    fake_credentials.update(credentials)
    runtime = await tts_tool._resolve_minimax_tts_runtime(config)
    assert (
        runtime.region,
        runtime.endpoint,
        runtime.credential_source,
        runtime.api_key,
    ) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("region", "credentials", "missing_source"),
    [
        ("global", {"MINIMAX_CN_API_KEY": CN_KEY}, "MINIMAX_API_KEY"),
        ("cn", {"MINIMAX_API_KEY": GLOBAL_KEY}, "MINIMAX_CN_API_KEY"),
    ],
)
async def test_explicit_region_requires_matching_credential(
    fake_credentials,
    region,
    credentials,
    missing_source,
):
    fake_credentials.update(credentials)
    with pytest.raises(ValueError, match=missing_source):
        await tts_tool._resolve_minimax_tts_runtime(
            {"minimax": {"region": region}}
        )


@pytest.mark.asyncio
async def test_availability_uses_atomic_runtime(monkeypatch, fake_credentials):
    fake_credentials["MINIMAX_CN_API_KEY"] = CN_KEY

    async def config():
        return {"provider": "minimax"}

    monkeypatch.setattr(tts_tool, "_load_tts_config", config)
    assert await tts_tool.check_tts_requirements() is True


@pytest.mark.asyncio
async def test_runtime_repr_excludes_raw_credential(fake_credentials):
    fake_credentials["MINIMAX_API_KEY"] = GLOBAL_KEY
    runtime = await tts_tool._resolve_minimax_tts_runtime({})
    assert GLOBAL_KEY not in repr(runtime)
