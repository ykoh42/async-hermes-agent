from __future__ import annotations

import json

import httpx
import pytest

from hermes_cli import codex_models
from hermes_cli.codex_models import DEFAULT_CODEX_MODELS, get_codex_model_ids


def test_public_codex_model_resolver_is_available() -> None:
    assert callable(get_codex_model_ids)


@pytest.mark.asyncio
async def test_fetch_from_api_keeps_supported_in_api_false_models(
    monkeypatch,
) -> None:
    async_client = httpx.AsyncClient

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "models": [
                    {"slug": "gpt-5.5", "priority": 0, "supported_in_api": True},
                    {
                        "slug": "gpt-5.3-codex-spark",
                        "priority": 7,
                        "supported_in_api": False,
                    },
                    {
                        "slug": "gpt-5-internal",
                        "priority": 99,
                        "visibility": "hidden",
                    },
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        codex_models.httpx,
        "AsyncClient",
        lambda **kwargs: async_client(
            transport=transport,
            **{
                key: value
                for key, value in kwargs.items()
                if key not in {"transport", "mounts"}
            },
        ),
    )

    models = await codex_models._fetch_models_from_api(access_token="tok")

    assert "gpt-5.5" in models
    assert "gpt-5.3-codex-spark" in models
    assert "gpt-5-internal" not in models


@pytest.mark.asyncio
async def test_codex_model_resolver_reads_local_sources_asynchronously(
    monkeypatch,
    tmp_path,
) -> None:
    codex_home = tmp_path / "codex"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text(
        'model = "configured-model"\n',
        encoding="utf-8",
    )
    (codex_home / "models_cache.json").write_text(
        json.dumps(
            {
                "models": [
                    {"slug": "cached-model", "priority": 1},
                    {"slug": "hidden-model", "visibility": "hidden"},
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    models = await get_codex_model_ids()

    assert models[0:2] == ["configured-model", "cached-model"]
    assert "hidden-model" not in models
    assert set(DEFAULT_CODEX_MODELS).issubset(models)
