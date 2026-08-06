import asyncio
import json
from collections.abc import Callable

import httpx
import pytest

from hermes_cli import models


MODEL = "publisher/model"
BASE_URL = "http://127.0.0.1:1234/v1"


def _catalog(*, loaded_context=None, maximum=262_144):
    loaded_instances = []
    if loaded_context is not None:
        loaded_instances.append(
            {
                "id": f"{MODEL}:active",
                "config": {"context_length": loaded_context},
            }
        )
    return {
        "models": [
            {
                "key": MODEL,
                "max_context_length": maximum,
                "loaded_instances": loaded_instances,
            }
        ]
    }


def _use_transport(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request], httpx.Response],
) -> None:
    async_client = httpx.AsyncClient
    transport = httpx.MockTransport(handler)

    def create_client(*args, **kwargs):
        kwargs["transport"] = transport
        return async_client(*args, **kwargs)

    monkeypatch.setattr(models.httpx, "AsyncClient", create_client)


@pytest.mark.asyncio
async def test_fetch_lmstudio_models_filters_embedding_type(monkeypatch):
    def handler(request):
        assert request.url.path == "/api/v1/models"
        return httpx.Response(
            200,
            json={
                "models": [
                    {"key": "publisher/chat-model", "type": "llm"},
                    {"key": "publisher/embed-model", "type": "embedding"},
                ]
            },
        )

    _use_transport(monkeypatch, handler)

    result = await models.fetch_lmstudio_models(base_url=BASE_URL)

    assert result == ["publisher/chat-model"]


@pytest.mark.asyncio
async def test_probe_lmstudio_models_preserves_auth_error(monkeypatch):
    from hermes_cli.auth import AuthError

    _use_transport(
        monkeypatch,
        lambda _request: httpx.Response(401, json={"error": "unauthorized"}),
    )

    with pytest.raises(AuthError, match="HTTP 401"):
        await models.probe_lmstudio_models(base_url=BASE_URL)


@pytest.mark.asyncio
async def test_explicit_load_posts_requested_context(monkeypatch):
    requests = []

    def handler(request):
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, json=_catalog())
        return httpx.Response(
            200,
            json={"load_config": {"context_length": 100_000}},
        )

    _use_transport(monkeypatch, handler)

    result = await models.ensure_lmstudio_model_loaded(
        MODEL,
        BASE_URL,
        api_key="secret",
        target_context_length=100_000,
        return_load_result=True,
    )

    assert result == models.LMStudioLoadResult(100_000, load_attempted=True)
    assert [request.method for request in requests] == ["GET", "POST"]
    assert requests[1].url.path == "/api/v1/models/load"
    assert requests[1].headers["authorization"] == "Bearer secret"
    assert json.loads(requests[1].content) == {
        "model": MODEL,
        "echo_load_config": True,
        "context_length": 100_000,
    }


@pytest.mark.asyncio
async def test_missing_echo_refreshes_loaded_state(monkeypatch):
    catalogs = iter([_catalog(), _catalog(loaded_context=88_000)])

    def handler(request):
        if request.method == "POST":
            return httpx.Response(200, json={"status": "loaded"})
        return httpx.Response(200, json=next(catalogs))

    _use_transport(monkeypatch, handler)

    result = await models.ensure_lmstudio_model_loaded(
        MODEL,
        BASE_URL,
        api_key="",
        target_context_length=100_000,
    )

    assert result == 88_000


@pytest.mark.asyncio
async def test_explicit_override_above_known_maximum_is_rejected(monkeypatch):
    methods = []

    def handler(request):
        methods.append(request.method)
        return httpx.Response(
            200,
            json=_catalog(loaded_context=64_000, maximum=128_000),
        )

    _use_transport(monkeypatch, handler)

    result = await models.ensure_lmstudio_model_loaded(
        MODEL,
        BASE_URL,
        api_key="",
        target_context_length=256_000,
        return_load_result=True,
    )

    assert result == models.LMStudioLoadResult(None, rejected=True)
    assert methods == ["GET"]


@pytest.mark.asyncio
async def test_lmstudio_management_path_never_uses_thread_bridge(monkeypatch):
    async def reject_thread_bridge(*_args, **_kwargs):
        pytest.fail("LM Studio management I/O must remain native async")

    monkeypatch.setattr(asyncio, "to_thread", reject_thread_bridge)
    _use_transport(
        monkeypatch,
        lambda _request: httpx.Response(200, json=_catalog(loaded_context=64_000)),
    )

    result = await models.ensure_lmstudio_model_loaded(
        MODEL,
        BASE_URL,
        api_key="",
        target_context_length=None,
    )

    assert result == 64_000
