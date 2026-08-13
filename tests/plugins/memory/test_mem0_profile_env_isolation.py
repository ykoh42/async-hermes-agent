"""Profile-scoped settings and credentials for Mem0 backends."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

import plugins.memory.mem0 as mem0_plugin
from agent.secret_scope import (
    UnscopedSecretError,
    is_multiplex_active,
    reset_secret_scope,
    set_multiplex_active,
    set_secret_scope,
)
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from plugins.memory.mem0 import Mem0MemoryProvider
from plugins.memory.mem0._native_oss import OpenAIEmbedding, OpenAILLM


class _FakeEmbeddings:
    def __init__(self, owner):
        self.owner = owner

    async def create(self, **kwargs):
        self.owner.requests.append(("embed", kwargs))
        return SimpleNamespace(
            data=[SimpleNamespace(index=0, embedding=[float(kwargs["input"][0])])]
        )


class _FakeCompletions:
    def __init__(self, owner):
        self.owner = owner

    async def create(self, **kwargs):
        self.owner.requests.append(("chat", kwargs))
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="answer", tool_calls=None)
                )
            ]
        )


class _FakeAsyncOpenAI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.requests = []
        self.close_calls = 0
        self.embeddings = _FakeEmbeddings(self)
        self.chat = SimpleNamespace(completions=_FakeCompletions(self))

    async def close(self):
        self.close_calls += 1


async def _run_oss_profile(name: str) -> dict:
    secrets = {
        "OPENAI_API_KEY": f"openai-{name}",
        "OPENAI_BASE_URL": f"https://openai-{name}.example/v1",
    }
    if name == "beta":
        secrets.update(
            {
                "OPENROUTER_API_KEY": "router-beta",
                "OPENROUTER_API_BASE": "https://router-beta.example/v1",
            }
        )
    token = set_secret_scope(secrets)
    embedder = OpenAIEmbedding({})
    llm = OpenAILLM({"model": "gpt-4.1-mini"})
    try:
        embedding = await embedder.embed("1")
        answer = await llm.generate_response(
            [{"role": "user", "content": "question"}]
        )
        embed_client = embedder._client
        llm_client = llm._client
        await embedder.close()
        await llm.close()
        return {
            "embedding": embedding,
            "answer": answer,
            "embed_kwargs": embed_client.kwargs,
            "llm_kwargs": llm_client.kwargs,
            "embed_closed": embed_client.close_calls,
            "llm_closed": llm_client.close_calls,
        }
    finally:
        reset_secret_scope(token)


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrent", [False, True])
async def test_sequential_and_concurrent_oss_clients_use_profile_secrets(
    monkeypatch,
    concurrent,
):
    clients = []

    async def create_client(client_class, **kwargs):
        client = client_class(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr("openai.AsyncOpenAI", _FakeAsyncOpenAI)
    monkeypatch.setattr(
        "agent.ssl_verify._create_openai_sdk_client",
        create_client,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "foreign-openai")
    monkeypatch.setenv("OPENROUTER_API_KEY", "foreign-router")
    previous_multiplex = is_multiplex_active()
    set_multiplex_active(True)
    try:
        if concurrent:
            alpha, beta = await asyncio.gather(
                _run_oss_profile("alpha"),
                _run_oss_profile("beta"),
            )
        else:
            alpha = await _run_oss_profile("alpha")
            beta = await _run_oss_profile("beta")
    finally:
        set_multiplex_active(previous_multiplex)

    assert alpha == {
        "embedding": [1.0],
        "answer": "answer",
        "embed_kwargs": {
            "api_key": "openai-alpha",
            "base_url": "https://openai-alpha.example/v1",
        },
        "llm_kwargs": {
            "api_key": "openai-alpha",
            "base_url": "https://openai-alpha.example/v1",
        },
        "embed_closed": 1,
        "llm_closed": 1,
    }
    assert beta == {
        "embedding": [1.0],
        "answer": "answer",
        "embed_kwargs": {
            "api_key": "openai-beta",
            "base_url": "https://openai-beta.example/v1",
        },
        "llm_kwargs": {
            "api_key": "router-beta",
            "base_url": "https://router-beta.example/v1",
        },
        "embed_closed": 1,
        "llm_closed": 1,
    }
    assert len(clients) == 4


async def _run_provider_profile(home, name: str, *, self_hosted: bool) -> dict:
    home_token = set_hermes_home_override(home)
    secret_token = set_secret_scope(
        {
            "MEM0_MODE": "platform",
            "MEM0_API_KEY": f"key-{name}",
            "MEM0_HOST": (
                f"https://env-{name}.invalid" if self_hosted else ""
            ),
            "MEM0_AGENT_ID": f"env-agent-{name}",
            "MEM0_USER_ID": f"env-user-{name}",
        }
    )
    provider = Mem0MemoryProvider()
    try:
        await provider.initialize("session")
        if self_hosted:
            provider._backend._client_transport = httpx.MockTransport(
                _mem0_handler
            )
        result = json.loads(
            await provider.handle_tool_call("mem0_add", {"content": name})
        )
        config = dict(provider._config)
        backend_client = provider._backend._client
        await provider.shutdown()
        return {
            "config": config,
            "user_id": provider._user_id,
            "agent_id": provider._agent_id,
            "result": result,
            "backend_closed": backend_client.is_closed,
        }
    finally:
        if provider._backend is not None:
            await provider.shutdown()
        reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)


_MEM0_REQUESTS: list[tuple[str, str, str | None, str | None, dict | None]] = []


def _mem0_handler(request: httpx.Request) -> httpx.Response:
    payload = json.loads(request.content) if request.content else None
    _MEM0_REQUESTS.append(
        (
            request.url.host or "",
            request.url.path,
            request.headers.get("Authorization"),
            request.headers.get("X-API-Key"),
            payload,
        )
    )
    if request.url.path == "/v1/ping/":
        return httpx.Response(
            200,
            json={"org_id": "org", "project_id": "project"},
            request=request,
        )
    return httpx.Response(
        200,
        json={"event_id": f"event-{request.url.host}"},
        request=request,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("concurrent", [False, True])
async def test_platform_and_self_hosted_backends_keep_profile_config_and_shape(
    tmp_path,
    monkeypatch,
    concurrent,
):
    home_a = tmp_path / "alpha"
    home_b = tmp_path / "beta"
    home_a.mkdir()
    home_b.mkdir()
    (home_a / "mem0.json").write_text(
        json.dumps(
            {
                "host": "",
                "agent_id": "file-agent-alpha",
                "user_id": "file-user-alpha",
            }
        ),
        encoding="utf-8",
    )
    (home_b / "mem0.json").write_text(
        json.dumps(
            {
                "host": "https://self-beta.example",
                "agent_id": "file-agent-beta",
                "user_id": "file-user-beta",
            }
        ),
        encoding="utf-8",
    )
    _MEM0_REQUESTS.clear()
    clients: list[httpx.AsyncClient] = []

    async def create_client(**kwargs):
        client = httpx.AsyncClient(
            **kwargs,
            transport=httpx.MockTransport(_mem0_handler),
        )
        clients.append(client)
        return client

    monkeypatch.setattr(
        "agent.ssl_verify._create_httpx_client",
        create_client,
    )
    monkeypatch.setenv("MEM0_API_KEY", "foreign-key")
    monkeypatch.setenv("MEM0_HOST", "https://foreign.invalid")
    previous_multiplex = is_multiplex_active()
    set_multiplex_active(True)
    try:
        if concurrent:
            alpha, beta = await asyncio.gather(
                _run_provider_profile(home_a, "alpha", self_hosted=False),
                _run_provider_profile(home_b, "beta", self_hosted=True),
            )
        else:
            alpha = await _run_provider_profile(
                home_a, "alpha", self_hosted=False
            )
            beta = await _run_provider_profile(home_b, "beta", self_hosted=True)
    finally:
        set_multiplex_active(previous_multiplex)

    assert alpha == {
        "config": {
            "mode": "platform",
            "api_key": "key-alpha",
            "host": "",
            "agent_id": "file-agent-alpha",
            "oss": {},
            "user_id": "file-user-alpha",
        },
        "user_id": "file-user-alpha",
        "agent_id": "file-agent-alpha",
        "result": {
            "result": "Fact queued for storage.",
            "event_id": "event-api.mem0.ai",
        },
        "backend_closed": True,
    }
    assert beta == {
        "config": {
            "mode": "platform",
            "api_key": "key-beta",
            "host": "https://self-beta.example",
            "agent_id": "file-agent-beta",
            "oss": {},
            "user_id": "file-user-beta",
        },
        "user_id": "file-user-beta",
        "agent_id": "file-agent-beta",
        "result": {
            "result": "Fact stored.",
            "event_id": "event-self-beta.example",
        },
        "backend_closed": True,
    }
    assert {
        (host, path, auth, api_key)
        for host, path, auth, api_key, _payload in _MEM0_REQUESTS
        if path in {"/v3/memories/add/", "/memories"}
    } == {
        ("api.mem0.ai", "/v3/memories/add/", "Token key-alpha", None),
        ("self-beta.example", "/memories", None, "key-beta"),
    }
    assert all(client.is_closed for client in clients)


@pytest.mark.asyncio
async def test_missing_multiplex_scope_fails_closed_for_config_and_oss_clients(
    monkeypatch,
):
    clients = []

    async def create_client(client_class, **kwargs):
        client = client_class(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr("openai.AsyncOpenAI", _FakeAsyncOpenAI)
    monkeypatch.setattr(
        "agent.ssl_verify._create_openai_sdk_client",
        create_client,
    )
    monkeypatch.setenv("MEM0_API_KEY", "foreign-mem0")
    monkeypatch.setenv("OPENAI_API_KEY", "foreign-openai")
    monkeypatch.setenv("OPENROUTER_API_KEY", "foreign-router")
    previous_multiplex = is_multiplex_active()
    set_multiplex_active(True)
    try:
        with pytest.raises(UnscopedSecretError):
            await mem0_plugin._load_config()
        with pytest.raises(UnscopedSecretError):
            Mem0MemoryProvider().get_config_schema()
        with pytest.raises(UnscopedSecretError):
            await OpenAIEmbedding({}).embed("1")
        with pytest.raises(UnscopedSecretError):
            await OpenAILLM({}).generate_response(
                [{"role": "user", "content": "question"}]
            )
    finally:
        set_multiplex_active(previous_multiplex)

    assert clients == []
