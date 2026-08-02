from types import SimpleNamespace

import pytest

pytest.importorskip("nemo_relay")

from agent import auxiliary_client, relay_llm, relay_runtime


@pytest.fixture()
def relay_turn(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    relay_runtime._reset_for_tests()
    relay_runtime.get_host()
    lease = relay_runtime.SESSION_COORDINATOR.acquire_conversation(
        profile_key=relay_runtime.current_profile_key(),
        session_id="session-1",
        platform="cli",
    )
    turn = relay_runtime.SESSION_COORDINATOR.begin_turn(
        lease,
        turn_id="turn-1",
        task_id="task-1",
    )
    try:
        yield lease.host.relay, turn
    finally:
        relay_runtime.SESSION_COORDINATOR.end_turn(turn, outcome="success")
        relay_runtime.SESSION_COORDINATOR.release_conversation(lease)
        relay_runtime._reset_for_tests()


@pytest.mark.asyncio
async def test_async_auxiliary_attempt_uses_inherited_relay_adapter(monkeypatch):
    captured = {}
    logical_completions = []

    async def create(**kwargs):
        return SimpleNamespace(
            request=kwargs,
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    async def execute_current(request, callback, **kwargs):
        captured.update(kwargs)
        return await callback(request)

    monkeypatch.setattr(
        relay_llm,
        "execute_current",
        execute_current,
    )
    async def complete_logical_call(request_id, *, outcome):
        logical_completions.append((request_id, outcome))

    monkeypatch.setattr(
        relay_llm,
        "complete_logical_call",
        complete_logical_call,
    )

    @auxiliary_client._relay_auxiliary_call
    async def run(task):
        auxiliary_client._set_relay_auxiliary_route(
            "anthropic",
            "claude-test",
            "chat_completions",
        )
        return await auxiliary_client._validate_llm_response(
            await auxiliary_client._relay_completion(
                client,
                {"model": "claude-test", "messages": []},
            ),
            task,
        )

    result = await run("title_generation")

    assert result.request["model"] == "claude-test"
    assert captured["name"] == "anthropic"
    assert captured["metadata"]["call_role"] == "auxiliary:title_generation"
    assert captured["defer_logical_completion"] is True
    assert logical_completions == [
        (captured["metadata"]["api_request_id"], "success")
    ]
