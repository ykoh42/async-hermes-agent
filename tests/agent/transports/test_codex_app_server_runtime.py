"""Tests for the optional codex app-server runtime gate.

These are unit tests for the api_mode rewriter and the wire-level transport
module. They do NOT require the `codex` CLI to be installed — that's
covered by a separate live test gated on `codex --version`.
"""

from __future__ import annotations

import asyncio
import os

import aiofiles
import pytest

from hermes_cli.runtime_provider import (
    _VALID_API_MODES,
    _maybe_apply_codex_app_server_runtime,
)


class TestApiModeRegistration:
    """The new api_mode must be registered or downstream parsing rejects it."""

    def test_codex_app_server_is_a_valid_api_mode(self) -> None:
        assert "codex_app_server" in _VALID_API_MODES

    def test_existing_api_modes_still_present(self) -> None:
        # Regression guard: don't accidentally delete other api_modes when
        # touching this set.
        for mode in (
            "chat_completions",
            "codex_responses",
            "anthropic_messages",
            "bedrock_converse",
        ):
            assert mode in _VALID_API_MODES


class TestMaybeApplyCodexAppServerRuntime:
    """The opt-in helper that rewrites api_mode → codex_app_server."""

    @pytest.mark.parametrize(
        "model_cfg",
        [
            None,
            {},
            {"openai_runtime": ""},
            {"openai_runtime": "auto"},
            {"openai_runtime": "AUTO"},
            {"other_key": "codex_app_server"},  # wrong key
        ],
    )
    def test_default_off_for_openai(self, model_cfg) -> None:
        """Default behavior is preserved when the flag is unset/auto."""
        got = _maybe_apply_codex_app_server_runtime(
            provider="openai", api_mode="chat_completions", model_cfg=model_cfg
        )
        assert got == "chat_completions"

    def test_opt_in_rewrites_openai(self) -> None:
        got = _maybe_apply_codex_app_server_runtime(
            provider="openai",
            api_mode="chat_completions",
            model_cfg={"openai_runtime": "codex_app_server"},
        )
        assert got == "codex_app_server"



    @pytest.mark.parametrize(
        "provider",
        [
            "anthropic",
            "openrouter",
            "xai",
            "qwen-oauth",
            "opencode-zen",
            "bedrock",
            "",
        ],
    )
    def test_other_providers_never_rerouted(self, provider) -> None:
        """Non-OpenAI providers MUST NOT be rerouted even with the flag set —
        codex's app-server can only run OpenAI/Codex auth flows."""
        got = _maybe_apply_codex_app_server_runtime(
            provider=provider,
            api_mode="anthropic_messages",
            model_cfg={"openai_runtime": "codex_app_server"},
        )
        assert got == "anthropic_messages", (
            f"provider={provider!r} should not be rerouted to codex_app_server"
        )


class TestCodexAppServerModule:
    """Module-surface tests for the JSON-RPC speaker. Don't require codex CLI."""




    @pytest.mark.asyncio
    async def test_check_binary_handles_missing_executable(self) -> None:
        from agent.transports.codex_app_server import check_codex_binary

        ok, msg = await check_codex_binary(
            codex_bin="/nonexistent/codex/binary/path"
        )
        assert ok is False
        assert "not found" in msg.lower() or "no such" in msg.lower()

    @pytest.mark.asyncio
    async def test_check_binary_reaps_process_through_repeated_cancellation(
        self, monkeypatch
    ) -> None:
        from agent.transports.codex_app_server import check_codex_binary

        communicate_started = asyncio.Event()
        release_communicate = asyncio.Event()
        communicate_completed = asyncio.Event()

        class BlockingProcess:
            returncode = None
            killed = False

            async def communicate(self):
                communicate_started.set()
                await release_communicate.wait()
                communicate_completed.set()
                return b"", b""

            def kill(self):
                self.killed = True
                self.returncode = -9

        process = BlockingProcess()

        async def create_process(*_args, **_kwargs):
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)

        task = asyncio.create_task(check_codex_binary("unused"))
        await communicate_started.wait()
        task.cancel()
        while not process.killed:
            await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)

        assert task.done() is False
        release_communicate.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert communicate_completed.is_set()

    @pytest.mark.asyncio
    async def test_check_binary_communicate_failure_reaps_process(
        self, monkeypatch
    ) -> None:
        from agent.transports.codex_app_server import check_codex_binary

        waited = asyncio.Event()

        class FailedProcess:
            returncode = None
            killed = False

            async def communicate(self):
                raise RuntimeError("pipe failed")

            async def wait(self):
                waited.set()
                return self.returncode

            def kill(self):
                self.killed = True
                self.returncode = -9

        process = FailedProcess()

        async def create_process(*_args, **_kwargs):
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)

        with pytest.raises(RuntimeError, match="pipe failed"):
            await check_codex_binary("unused")
        assert process.killed is True
        assert waited.is_set()

    @pytest.mark.asyncio
    async def test_close_finishes_cleanup_through_repeated_cancellation(
        self, monkeypatch
    ) -> None:
        from agent.transports.codex_app_server import CodexAppServerClient

        cleanup_started = asyncio.Event()
        release_cleanup = asyncio.Event()
        cleanup_completed = asyncio.Event()
        client = CodexAppServerClient(codex_bin="unused")

        async def close_owned_resources(_timeout):
            cleanup_started.set()
            await release_cleanup.wait()
            cleanup_completed.set()

        monkeypatch.setattr(client, "_close_owned_resources", close_owned_resources)

        task = asyncio.create_task(client.close())
        await cleanup_started.wait()
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)

        assert task.done() is False
        release_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cleanup_completed.is_set()
        assert client._closed is True

    @pytest.mark.asyncio
    async def test_close_reaps_process_after_kill_through_repeated_cancellation(
        self,
    ) -> None:
        from agent.transports.codex_app_server import CodexAppServerClient

        kill_wait_started = asyncio.Event()
        release_kill_wait = asyncio.Event()
        kill_wait_completed = asyncio.Event()

        class BlockingProcess:
            stdin = None
            returncode = None
            wait_calls = 0
            killed = False

            def terminate(self):
                pass

            def kill(self):
                self.killed = True
                self.returncode = -9

            async def wait(self):
                self.wait_calls += 1
                if self.wait_calls == 1:
                    await asyncio.Event().wait()
                kill_wait_started.set()
                await release_kill_wait.wait()
                kill_wait_completed.set()
                return self.returncode

        process = BlockingProcess()
        client = CodexAppServerClient(codex_bin="unused")
        client._proc = process

        task = asyncio.create_task(client.close(timeout=0.01))
        await kill_wait_started.wait()
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)

        assert task.done() is False
        release_kill_wait.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert process.killed is True
        assert kill_wait_completed.is_set()
        assert client._proc is None

    def test_codex_error_class_is_runtimeerror(self) -> None:
        from agent.transports.codex_app_server import CodexAppServerError

        err = CodexAppServerError(code=-32600, message="boom")
        assert isinstance(err, RuntimeError)
        assert "boom" in str(err)
        assert "-32600" in str(err)

    @pytest.mark.asyncio
    async def test_json_rpc_round_trip_uses_native_async_stdio(
        self,
        tmp_path,
    ) -> None:
        from agent.transports.codex_app_server import CodexAppServerClient

        executable = tmp_path / "fake-codex"
        source = """#!/usr/bin/env python3
import json
import sys

for line in sys.stdin:
    message = json.loads(line)
    method = message.get("method")
    if "id" not in message:
        continue
    if method == "initialize":
        result = {"userAgent": "fake-codex"}
    else:
        result = {"method": method, "params": message.get("params", {})}
    print(json.dumps({"id": message["id"], "result": result}), flush=True)
    if method == "echo":
        print(json.dumps({"method": "turn/completed", "params": {}}), flush=True)
"""
        async with aiofiles.open(executable, "w", encoding="utf-8") as output:
            await output.write(source)
        os.chmod(executable, 0o755)

        client = CodexAppServerClient(codex_bin=str(executable))
        try:
            initialized = await client.initialize()
            assert initialized == {"userAgent": "fake-codex"}
            response = await client.request("echo", {"value": 42})
            assert response == {"method": "echo", "params": {"value": 42}}
            assert await client.take_notification(timeout=1) == {
                "method": "turn/completed",
                "params": {},
            }
        finally:
            await client.close()
        assert client.is_alive() is False

    @pytest.mark.asyncio
    async def test_cancelled_request_does_not_leave_pending_future(
        self,
        monkeypatch,
    ) -> None:
        from agent.transports.codex_app_server import CodexAppServerClient

        client = CodexAppServerClient(codex_bin="unused")
        client._proc = object()

        async def send(_message):
            await asyncio.sleep(10)

        monkeypatch.setattr(client, "_send", send)
        request = asyncio.create_task(client.request("never"))
        await asyncio.sleep(0)
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        assert client._pending == {}
        client._closed = True


class TestSpawnEnvIsolation:
    """The codex spawn must NOT rewrite HOME — codex's shell tool spawns
    subprocesses (gh, git, npm, aws, gcloud, ...) that need to find their
    config in the real user $HOME. CODEX_HOME isolates codex's own state,
    HOME stays unchanged.

    OpenClaw hit this footgun (openclaw/openclaw#81562) — they were
    rewriting HOME to a synthetic per-agent dir alongside CODEX_HOME,
    and then `gh auth status` / git config / etc. all broke inside codex
    shell calls. We avoid the same bug by only overlaying CODEX_HOME and
    RUST_LOG on top of os.environ.copy().
    """

    @pytest.mark.asyncio
    async def test_spawn_env_preserves_HOME(self, monkeypatch):
        """The spawn env must contain the parent process's HOME unchanged.
        Verifies the immutable spawn state built before lazy startup."""
        from agent.transports import codex_app_server as cas
        monkeypatch.setenv("HOME", "/users/alice")

        client = cas.CodexAppServerClient(codex_bin="codex")
        spawn_env = await client._build_spawn_env()

        # The spawn env must have HOME=/users/alice unchanged
        assert spawn_env.get("HOME") == "/users/alice", (
            f"HOME got rewritten in codex spawn env: "
            f"{spawn_env.get('HOME')!r}. Codex's shell tool's "
            "subprocesses (gh, git, aws, npm) need the user's real HOME."
        )

    @pytest.mark.asyncio
    async def test_spawn_env_sets_CODEX_HOME_when_provided(self, monkeypatch):
        """CODEX_HOME isolation must still work — that's the whole point
        of the codex_home arg."""
        from agent.transports import codex_app_server as cas
        monkeypatch.setenv("HOME", "/users/alice")

        client = cas.CodexAppServerClient(
            codex_bin="codex", codex_home="/tmp/profile/codex"
        )
        spawn_env = await client._build_spawn_env()
        assert spawn_env.get("CODEX_HOME") == "/tmp/profile/codex"
        # And HOME still passes through unchanged
        assert spawn_env.get("HOME") == "/users/alice"

class TestSpawnEnvSecretStripping:
    """codex app-server routes its spawn env through hermes_subprocess_env(
    inherit_credentials=True) instead of a raw os.environ.copy().

    codex is a model-driving CLI executor: it legitimately needs LLM provider
    credentials to authenticate, but it must NOT inherit Tier-1 Hermes secrets
    (gateway bot tokens, GitHub/infra auth, dashboard session token) or the
    dynamic-internal secrets (AUXILIARY_*_API_KEY / _BASE_URL side-LLM keys,
    GATEWAY_RELAY_* relay-auth) — a coding subprocess has no use for those and
    a model-controlled action could exfiltrate them. This closes the #29157
    sibling spawn-site gap (copilot_acp_client already routes through the
    helper; codex app-server predated it).
    """

    @staticmethod
    async def _capture_spawn_env(monkeypatch):
        from agent.transports import codex_app_server as cas
        client = cas.CodexAppServerClient(codex_bin="codex")
        return await client._build_spawn_env()

    @pytest.mark.asyncio
    async def test_tier1_and_internal_secrets_stripped_from_spawn_env(self, monkeypatch):
        for var, val in {
            "GH_TOKEN": "ghp-secret",
            "TELEGRAM_BOT_TOKEN": "bot-secret",
            "MODAL_TOKEN_SECRET": "modal-secret",
            "HERMES_DASHBOARD_SESSION_TOKEN": "dash-secret",
            "AUXILIARY_VISION_API_KEY": "aux-secret",
            "GATEWAY_RELAY_SECRET": "relay-secret",
            "GATEWAY_RELAY_ID": "relay-id",
            "GATEWAY_RELAY_DELIVERY_KEY": "relay-delivery",
        }.items():
            monkeypatch.setenv(var, val)

        env = await self._capture_spawn_env(monkeypatch)
        for var in (
            "GH_TOKEN", "TELEGRAM_BOT_TOKEN", "MODAL_TOKEN_SECRET",
            "HERMES_DASHBOARD_SESSION_TOKEN", "AUXILIARY_VISION_API_KEY",
            "GATEWAY_RELAY_SECRET", "GATEWAY_RELAY_ID", "GATEWAY_RELAY_DELIVERY_KEY",
        ):
            assert var not in env, f"{var} leaked into codex app-server spawn env"

    @pytest.mark.asyncio
    async def test_provider_credentials_still_reach_codex(self, monkeypatch):
        """codex authenticates against the model endpoint — provider keys must
        still flow through (inherit_credentials=True)."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-codex-needs-this")
        env = await self._capture_spawn_env(monkeypatch)
        assert env.get("OPENAI_API_KEY") == "sk-codex-needs-this"

    @pytest.mark.asyncio
    async def test_multiplex_provider_credentials_come_only_from_active_scope(
        self, monkeypatch
    ):
        from agent import secret_scope

        monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
        monkeypatch.setenv("OPENAI_API_KEY", "foreign-process-key")
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "foreign-claude-token")
        monkeypatch.setenv("CODEX_HOME", "/foreign/process/codex")
        token = secret_scope.set_secret_scope(
            {
                "OPENAI_API_KEY": "profile-key",
                "CLAUDE_CODE_OAUTH_TOKEN": "profile-claude-token",
                "CODEX_HOME": "/profiles/a/codex",
                "GH_TOKEN": "must-always-strip",
            }
        )
        try:
            env = await self._capture_spawn_env(monkeypatch)
        finally:
            secret_scope.reset_secret_scope(token)

        assert env.get("OPENAI_API_KEY") == "profile-key"
        assert env.get("CLAUDE_CODE_OAUTH_TOKEN") == "profile-claude-token"
        assert env.get("CODEX_HOME") == "/profiles/a/codex"
        assert "GH_TOKEN" not in env

    @pytest.mark.asyncio
    async def test_multiplex_provider_env_fails_without_scope(self, monkeypatch):
        from agent import secret_scope

        monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
        monkeypatch.setenv("OPENAI_API_KEY", "foreign-process-key")

        with pytest.raises(secret_scope.UnscopedSecretError):
            await self._capture_spawn_env(monkeypatch)

    @pytest.mark.asyncio
    async def test_direct_client_requires_scoped_codex_home_in_multiplex(
        self, monkeypatch
    ):
        from agent import secret_scope
        from agent.transports import codex_app_server as cas

        monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
        monkeypatch.setenv("CODEX_HOME", "/foreign/process/codex")
        token = secret_scope.set_secret_scope({"OPENAI_API_KEY": "profile-key"})
        try:
            with pytest.raises(RuntimeError, match="profile-scoped CODEX_HOME"):
                await cas.CodexAppServerClient()._build_spawn_env()
        finally:
            secret_scope.reset_secret_scope(token)
