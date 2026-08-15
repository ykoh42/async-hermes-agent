"""Tests for tools/mcp_oauth.py — OAuth 2.1 PKCE support for MCP servers."""

import json
import os
import stat
import sys
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import urlparse

import pytest

import asyncio

from tools.mcp_oauth import (
    HermesTokenStorage,
    OAuthNonInteractiveError,
    build_oauth_auth,
    remove_oauth_tokens,
    _find_free_port,
    _is_interactive,
    _wait_for_callback,
    _apply_pasted_callback,
    _make_callback_handler,
    _make_redirect_handler,
)


def _set_interactive_stdin(monkeypatch, *, is_tty: bool = True) -> None:
    mock_stdin = MagicMock()
    mock_stdin.isatty.return_value = is_tty
    monkeypatch.setattr("tools.mcp_oauth.sys.stdin", mock_stdin)


async def _hit_callback_when_ready(url: str, timeout: float = 15.0) -> None:
    """Drive the loopback callback as soon as the waiter's server answers.

    Polls instead of sleeping a fixed interval: the reserved socket is bound
    but NOT listening until ``_wait_for_callback`` adopts it, so attempts
    before adoption fail fast with a connection error.
    """
    target = urlparse(url)
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        try:
            reader, writer = await asyncio.open_connection(target.hostname, target.port)
            request_target = target.path + (f"?{target.query}" if target.query else "")
            writer.write(
                f"GET {request_target} HTTP/1.1\r\nHost: {target.hostname}\r\n\r\n".encode()
            )
            await writer.drain()
            await reader.read()
            writer.close()
            await writer.wait_closed()
            return
        except OSError:
            await asyncio.sleep(0.01)
    raise AssertionError(f"callback listener never came up: {url}")


# ---------------------------------------------------------------------------
# HermesTokenStorage
# ---------------------------------------------------------------------------

class TestHermesTokenStorage:
    @pytest.mark.asyncio
    async def test_cancelled_atomic_write_removes_temporary_file(
        self, tmp_path, monkeypatch
    ):
        from tools import mcp_oauth

        replace_started = asyncio.Event()

        async def stalled_replace(_source, _destination):
            replace_started.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(mcp_oauth.aiofiles.os, "replace", stalled_replace)
        target = tmp_path / "tokens.json"
        write = asyncio.create_task(mcp_oauth._write_json(target, {"token": "secret"}))
        await asyncio.wait_for(replace_started.wait(), timeout=1)
        write.cancel()
        with pytest.raises(asyncio.CancelledError):
            await write

        assert not list(tmp_path.glob("tokens.tmp.*"))

    @pytest.mark.asyncio
    async def test_roundtrip_tokens(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        storage = HermesTokenStorage("test-server")

        # Initially empty
        assert await storage.get_tokens() is None

        # Save and retrieve
        mock_token = MagicMock()
        mock_token.model_dump.return_value = {
            "access_token": "abc123",
            "token_type": "Bearer",
            "refresh_token": "ref456",
        }
        await storage.set_tokens(mock_token)

        # File exists with correct permissions
        token_path = tmp_path / "mcp-tokens" / "test-server.json"
        assert token_path.exists()
        data = json.loads(token_path.read_text())
        assert data["access_token"] == "abc123"

    @pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX mode bits not enforced on Windows")
    @pytest.mark.asyncio
    async def test_token_file_created_with_0o600(self, tmp_path, monkeypatch):
        """Tokens must land on disk at 0o600 with no umask-default exposure window.

        Regression for the TOCTOU race where ``write_text`` + post-write
        ``chmod`` briefly left credentials at the process umask (commonly
        0o644 = world-readable) before tightening to owner-only. Mirrors
        the fix shipped for ``agent/google_oauth.py`` in #19673.
        """
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        storage = HermesTokenStorage("perm-test-server")

        mock_token = MagicMock()
        mock_token.model_dump.return_value = {
            "access_token": "secret-abc",
            "token_type": "Bearer",
            "refresh_token": "secret-ref",
        }
        await storage.set_tokens(mock_token)

        token_path = tmp_path / "mcp-tokens" / "perm-test-server.json"
        assert token_path.exists()
        mode = stat.S_IMODE(token_path.stat().st_mode)
        assert mode == 0o600, f"token file mode {oct(mode)} != 0o600 — TOCTOU race regressed"

        parent_mode = stat.S_IMODE(token_path.parent.stat().st_mode)
        assert parent_mode == 0o700, (
            f"token parent dir mode {oct(parent_mode)} != 0o700 — siblings can traverse"
        )


    @pytest.mark.asyncio
    async def test_corrupt_tokens_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        storage = HermesTokenStorage("bad-server")

        d = tmp_path / "mcp-tokens"
        d.mkdir(parents=True)
        (d / "bad-server.json").write_text("NOT VALID JSON{{{")

        assert await storage.get_tokens() is None


# ---------------------------------------------------------------------------
# build_oauth_auth
# ---------------------------------------------------------------------------

class TestBuildOAuthAuth:
    @pytest.mark.asyncio
    async def test_returns_none_without_sdk(self, monkeypatch):
        import tools.mcp_oauth as mod
        monkeypatch.setattr(mod, "_OAUTH_AVAILABLE", False)
        result = await build_oauth_auth("test", "https://example.com")
        assert result is None


    @pytest.mark.asyncio
    async def test_scope_passed_through(self, tmp_path, monkeypatch):
        pytest.importorskip("mcp.client.auth")

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _set_interactive_stdin(monkeypatch)
        provider = await build_oauth_auth("scoped", "https://example.com/mcp", {
            "scope": "read write admin",
        })
        assert provider is not None
        assert provider.context.client_metadata.scope == "read write admin"


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

class TestRedirectHandlerSshHint:
    """_make_redirect_handler must print an SSH tunnel hint on remote sessions."""

    @pytest.mark.asyncio
    async def test_ssh_hint_shown_on_ssh_session(self, monkeypatch, capsys):
        import tools.mcp_oauth as mco
        monkeypatch.setattr(mco, "_is_interactive", lambda: True)
        monkeypatch.setenv("SSH_CLIENT", "1.2.3.4 1234 22")
        monkeypatch.delenv("SSH_TTY", raising=False)

        handler = _make_redirect_handler(49200)
        await handler("https://example.com/auth?foo=bar")

        err = capsys.readouterr().err
        assert "49200" in err
        assert "ssh -N -L" in err
        assert "Remote session detected" in err

    @pytest.mark.asyncio
    async def test_configured_redirect_uri_shows_proxy_hint_not_tunnel(self, monkeypatch, capsys):
        """With a proxy redirect_uri, the SSH hint must not push the loopback tunnel.

        The Funnel/proxy callback reaches this machine on its own, so the
        ``ssh -N -L`` guidance would be actively misleading.
        """
        import tools.mcp_oauth as mco
        monkeypatch.setattr(mco, "_oauth_port", 49203)
        monkeypatch.setattr(mco, "_is_interactive", lambda: True)
        monkeypatch.setenv("SSH_CLIENT", "1.2.3.4 1234 22")

        handler = _make_redirect_handler(
            49203, redirect_uri="https://oauth.example.ts.net/callback"
        )
        await handler("https://example.com/auth")

        err = capsys.readouterr().err
        assert "https://oauth.example.ts.net/callback" in err
        assert "no SSH tunnel needed" in err
        assert "ssh -N -L" not in err
        assert "127.0.0.1" not in err


# ---------------------------------------------------------------------------
# Path traversal protection
# ---------------------------------------------------------------------------

class TestPathTraversal:
    """Verify server_name is sanitized to prevent path traversal."""

    def test_dots_and_slashes_sanitized(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        storage = HermesTokenStorage("../../../etc/passwd")
        path = storage._tokens_path()
        resolved = path.resolve()
        assert resolved.is_relative_to((tmp_path / "mcp-tokens").resolve())

    def test_normal_name_unchanged(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        storage = HermesTokenStorage("my-mcp-server")
        assert "my-mcp-server.json" in str(storage._tokens_path())

    def test_special_chars_sanitized(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        storage = HermesTokenStorage("server@host:8080/path")
        path = storage._tokens_path()
        assert "@" not in path.name
        assert ":" not in path.name
        assert "/" not in path.stem


# ---------------------------------------------------------------------------
# Callback handler isolation
# ---------------------------------------------------------------------------

class TestCallbackHandlerIsolation:
    """Verify concurrent OAuth flows don't share state."""

    async def _serve_callback(self, query: str):
        handler, result = _make_callback_handler()
        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            await _hit_callback_when_ready(f"http://127.0.0.1:{port}/callback?{query}")
        finally:
            server.close()
            await server.wait_closed()
        return result

    @pytest.mark.asyncio
    async def test_handler_writes_to_own_result(self):
        result = await self._serve_callback("code=test123&state=mystate")
        assert result["auth_code"] == "test123"
        assert result["state"] == "mystate"

    @pytest.mark.asyncio
    async def test_handler_captures_error(self):
        result = await self._serve_callback("error=access_denied")

        assert result["auth_code"] is None
        assert result["error"] == "access_denied"


# ---------------------------------------------------------------------------
# TOCTOU port reservation (#22161)
# ---------------------------------------------------------------------------

class TestCallbackPortReservation:
    """The socket picked at selection time stays bound until callback bind.

    _find_free_port() closed its probe socket before HTTPServer re-bound the
    port, leaving a race window where another process could steal it
    (#22161). _reserve_callback_port() keeps the bound socket parked in
    _reserved_sockets until _wait_for_callback adopts it.
    """

    def test_reserved_port_cannot_be_stolen(self):
        import socket as sock
        import tools.mcp_oauth as mod

        port = mod._reserve_callback_port()
        try:
            # The reservation holds the bind — a competing bind must fail.
            thief = sock.socket(sock.AF_INET, sock.SOCK_STREAM)
            with pytest.raises(OSError):
                thief.bind(("127.0.0.1", port))
            thief.close()
        finally:
            reserved = mod._reserved_sockets.pop(port, None)
            if reserved is not None:
                reserved.close()

    @pytest.mark.asyncio
    async def test_pinned_port_is_not_reserved(self):
        import tools.mcp_oauth as mod

        cfg: dict = {"redirect_port": 49399}
        port = await mod._configure_callback_port(cfg)
        assert port == 49399
        assert cfg["_resolved_port"] == 49399
        assert 49399 not in mod._reserved_sockets

    @pytest.mark.asyncio
    async def test_wait_for_callback_adopts_reserved_socket(self, monkeypatch):
        """E2E: reserve → _wait_for_callback binds the SAME socket and the
        callback round-trips through it."""
        import asyncio
        import tools.mcp_oauth as mod

        cfg: dict = {}
        port = await mod._configure_callback_port(cfg)
        monkeypatch.setattr(mod, "_is_interactive", lambda: False)
        # Bypass the non-interactive guard — this test drives the flow directly.
        monkeypatch.setattr(mod, "_raise_if_non_interactive", lambda lead: None)

        async def drive():
            task = asyncio.create_task(mod._wait_for_callback())
            await _hit_callback_when_ready(
                f"http://127.0.0.1:{port}/callback?code=abc123&state=xyz"
            )
            return await asyncio.wait_for(task, timeout=20)

        code, state = await drive()
        assert code == "abc123"
        assert state == "xyz"
        # Reservation was consumed by adoption.
        assert port not in mod._reserved_sockets

    @pytest.mark.asyncio
    async def test_concurrent_flows_keep_their_own_callback_ports(self, monkeypatch):
        """#34260: flow A's waiter listens on A's port even after flow B
        overwrites the legacy module-level global.

        This is the callback-side sibling of the #44588 redirect-handler fix:
        without a per-flow waiter, A's callback wait would bind B's port and
        A's redirect (pointing at A's port) would never be received.
        """
        import asyncio
        import tools.mcp_oauth as mod

        monkeypatch.setattr(mod, "_is_interactive", lambda: False)
        monkeypatch.setattr(mod, "_raise_if_non_interactive", lambda lead: None)

        cfg_a: dict = {}
        port_a = await mod._configure_callback_port(cfg_a)
        waiter_a = mod._make_callback_waiter(port_a)
        # Flow B configures afterwards — overwrites mod._oauth_port.
        cfg_b: dict = {}
        port_b = await mod._configure_callback_port(cfg_b)
        assert mod._oauth_port == port_b != port_a

        async def drive():
            task = asyncio.create_task(waiter_a())
            # The redirect goes to flow A's port — where A's waiter must be
            # listening despite the clobbered global.
            await _hit_callback_when_ready(
                f"http://127.0.0.1:{port_a}/callback?code=flowA&state=sA"
            )
            return await asyncio.wait_for(task, timeout=20)

        try:
            code, state = await drive()
        finally:
            leftover = mod._reserved_sockets.pop(port_b, None)
            if leftover is not None:
                leftover.close()
        assert code == "flowA"
        assert state == "sA"


# ---------------------------------------------------------------------------
# remove_oauth_tokens
# ---------------------------------------------------------------------------

class TestRemoveOAuthTokens:
    @pytest.mark.asyncio
    async def test_removes_files(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        d = tmp_path / "mcp-tokens"
        d.mkdir()
        (d / "myserver.json").write_text("{}")
        (d / "myserver.client.json").write_text("{}")

        await remove_oauth_tokens("myserver")

        assert not (d / "myserver.json").exists()
        assert not (d / "myserver.client.json").exists()


# ---------------------------------------------------------------------------
# Non-interactive / startup-safety tests
# ---------------------------------------------------------------------------

class TestIsInteractive:
    """_is_interactive() detects headless/daemon/container environments."""

    def test_suppress_interactive_oauth_disables_stdin_prompts(self, monkeypatch):
        import tools.mcp_oauth as mod

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True
        monkeypatch.setattr("tools.mcp_oauth.sys.stdin", mock_stdin)

        assert _is_interactive() is True
        with mod.suppress_interactive_oauth():
            assert _is_interactive() is False
        assert _is_interactive() is True

    def test_suppression_propagates_across_run_coroutine_threadsafe(self, monkeypatch):
        """#35927 core: suppression set on the discovery thread MUST reach the
        coroutine asyncio runs on a *different* (event-loop) thread — that is
        where the OAuth callback / _is_interactive() actually executes via
        run_coroutine_threadsafe. A threading.local would NOT propagate here
        (the original fix's defect); a ContextVar does."""
        import asyncio
        import threading
        import tools.mcp_oauth as mod

        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True
        monkeypatch.setattr("tools.mcp_oauth.sys.stdin", mock_stdin)

        loop = asyncio.new_event_loop()
        loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
        loop_thread.start()
        result = {}
        try:
            async def _probe_on_loop_thread():
                # runs on the loop thread, NOT the one that set suppression
                return (threading.current_thread() is not discovery_thread,
                        _is_interactive())

            discovery_thread = None

            def _discovery():
                nonlocal discovery_thread
                discovery_thread = threading.current_thread()
                with mod.suppress_interactive_oauth():
                    fut = asyncio.run_coroutine_threadsafe(
                        _probe_on_loop_thread(), loop
                    )
                    result["cross_thread"], result["interactive"] = fut.result(timeout=5)

            dt = threading.Thread(target=_discovery)
            dt.start()
            dt.join()
        finally:
            loop.call_soon_threadsafe(loop.stop)
            loop_thread.join(timeout=5)
            loop.close()

        assert result["cross_thread"] is True, "probe must run on the loop thread"
        # The whole point: suppression must hold on the loop thread.
        assert result["interactive"] is False


class TestWaitForCallbackNoBlocking:
    """_wait_for_callback() must never call input() — it raises instead."""

    def test_raises_on_timeout_instead_of_input(self, monkeypatch):
        """Interactive session: when no auth code arrives, raises on timeout.

        Marked interactive so the fail-fast non-interactive guard (#57836)
        does not short-circuit — this test exercises the timeout path.
        """
        import tools.mcp_oauth as mod
        import asyncio

        mod._oauth_port = _find_free_port()
        monkeypatch.setattr(mod, "_is_interactive", lambda: True)
        # EOF on the paste reader so only the HTTP-listener timeout drives it.
        monkeypatch.setattr("sys.stdin", MagicMock(readline=lambda: ""))

        real_wait_for = mod.asyncio.wait_for

        async def immediate_timeout(awaitable, *, timeout):
            if timeout == 300.0:
                awaitable.close()
                raise TimeoutError
            return await real_wait_for(awaitable, timeout=timeout)

        with patch.object(mod.asyncio, "wait_for", immediate_timeout):
            with patch("builtins.input", side_effect=AssertionError("input() must not be called")):
                with pytest.raises(OAuthNonInteractiveError, match="callback timed out"):
                    asyncio.run(_wait_for_callback())


class TestBuildOAuthAuthNonInteractive:
    """build_oauth_auth() in non-interactive mode."""

    @pytest.mark.asyncio
    async def test_noninteractive_without_cached_tokens_fails_fast(self, tmp_path, monkeypatch):
        """Without cached tokens, non-interactive mode skips browser auth."""
        pytest.importorskip("mcp.client.auth")

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = False
        monkeypatch.setattr("tools.mcp_oauth.sys.stdin", mock_stdin)

        with pytest.raises(OAuthNonInteractiveError, match="non-interactive"):
            await build_oauth_auth("atlassian", "https://mcp.atlassian.com/v1/mcp")


class TestNonInteractiveFailFastAtCallbackBoundary:
    """#57836: a cached-but-unusable token (expired/revoked, refresh rejected)
    makes the MCP SDK fall through to the authorization-code flow even though
    build_oauth_auth's token-file guard passed. In a non-interactive context
    (systemd gateway, cron, background discovery) that flow must fail fast at
    the redirect/callback boundary — never launch a browser flow or bind a
    callback listener, and never block for the full timeout — so gateway
    startup is not gated on an unusable optional MCP server, and retries do not
    collide on the callback port ('Address already in use').
    """

    def test_wait_for_callback_rejects_before_binding_when_noninteractive(self, monkeypatch):
        """No listener bound and no poll loop entered when non-interactive."""
        import tools.mcp_oauth as mod
        import asyncio

        mod._oauth_port = _find_free_port()
        monkeypatch.setattr(mod, "_is_interactive", lambda: False)

        start_server = AsyncMock(
            side_effect=AssertionError("must not bind callback listener")
        )
        monkeypatch.setattr(mod.asyncio, "start_server", start_server)

        with pytest.raises(OAuthNonInteractiveError, match="interactive session"):
            asyncio.run(mod._wait_for_callback())
        start_server.assert_not_awaited()

    def test_redirect_handler_rejects_and_does_not_open_browser(self, monkeypatch, capsys):
        """Non-interactive redirect must not print an auth URL or open a browser."""
        import tools.mcp_oauth as mod
        import asyncio

        monkeypatch.setattr(mod, "_is_interactive", lambda: False)
        with pytest.raises(OAuthNonInteractiveError, match="browser authorization"):
            asyncio.run(mod._make_redirect_handler(49300)("https://idp.example.com/authorize?x=1"))

        err = capsys.readouterr().err
        assert "https://idp.example.com/authorize" not in err

    def test_guard_does_not_fire_on_interactive_redirect(self, monkeypatch, capsys):
        """Positive control: the fail-fast guard is scoped to the auth-code path.

        #57836 regression coverage asks that valid/refreshable OAuth keeps
        working non-interactively — a good token never reaches these handlers,
        so the guard must be inert once a real flow is in progress. Assert the
        interactive path still prints the URL and does not raise, proving the
        guard does not over-fire and swallow legitimate authorization.
        """
        import tools.mcp_oauth as mod
        import asyncio

        monkeypatch.setattr(mod, "_is_interactive", lambda: True)
        # Local (non-SSH) interactive session prints the URL for the caller to open.
        monkeypatch.delenv("SSH_CLIENT", raising=False)
        monkeypatch.delenv("SSH_TTY", raising=False)

        asyncio.run(mod._make_redirect_handler(49302)("https://idp.example.com/authorize?x=9"))

        err = capsys.readouterr().err
        assert "https://idp.example.com/authorize?x=9" in err


# ---------------------------------------------------------------------------
# Extracted helper tests (Task 3 of MCP OAuth consolidation)
# ---------------------------------------------------------------------------

_PROXY_REDIRECT = "https://oauth.example.ts.net/callback"


@pytest.mark.parametrize("cfg, expected_auth", [
    ({}, "none"),                                    # public client
    ({"client_secret": "shh"}, "client_secret_post"),  # confidential client
])
@pytest.mark.asyncio
async def test_build_client_metadata_token_endpoint_auth(cfg, expected_auth):
    pytest.importorskip("mcp")
    from tools.mcp_oauth import _build_client_metadata, _configure_callback_port

    await _configure_callback_port(cfg)
    md = _build_client_metadata(cfg)
    assert md.token_endpoint_auth_method == expected_auth
    assert "authorization_code" in md.grant_types
    assert "refresh_token" in md.grant_types


@pytest.mark.asyncio
async def test_client_identity_change_discards_cached_tokens_and_metadata(tmp_path):
    """A pre-registered client change cannot reuse tokens minted for the old client."""
    from tools import mcp_oauth

    storage = HermesTokenStorage("srv", hermes_home=tmp_path)
    await mcp_oauth._write_json(
        storage._client_info_path(),
        {"client_id": "old-client", "client_secret": "old-secret"},
    )
    await mcp_oauth._write_json(storage._tokens_path(), {"access_token": "stale"})
    await mcp_oauth._write_json(
        storage._meta_path(), {"token_endpoint": "https://old.example/token"}
    )

    await mcp_oauth._invalidate_tokens_on_client_change(
        storage, "new-client", "new-secret"
    )

    assert not storage._tokens_path().exists()
    assert not storage._meta_path().exists()
    assert storage._client_info_path().exists()


@pytest.mark.asyncio
async def test_matching_client_identity_preserves_cached_tokens(tmp_path):
    """Rebuilding auth with the same registration leaves live cached state intact."""
    from tools import mcp_oauth

    storage = HermesTokenStorage("srv", hermes_home=tmp_path)
    await mcp_oauth._write_json(
        storage._client_info_path(),
        {"client_id": "same-client", "client_secret": "same-secret"},
    )
    await mcp_oauth._write_json(storage._tokens_path(), {"access_token": "live"})
    await mcp_oauth._write_json(
        storage._meta_path(), {"token_endpoint": "https://idp/token"}
    )

    await mcp_oauth._invalidate_tokens_on_client_change(
        storage, "same-client", "same-secret"
    )

    assert storage._tokens_path().read_text()
    assert storage._meta_path().read_text()


@pytest.mark.parametrize("cfg, expected", [
    ({"redirect_uri": _PROXY_REDIRECT}, _PROXY_REDIRECT),
    ({}, "http://127.0.0.1:1234/callback"),
    # ``redirect_host: localhost`` swaps only the loopback hostname (WAF-safe)
    ({"redirect_host": "localhost"}, "http://localhost:1234/callback"),
])
def test_resolve_redirect_uri(cfg, expected):
    from tools.mcp_oauth import _resolve_redirect_uri

    assert _resolve_redirect_uri(cfg, 1234) == expected


@pytest.mark.asyncio
async def test_build_oauth_auth_preserves_server_url_path():
    """server_url with path is forwarded to OAuthClientProvider unmodified.

    Regression for #16015: previously ``_parse_base_url`` stripped the path,
    collapsing ``https://mcp.notion.com/mcp`` to ``https://mcp.notion.com`` and
    breaking RFC 9728 protected-resource validation against servers whose PRM
    advertises a path-scoped resource (Notion). The MCP SDK strips the path
    itself for authorization-server discovery via
    ``OAuthContext.get_authorization_base_url``; Hermes must not pre-strip.
    """
    from tools import mcp_oauth

    captured: dict = {}

    class _FakeProvider:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    with patch.object(mcp_oauth, "_OAUTH_AVAILABLE", True), \
         patch.object(mcp_oauth, "OAuthClientProvider", _FakeProvider), \
         patch.object(mcp_oauth, "_is_interactive", return_value=True), \
         patch.object(mcp_oauth, "_maybe_preregister_client", new_callable=AsyncMock), \
         patch.object(mcp_oauth, "HermesTokenStorage") as mock_storage_cls:
        mock_storage_cls.return_value.has_cached_tokens = AsyncMock(return_value=True)
        await build_oauth_auth(
            server_name="notion",
            server_url="https://mcp.notion.com/mcp",
            oauth_config={},
        )

    assert captured["server_url"] == "https://mcp.notion.com/mcp"


class TestPasteCallbackReader:
    """Pasted redirect parsing is independent from async stdin transport."""

    def _empty_result(self):
        return {
            "auth_code": None,
            "state": None,
            "error": None,
            "ready": asyncio.Event(),
        }

    def test_parses_pasted_callback(self):
        result = self._empty_result()
        pasted = "http://127.0.0.1:37949/callback?code=abc&state=xyz\n"
        _apply_pasted_callback(pasted, result)
        assert result["auth_code"] == "abc"
        assert result["state"] == "xyz"
        assert result["error"] is None


class TestWaitForCallbackPasteIntegration:
    """_wait_for_callback offers the paste prompt only when interactive."""

    def test_paste_prompt_shown_on_tty(self, monkeypatch, capsys):
        import tools.mcp_oauth as mod
        mod._oauth_port = _find_free_port()
        monkeypatch.setattr(mod, "_is_interactive", lambda: True)
        async def timeout(awaitable, **_kwargs):
            awaitable.close()
            raise TimeoutError

        monkeypatch.setattr(mod.asyncio, "wait_for", timeout)
        monkeypatch.setattr(mod, "_wait_for_pasted_callback", AsyncMock())
        with patch.object(mod.asyncio, "wait_for", timeout):
            with pytest.raises(OAuthNonInteractiveError):
                asyncio.run(_wait_for_callback())
        err = capsys.readouterr().err
        assert "paste the redirect URL" in err

    def test_paste_prompt_NOT_shown_when_interactivity_suppressed(self, monkeypatch, capsys):
        """Background MCP discovery must not race the CLI/TUI stdin reader."""
        import tools.mcp_oauth as mod

        mod._oauth_port = _find_free_port()
        mock_stdin = MagicMock()
        mock_stdin.isatty.return_value = True
        monkeypatch.setattr(mod.sys, "stdin", mock_stdin)

        async def timeout(awaitable, **_kwargs):
            awaitable.close()
            raise TimeoutError

        with patch.object(mod.asyncio, "wait_for", timeout):
            with mod.suppress_interactive_oauth():
                with pytest.raises(OAuthNonInteractiveError):
                    asyncio.run(_wait_for_callback())
        err = capsys.readouterr().err
        assert "paste the redirect URL" not in err
        mock_stdin.readline.assert_not_called()

    @pytest.mark.asyncio
    async def test_repeated_cancellation_finishes_callback_cleanup(
        self,
        monkeypatch,
    ):
        import tools.mcp_oauth as mod

        paste_started = asyncio.Event()
        paste_cancelled = asyncio.Event()
        release_paste = asyncio.Event()
        paste_finished = asyncio.Event()

        async def uncooperative_paste(_result):
            paste_started.set()
            try:
                while not release_paste.is_set():
                    try:
                        await release_paste.wait()
                    except asyncio.CancelledError:  # noqa: ASYNC103 - test stub
                        paste_cancelled.set()
            finally:
                paste_finished.set()

        server = MagicMock()
        server.wait_closed = AsyncMock()
        monkeypatch.setattr(mod, "_is_interactive", lambda: True)
        monkeypatch.setattr(mod, "_raise_if_non_interactive", lambda _message: None)
        monkeypatch.setattr(mod, "_wait_for_pasted_callback", uncooperative_paste)
        monkeypatch.setattr(
            mod.asyncio,
            "start_server",
            AsyncMock(return_value=server),
        )

        waiter = asyncio.create_task(mod._make_callback_waiter(54321)())
        await paste_started.wait()
        waiter.cancel()
        await paste_cancelled.wait()
        waiter.cancel()
        await asyncio.sleep(0)

        assert waiter.done() is False
        release_paste.set()
        with pytest.raises(asyncio.CancelledError):
            await waiter

        assert paste_finished.is_set()
        server.close.assert_called_once_with()
        server.wait_closed.assert_awaited_once_with()


class TestPasteCallbackSkipToken:
    """User can type `skip` (or similar) at the paste prompt to bail out."""

    def _empty_result(self):
        return {
            "auth_code": None,
            "state": None,
            "error": None,
            "ready": asyncio.Event(),
        }

    @pytest.mark.parametrize("token", ["skip", "QUIT"])
    def test_skip_tokens_set_sentinel(self, token):
        from tools.mcp_oauth import _USER_SKIPPED_SENTINEL
        result = self._empty_result()
        _apply_pasted_callback(token + "\n", result)
        assert result["error"] == _USER_SKIPPED_SENTINEL
        assert result["auth_code"] is None

    def test_skip_does_not_overwrite_http_winner(self):
        """If HTTP listener already wrote a code, `skip` must not stomp it."""
        result = {
            "auth_code": "from_http",
            "state": "x",
            "error": None,
            "ready": asyncio.Event(),
        }
        _apply_pasted_callback("skip\n", result)
        assert result["auth_code"] == "from_http"
        assert result["error"] is None


class TestWaitForCallbackSkipIntegration:
    """_wait_for_callback maps the skip sentinel to OAuthNonInteractiveError."""

    def test_skip_raises_non_interactive_error(self, monkeypatch):
        """Skip token must raise OAuthNonInteractiveError (mcp_tool handles as non-fatal)."""
        import tools.mcp_oauth as mod
        mod._oauth_port = _find_free_port()
        monkeypatch.setattr(mod, "_is_interactive", lambda: True)
        async def paste_skip(result):
            _apply_pasted_callback("skip\n", result)

        monkeypatch.setattr(mod, "_wait_for_pasted_callback", paste_skip)
        with patch.object(mod, "_raise_if_non_interactive", lambda _message: None):
            with pytest.raises(OAuthNonInteractiveError, match="user_skipped"):
                asyncio.run(_wait_for_callback())


# ---------------------------------------------------------------------------
# poison_client_registration (GH#36767)
# ---------------------------------------------------------------------------

class TestPoisonClientRegistration:
    @pytest.mark.asyncio
    async def test_poison_backs_up_and_removes_client_and_meta(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        storage = HermesTokenStorage("srv")
        d = tmp_path / "mcp-tokens"
        d.mkdir(parents=True)
        (d / "srv.json").write_text('{"access_token": "keep-me"}')
        (d / "srv.client.json").write_text('{"client_id": "dead"}')
        (d / "srv.meta.json").write_text('{"token_endpoint": "https://idp/token"}')

        removed = await storage.poison_client_registration()

        assert removed is True
        # Client + metadata gone, forcing re-registration on the next flow.
        assert not (d / "srv.client.json").exists()
        assert not (d / "srv.meta.json").exists()
        # Backup of the client file kept for recovery.
        assert (d / "srv.client.json.bak").read_text() == '{"client_id": "dead"}'
        # Tokens are intentionally preserved.
        assert (d / "srv.json").read_text() == '{"access_token": "keep-me"}'


def test_wait_for_callback_port_in_use_reports_clear_error(monkeypatch):
    """A busy loopback callback port surfaces a clear 'already in use' error,
    not a misleading 'timed out'. Guards the stale-comment fix where the branch
    also wrongly claimed build_oauth_auth had started a server to poll."""
    import tools.mcp_oauth as mo

    monkeypatch.setattr(mo, "_is_interactive", lambda: True)
    with patch.object(mo, "_oauth_port", 54321), patch.object(
        mo.asyncio,
        "start_server",
        AsyncMock(side_effect=OSError("address already in use")),
    ):
        with pytest.raises(mo.OAuthNonInteractiveError) as excinfo:
            asyncio.run(mo._wait_for_callback())

    msg = str(excinfo.value)
    assert "54321" in msg
    assert "already in use" in msg
    assert "timed out" not in msg


# ---------------------------------------------------------------------------
# Figma remote MCP DCR allowlist workarounds
# ---------------------------------------------------------------------------


def test_figma_provider_defaults_set_allowlisted_client_name():
    from tools.mcp_oauth import (
        apply_oauth_provider_defaults,
        _FIGMA_DCR_CLIENT_NAME,
        _FIGMA_DEFAULT_SCOPE,
    )

    cfg = apply_oauth_provider_defaults(
        {},
        server_name="figma",
        server_url="https://mcp.figma.com/mcp",
    )
    assert cfg["client_name"] == _FIGMA_DCR_CLIENT_NAME
    assert cfg["scope"] == _FIGMA_DEFAULT_SCOPE


def test_humanize_non_registration_403_passthrough():
    from tools.mcp_oauth import humanize_oauth_registration_error

    assert (
        humanize_oauth_registration_error(
            "linear",
            RuntimeError("HTTP 403: insufficient_scope"),
            server_url="https://mcp.linear.app/mcp",
        )
        is None
    )
