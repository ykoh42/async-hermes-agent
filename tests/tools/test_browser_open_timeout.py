"""Tests for browser first-open timeout and timeout diagnostics."""

from unittest.mock import AsyncMock, patch

import pytest

import tools.browser_tool as bt


@pytest.fixture(autouse=True)
def _reset_browser_caches():
    bt._cached_command_timeout = None
    bt._command_timeout_resolved = False
    yield
    bt._cached_command_timeout = None
    bt._command_timeout_resolved = False


class TestOpenCommandTimeout:
    pytestmark = pytest.mark.asyncio

    async def test_first_open_uses_longer_floor(self, monkeypatch):
        monkeypatch.setattr(bt, "_get_command_timeout", AsyncMock(return_value=30))
        assert (
            await bt._get_open_command_timeout(first_open=True)
            == bt.MIN_FIRST_OPEN_TIMEOUT
        )
        assert await bt._get_open_command_timeout(first_open=False) == bt.MIN_OPEN_TIMEOUT

    async def test_respects_config_above_floor(self, monkeypatch):
        monkeypatch.setattr(bt, "_get_command_timeout", AsyncMock(return_value=180))
        assert await bt._get_open_command_timeout(first_open=True) == 180
        assert await bt._get_open_command_timeout(first_open=False) == 180


class TestSandboxBypass:
    pytestmark = pytest.mark.asyncio

    async def test_docker_triggers_bypass(self, monkeypatch):
        monkeypatch.setattr(bt, "_running_in_docker", AsyncMock(return_value=True))
        assert await bt._needs_chromium_sandbox_bypass() is True

    async def test_apparmor_userns_triggers_bypass(self, monkeypatch, tmp_path):
        monkeypatch.setattr(bt, "_running_in_docker", AsyncMock(return_value=False))
        monkeypatch.setattr(bt.os, "geteuid", lambda: 1000)
        sysctl = tmp_path / "apparmor_restrict_unprivileged_userns"
        sysctl.write_text("1\n", encoding="utf-8")
        real_open = bt.aiofiles.open

        def async_open(path, *args, **kwargs):
            if "apparmor_restrict_unprivileged_userns" in str(path):
                return real_open(sysctl, *args, **kwargs)
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(bt.aiofiles, "open", async_open)
        assert await bt._needs_chromium_sandbox_bypass() is True


class TestTimeoutErrorFormatting:
    pytestmark = pytest.mark.asyncio

    async def test_includes_stderr_detail(self):
        err = await bt._format_browser_timeout_error(
            "open",
            120,
            "",
            "Daemon process exited during startup",
        )
        assert "120 seconds" in err
        assert "Daemon process exited" in err


    async def test_local_install_hint(self, monkeypatch):
        monkeypatch.setattr(bt, "_is_local_mode", AsyncMock(return_value=True))
        monkeypatch.setattr(bt, "_running_in_docker", AsyncMock(return_value=False))
        err = await bt._format_browser_timeout_error("open", 60, "", "")
        assert "agent-browser install --with-deps" in err


class TestReadCommandOutputFiles:
    pytestmark = pytest.mark.asyncio

    async def test_reads_stdout_and_stderr(self, tmp_path):
        stdout_path = tmp_path / "out"
        stderr_path = tmp_path / "err"
        stdout_path.write_text("ok", encoding="utf-8")
        stderr_path.write_text("warn", encoding="utf-8")
        stdout, stderr = await bt._read_command_output_files(
            str(stdout_path), str(stderr_path)
        )
        assert stdout == "ok"
        assert stderr == "warn"


class TestBrowserNavigateOpenTimeout:
    pytestmark = pytest.mark.asyncio

    async def test_first_navigation_uses_first_open_timeout(self, monkeypatch):
        captured: dict = {}

        async def fake_run(task_id, command, args, timeout=None):
            if command == "open":
                captured["timeout"] = timeout
            return {"success": True, "data": {"title": "t", "url": args[0] if args else ""}}

        monkeypatch.setattr(
            bt,
            "_get_open_command_timeout",
            AsyncMock(side_effect=lambda first_open=False: 120 if first_open else 60),
        )
        monkeypatch.setattr(bt, "_run_browser_command", fake_run)
        monkeypatch.setattr(
            bt,
            "_get_session_info",
            AsyncMock(return_value={"_first_nav": True, "features": {}}),
        )
        monkeypatch.setattr(bt, "_is_camofox_mode", AsyncMock(return_value=False))
        monkeypatch.setattr(bt, "_is_local_backend", AsyncMock(return_value=True))
        monkeypatch.setattr(bt, "_is_local_sidecar_key", lambda key: False)
        monkeypatch.setattr(
            bt,
            "_navigation_session_key",
            AsyncMock(side_effect=lambda task_id, url: task_id),
        )
        monkeypatch.setattr(bt, "_maybe_start_recording", AsyncMock(return_value=None))
        monkeypatch.setattr(bt, "check_website_access", AsyncMock(return_value=None))
        monkeypatch.setattr(bt, "_is_always_blocked_url", AsyncMock(return_value=False))

        await bt.browser_navigate("https://example.com", task_id="task-1")
        assert captured["timeout"] == 120
