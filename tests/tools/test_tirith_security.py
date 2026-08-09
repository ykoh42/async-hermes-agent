"""Tests for the native-async Tirith security scanning wrapper."""

import asyncio
import io
import json
import os
import tarfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiofiles
import pytest
import pytest_asyncio
from blockbuster import BlockBuster
from pyleak import no_event_loop_blocking, no_task_leaks
from pyleak.eventloop import LeakAction

import tools.tirith_security as tirith
import tools.approval as approval
from tools.tirith_security import check_command_security, ensure_installed


_CFG = {
    "tirith_enabled": True,
    "tirith_path": "tirith",
    "tirith_timeout": 5,
    "tirith_fail_open": True,
}


class _Process:
    def __init__(self, returncode=0, stdout="", stderr="", error=None):
        self.returncode = returncode
        self._stdout = stdout.encode()
        self._stderr = stderr.encode()
        self._error = error
        self.killed = False
        self.waited = False

    async def communicate(self):
        if self._error is not None:
            raise self._error
        return self._stdout, self._stderr

    def kill(self):
        self.killed = True

    async def wait(self):
        self.waited = True
        return self.returncode


class _WaitingProcess(_Process):
    def __init__(self):
        super().__init__(returncode=None)
        self.started = asyncio.Event()
        self.release_communicate = asyncio.Event()
        self.communicate_completed = asyncio.Event()

    async def communicate(self):
        self.started.set()
        await self.release_communicate.wait()
        self.communicate_completed.set()
        return self._stdout, self._stderr


@pytest_asyncio.fixture(autouse=True)
async def _reset_state():
    tirith._resolved_path = "tirith"
    tirith._install_task = None
    tirith._install_failure_reason = ""
    tirith._crash_count = 0
    tirith._circuit_open = False
    tirith._reset_spawn_warning_state()
    yield
    task = tirith._install_task
    if task is not None and not task.done():
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    tirith._resolved_path = None
    tirith._install_task = None
    tirith._install_failure_reason = ""
    tirith._crash_count = 0
    tirith._circuit_open = False


def _json_stdout(findings=None, summary=""):
    return json.dumps({"findings": findings or [], "summary": summary})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("returncode", "findings", "summary", "action"),
    [
        (0, [], "", "allow"),
        (1, [{"rule_id": "homograph_url", "severity": "high"}], "homograph", "block"),
        (2, [{"rule_id": "shortened_url", "severity": "medium"}], "short URL", "warn"),
    ],
)
async def test_exit_code_is_verdict_source(
    monkeypatch,
    returncode,
    findings,
    summary,
    action,
):
    monkeypatch.setattr(tirith, "_load_security_config", AsyncMock(return_value=_CFG))
    spawn = AsyncMock(
        return_value=_Process(
            returncode,
            _json_stdout(findings, summary),
        )
    )
    monkeypatch.setattr(tirith.asyncio, "create_subprocess_exec", spawn)

    result = await check_command_security("echo hello")

    assert result["action"] == action
    assert result["findings"] == findings
    assert result["summary"] == summary


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("returncode", "action", "summary"),
    [
        (0, "allow", ""),
        (1, "block", "security issue detected (details unavailable)"),
        (2, "warn", "security warning detected (details unavailable)"),
    ],
)
async def test_invalid_json_never_overrides_exit_code(
    monkeypatch,
    returncode,
    action,
    summary,
):
    monkeypatch.setattr(tirith, "_load_security_config", AsyncMock(return_value=_CFG))
    monkeypatch.setattr(
        tirith.asyncio,
        "create_subprocess_exec",
        AsyncMock(return_value=_Process(returncode, "not json")),
    )

    result = await check_command_security("command")

    assert result["action"] == action
    assert result["summary"] == summary


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_open,action", [(True, "allow"), (False, "block")])
async def test_spawn_failure_respects_fail_open(monkeypatch, fail_open, action):
    config = {**_CFG, "tirith_fail_open": fail_open}
    monkeypatch.setattr(tirith, "_load_security_config", AsyncMock(return_value=config))
    monkeypatch.setattr(
        tirith.asyncio,
        "create_subprocess_exec",
        AsyncMock(side_effect=FileNotFoundError("tirith missing")),
    )

    result = await check_command_security("echo hi")

    assert result["action"] == action
    assert ("unavailable" if fail_open else "fail-closed") in result["summary"]


@pytest.mark.asyncio
async def test_timeout_kills_process_and_respects_fail_closed(monkeypatch):
    config = {**_CFG, "tirith_fail_open": False}
    process = _Process(error=TimeoutError())
    monkeypatch.setattr(tirith, "_load_security_config", AsyncMock(return_value=config))
    monkeypatch.setattr(
        tirith.asyncio,
        "create_subprocess_exec",
        AsyncMock(return_value=process),
    )

    result = await check_command_security("slow command")

    assert result["action"] == "block"
    assert "fail-closed" in result["summary"]
    assert process.killed is True
    assert process.waited is True


@pytest.mark.asyncio
async def test_repeated_cancellation_kills_and_reaps_scanner(monkeypatch):
    process = _WaitingProcess()
    monkeypatch.setattr(tirith, "_load_security_config", AsyncMock(return_value=_CFG))
    monkeypatch.setattr(
        tirith.asyncio,
        "create_subprocess_exec",
        AsyncMock(return_value=process),
    )
    task = asyncio.create_task(check_command_security("command"))
    await process.started.wait()

    task.cancel()
    while not process.killed:
        await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)

    assert task.done() is False
    process.release_communicate.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert process.killed is True
    assert process.communicate_completed.is_set()


@pytest.mark.asyncio
async def test_unknown_exit_code_respects_fail_closed(monkeypatch):
    config = {**_CFG, "tirith_fail_open": False}
    monkeypatch.setattr(tirith, "_load_security_config", AsyncMock(return_value=config))
    monkeypatch.setattr(
        tirith.asyncio,
        "create_subprocess_exec",
        AsyncMock(return_value=_Process(99)),
    )

    result = await check_command_security("command")

    assert result["action"] == "block"
    assert "exit code 99" in result["summary"]


@pytest.mark.asyncio
async def test_disabled_scanner_does_not_spawn(monkeypatch):
    config = {**_CFG, "tirith_enabled": False}
    spawn = AsyncMock()
    monkeypatch.setattr(tirith, "_load_security_config", AsyncMock(return_value=config))
    monkeypatch.setattr(tirith.asyncio, "create_subprocess_exec", spawn)

    result = await check_command_security("rm -rf /")

    assert result == {"action": "allow", "findings": [], "summary": ""}
    spawn.assert_not_awaited()


@pytest.mark.asyncio
async def test_findings_and_summary_are_capped(monkeypatch):
    findings = [{"rule_id": f"rule_{index}"} for index in range(100)]
    monkeypatch.setattr(tirith, "_load_security_config", AsyncMock(return_value=_CFG))
    monkeypatch.setattr(
        tirith.asyncio,
        "create_subprocess_exec",
        AsyncMock(
            return_value=_Process(2, _json_stdout(findings, "x" * 1000))
        ),
    )

    result = await check_command_security("command")

    assert len(result["findings"]) == 50
    assert len(result["summary"]) == 500


@pytest.mark.asyncio
async def test_programming_errors_propagate(monkeypatch):
    monkeypatch.setattr(tirith, "_load_security_config", AsyncMock(return_value=_CFG))
    monkeypatch.setattr(
        tirith.asyncio,
        "create_subprocess_exec",
        AsyncMock(side_effect=AttributeError("programming bug")),
    )

    with pytest.raises(AttributeError, match="programming bug"):
        await check_command_security("command")


@pytest.mark.asyncio
async def test_unsupported_platform_is_silent(monkeypatch):
    monkeypatch.setattr(tirith, "_load_security_config", AsyncMock(return_value=_CFG))
    monkeypatch.setattr(tirith, "is_platform_supported", lambda: False)
    resolve = AsyncMock()
    spawn = AsyncMock()
    monkeypatch.setattr(tirith, "_resolve_tirith_path", resolve)
    monkeypatch.setattr(tirith.asyncio, "create_subprocess_exec", spawn)

    result = await check_command_security("command")

    assert result == {"action": "allow", "findings": [], "summary": ""}
    resolve.assert_not_awaited()
    spawn.assert_not_awaited()


@pytest.mark.parametrize(
    ("system", "machine", "expected"),
    [
        ("Linux", "x86_64", True),
        ("Windows", "AMD64", False),
        ("Linux", "riscv64", False),
    ],
)
def test_platform_support_matrix(monkeypatch, system, machine, expected):
    monkeypatch.setattr(tirith.platform, "system", lambda: system)
    monkeypatch.setattr(tirith.platform, "machine", lambda: machine)
    assert tirith.is_platform_supported() is expected


@pytest.mark.asyncio
async def test_explicit_path_is_authoritative(monkeypatch, tmp_path):
    binary = tmp_path / "tirith"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)
    tirith._resolved_path = None
    install = AsyncMock()
    monkeypatch.setattr(tirith, "_install_tirith", install)

    result = await tirith._resolve_tirith_path(str(binary))

    assert result == str(binary)
    install.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_install_is_cached(monkeypatch, tmp_path):
    tirith._resolved_path = None
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(tirith, "_detect_target", lambda: "aarch64-apple-darwin")
    monkeypatch.setattr(
        tirith.aiofiles.os,
        "wrap",
        lambda _function: AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        tirith,
        "_read_failure_reason",
        AsyncMock(return_value=None),
    )
    install = AsyncMock(return_value=(None, "download_failed"))
    monkeypatch.setattr(tirith, "_install_tirith", install)
    mark = AsyncMock()
    monkeypatch.setattr(tirith, "_mark_install_failed", mark)

    first = await tirith._resolve_tirith_path("tirith")
    second = await tirith._resolve_tirith_path("tirith")

    assert first == second == "tirith"
    install.assert_awaited_once()
    mark.assert_awaited_once_with("download_failed")
    assert tirith._resolved_path is tirith._INSTALL_FAILED


@pytest.mark.asyncio
async def test_ensure_installed_returns_resolved_path(monkeypatch, tmp_path):
    binary = tmp_path / "tirith"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)
    tirith._resolved_path = str(binary)
    monkeypatch.setattr(tirith, "_load_security_config", AsyncMock(return_value=_CFG))

    assert await ensure_installed() == str(binary)


@pytest.mark.asyncio
async def test_ensure_installed_starts_one_tracked_task(monkeypatch, tmp_path):
    tirith._resolved_path = None
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(tirith, "_load_security_config", AsyncMock(return_value=_CFG))
    monkeypatch.setattr(tirith, "_detect_target", lambda: "aarch64-apple-darwin")
    monkeypatch.setattr(
        tirith.aiofiles.os,
        "wrap",
        lambda _function: AsyncMock(return_value=None),
    )
    monkeypatch.setattr(tirith, "_read_failure_reason", AsyncMock(return_value=None))
    started = asyncio.Event()
    release = asyncio.Event()

    async def install(*, log_failures=True):
        del log_failures
        started.set()
        await release.wait()
        return None

    monkeypatch.setattr(tirith, "_background_install", install)

    assert await ensure_installed() is None
    first_task = tirith._install_task
    assert first_task is not None
    await started.wait()
    assert await ensure_installed() is None
    assert tirith._install_task is first_task
    release.set()
    await first_task


@pytest.mark.asyncio
async def test_disk_failure_marker_expires(monkeypatch, tmp_path):
    marker = tmp_path / ".tirith-install-failed"
    monkeypatch.setattr(tirith, "_failure_marker_path", lambda: str(marker))

    assert not await tirith._is_install_failed_on_disk()
    await tirith._mark_install_failed("download_failed")
    assert await tirith._is_install_failed_on_disk()
    old_time = time.time() - 90_000
    os.utime(marker, (old_time, old_time))
    assert not await tirith._is_install_failed_on_disk()


@pytest.mark.asyncio
async def test_hermes_bin_dir_respects_hermes_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    result = await tirith._hermes_bin_dir()

    assert result == str(tmp_path / "bin")
    assert Path(result).is_dir()


@pytest.mark.asyncio
async def test_checksum_verification(monkeypatch, tmp_path):
    archive = tmp_path / "tirith.tar.gz"
    archive.write_bytes(b"payload")
    digest = tirith.hashlib.sha256(b"payload").hexdigest()
    checksums = tmp_path / "checksums.txt"
    checksums.write_text(f"{digest}  tirith.tar.gz\n", encoding="utf-8")

    async with (
        no_event_loop_blocking(action=LeakAction.RAISE, threshold=0.1),
        no_task_leaks(action=LeakAction.RAISE),
    ):
        blocker = BlockBuster()
        blocker.activate()
        try:
            assert await tirith._verify_checksum(
                str(archive),
                str(checksums),
                "tirith.tar.gz",
            )
        finally:
            blocker.deactivate()


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="Tirith supports POSIX hosts")
async def test_real_scanner_subprocess_is_native_async(monkeypatch, tmp_path):
    binary = tmp_path / "tirith"
    binary.write_text(
        "#!/bin/sh\nprintf '%s\\n' '{\"findings\": [], \"summary\": \"warn\"}'\n"
        "exit 2\n",
        encoding="utf-8",
    )
    binary.chmod(0o755)
    config = {**_CFG, "tirith_path": str(binary)}
    monkeypatch.setattr(tirith, "_load_security_config", AsyncMock(return_value=config))
    tirith._resolved_path = None
    await aiofiles.os.stat(binary)

    async with (
        no_event_loop_blocking(action=LeakAction.RAISE, threshold=0.1),
        no_task_leaks(action=LeakAction.RAISE),
    ):
        blocker = BlockBuster()
        blocker.activate()
        try:
            result = await check_command_security("echo hello")
        finally:
            blocker.deactivate()

    assert result == {"action": "warn", "findings": [], "summary": "warn"}


@pytest.mark.asyncio
async def test_combined_guard_preserves_tirith_session_scope(monkeypatch):
    async def load_config_readonly():
        return {"approvals": {"mode": "manual"}}

    async def warn(_command):
        return {
            "action": "warn",
            "findings": [
                {
                    "rule_id": "shortened_url",
                    "severity": "MEDIUM",
                    "title": "Shortened URL",
                    "description": "destination is hidden",
                }
            ],
            "summary": "short URL",
        }

    prompt = AsyncMock(return_value="always")
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        load_config_readonly,
    )
    monkeypatch.setattr(tirith, "check_command_security", warn)
    approval._permanent_approved.clear()
    approval._session_approved.clear()
    token = approval.set_current_session_key("tirith-session")
    try:
        result = await approval.check_all_command_guards(
            "curl https://bit.ly/example",
            "local",
            approval_callback=prompt,
        )
    finally:
        approval.reset_current_session_key(token)

    assert result["approved"] is True
    assert prompt.await_args.kwargs["allow_permanent"] is False
    assert approval.is_approved("tirith-session", "tirith:shortened_url")
    assert "tirith:shortened_url" not in approval._permanent_approved


def _write_archive(tmp_path, member, payload=None):
    archive = tmp_path / "tirith-aarch64-apple-darwin.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.addfile(member, io.BytesIO(payload) if payload is not None else None)
    checksums = tmp_path / "checksums.txt"
    checksums.write_text(
        "ignored  tirith-aarch64-apple-darwin.tar.gz\n",
        encoding="utf-8",
    )
    return archive, checksums


def _download_side_effect(archive, checksums):
    async def download(url, dest, timeout=10):
        del timeout
        source_path = archive if url.endswith(".tar.gz") else checksums
        if not (url.endswith(".tar.gz") or url.endswith("checksums.txt")):
            raise AssertionError(f"unexpected URL: {url}")
        async with (
            aiofiles.open(source_path, "rb") as source,
            aiofiles.open(dest, "wb") as target,
        ):
            await target.write(await source.read())

    return download


@pytest.mark.asyncio
async def test_install_extracts_only_regular_tirith_member(monkeypatch, tmp_path):
    payload = b"#!/bin/sh\nexit 0\n"
    member = tarfile.TarInfo("bin/tirith")
    member.mode = 0o755
    member.size = len(payload)
    archive, checksums = _write_archive(tmp_path, member, payload)
    hermes_home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(tirith, "_detect_target", lambda: "aarch64-apple-darwin")
    monkeypatch.setattr(tirith, "_download_file", _download_side_effect(archive, checksums))
    monkeypatch.setattr(tirith, "_verify_checksum", AsyncMock(return_value=True))
    real_wrap = aiofiles.os.wrap
    monkeypatch.setattr(
        tirith.aiofiles.os,
        "wrap",
        lambda function: AsyncMock(return_value=None)
        if function is tirith.shutil.which
        else real_wrap(function),
    )

    path, reason = await tirith._install_tirith(log_failures=False)

    assert reason == ""
    assert path == str(hermes_home / "bin" / "tirith")
    assert Path(path).read_bytes() == payload


@pytest.mark.asyncio
async def test_install_rejects_link_member(monkeypatch, tmp_path):
    member = tarfile.TarInfo("bin/tirith")
    member.type = tarfile.SYMTYPE
    member.linkname = "/bin/sh"
    archive, checksums = _write_archive(tmp_path, member)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    monkeypatch.setattr(tirith, "_detect_target", lambda: "aarch64-apple-darwin")
    monkeypatch.setattr(tirith, "_download_file", _download_side_effect(archive, checksums))
    monkeypatch.setattr(tirith, "_verify_checksum", AsyncMock(return_value=True))
    monkeypatch.setattr(
        tirith.aiofiles.os,
        "wrap",
        lambda _function: AsyncMock(return_value=None),
    )

    path, reason = await tirith._install_tirith(log_failures=False)

    assert path is None
    assert reason == "binary_not_regular_file"


@pytest.mark.asyncio
async def test_cosign_identity_is_pinned(monkeypatch):
    process = _Process(0)
    monkeypatch.setattr(
        tirith.aiofiles.os,
        "wrap",
        lambda _function: AsyncMock(return_value="/usr/bin/cosign"),
    )
    spawn = AsyncMock(return_value=process)
    monkeypatch.setattr(tirith.asyncio, "create_subprocess_exec", spawn)

    assert await tirith._verify_cosign("/tmp/checksums", "/tmp/sig", "/tmp/cert")

    args = spawn.await_args.args
    index = args.index("--certificate-identity-regexp")
    assert "workflows/release" in args[index + 1]
    assert "refs/tags/v" in args[index + 1]


@pytest.mark.asyncio
async def test_repeated_spawn_failure_logs_once(monkeypatch, caplog):
    monkeypatch.setattr(tirith, "_load_security_config", AsyncMock(return_value=_CFG))
    monkeypatch.setattr(
        tirith.asyncio,
        "create_subprocess_exec",
        AsyncMock(side_effect=FileNotFoundError("tirith missing")),
    )

    with caplog.at_level("WARNING", logger="tools.tirith_security"):
        for _ in range(15):
            assert (await check_command_security("echo hi"))["action"] == "allow"

    warnings = [
        record for record in caplog.records if "tirith spawn failed" in record.message
    ]
    assert len(warnings) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("returncode", "findings", "action"),
    [
        (
            2,
            [{"rule_id": "lookalike_tld", "value": ".app"}],
            "allow",
        ),
        (
            2,
            [
                {"rule_id": "lookalike_tld", "value": ".app"},
                {"rule_id": "shortened_url"},
            ],
            "warn",
        ),
        (
            1,
            [{"rule_id": "lookalike_tld", "value": ".app"}],
            "block",
        ),
    ],
)
async def test_app_tld_suppression(monkeypatch, returncode, findings, action):
    monkeypatch.setattr(tirith, "_load_security_config", AsyncMock(return_value=_CFG))
    monkeypatch.setattr(
        tirith.asyncio,
        "create_subprocess_exec",
        AsyncMock(
            return_value=_Process(
                returncode,
                _json_stdout(findings, "finding"),
            )
        ),
    )

    result = await check_command_security("curl https://example.app")

    assert result["action"] == action


@pytest.mark.parametrize(
    ("finding", "expected"),
    [
        ({"rule_id": "lookalike_tld", "value": ".APP"}, True),
        ({"rule_id": "lookalike_tld", "message": "uses .app"}, True),
        ({"rule_id": "shortened_url", "value": ".app"}, False),
        ({"rule_id": "lookalike_tld", "value": ".zip"}, False),
    ],
)
def test_app_tld_finding(finding, expected):
    assert tirith._is_app_tld_finding(finding) is expected


@pytest.mark.asyncio
async def test_tempdir_failure_returns_no_space(monkeypatch):
    monkeypatch.setattr(tirith, "_detect_target", lambda: "aarch64-apple-darwin")
    monkeypatch.setattr(
        tirith.aiofiles.tempfile,
        "TemporaryDirectory",
        MagicMock(side_effect=OSError(28, "no space")),
    )

    path, reason = await tirith._install_tirith(log_failures=False)

    assert path is None
    assert reason == "no_space"
