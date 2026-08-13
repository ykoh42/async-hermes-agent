"""Credential isolation for Mem0's owned local transform workers."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agent.secret_scope import (
    is_multiplex_active,
    reset_secret_scope,
    set_multiplex_active,
    set_secret_scope,
)
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from plugins.memory.mem0 import _native_worker


_CREDENTIAL_ENV = (
    "OPENAI_API_KEY",
    "TELEGRAM_BOT_TOKEN",
    "AUXILIARY_VISION_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_PROFILE",
    "AWS_SHARED_CREDENTIALS_FILE",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "AWS_ROLE_ARN",
    "AWS_ROLE_SESSION_NAME",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
    "AWS_CONTAINER_AUTHORIZATION_TOKEN",
    "AZURE_CLIENT_ID",
    "AZURE_CLIENT_SECRET",
    "AZURE_TENANT_ID",
    "AZURE_CLIENT_CERTIFICATE_PATH",
    "AZURE_CLIENT_CERTIFICATE_PASSWORD",
    "AZURE_CLIENT_SEND_CERTIFICATE_CHAIN",
    "AZURE_FEDERATED_TOKEN_FILE",
    "AZURE_TOKEN_CREDENTIALS",
    "AZURE_USERNAME",
    "AZURE_PASSWORD",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "VERTEX_CREDENTIALS_PATH",
    "CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE",
)


class _FakeStdin:
    def __init__(self, process: "_FakeProcess") -> None:
        self._process = process

    def write(self, data: bytes) -> None:
        if b'"operation":"close"' in data.replace(b" ", b""):
            self._process.returncode = 0
            self._process.finished.set()

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        return None

    async def wait_closed(self) -> None:
        return None


class _FakeProcess:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self.finished = asyncio.Event()
        self.stdin = _FakeStdin(self)
        self.stdout = object()

    async def wait(self) -> int:
        await self.finished.wait()
        assert self.returncode is not None
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15
        self.finished.set()


async def _start_worker_for_profile(home: Path) -> None:
    home_token = set_hermes_home_override(home)
    secret_token = set_secret_scope({"OPENAI_API_KEY": f"scoped-{home.name}"})
    worker = _native_worker.NativeWorker("spacy")
    try:
        await worker._start()
        await worker.close()
    finally:
        reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)


@pytest.mark.asyncio
async def test_worker_spawn_scrubs_credentials_and_preserves_profile_home(
    monkeypatch,
    tmp_path,
):
    captured: list[dict[str, str]] = []

    async def locate(_module_name):
        return Path("/fake/spacy/__init__.py"), True

    async def spawn(*_args, **kwargs):
        captured.append(dict(kwargs["env"]))
        return _FakeProcess()

    for name in _CREDENTIAL_ENV:
        monkeypatch.setenv(name, f"process-{name.lower()}")
    benign = {
        "SAFE_MEM0_WORKER_SENTINEL": "retained",
        "AWS_REGION": "ap-northeast-2",
        "AWS_DEFAULT_REGION": "us-west-2",
        "AWS_CONFIG_FILE": "/non-secret/aws-config",
        "AZURE_AUTHORITY_HOST": "login.example.test",
        "GOOGLE_CLOUD_PROJECT": "public-project-id",
    }
    for name, value in benign.items():
        monkeypatch.setenv(name, value)

    monkeypatch.setattr(_native_worker, "_locate_source_module", locate)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    previous_multiplex = is_multiplex_active()
    set_multiplex_active(True)
    try:
        homes = [tmp_path / "profile-a", tmp_path / "profile-b"]
        await asyncio.gather(*(_start_worker_for_profile(home) for home in homes))
    finally:
        set_multiplex_active(previous_multiplex)

    assert {env["HERMES_HOME"] for env in captured} == {str(home) for home in homes}
    for environment in captured:
        assert all(name not in environment for name in _CREDENTIAL_ENV)
        assert environment["PYTHONUNBUFFERED"] == "1"
        assert {name: environment.get(name) for name in benign} == benign


@pytest.mark.asyncio
async def test_worker_spawn_is_fail_safe_without_secret_scope(monkeypatch, tmp_path):
    captured: list[dict[str, str]] = []

    async def locate(_module_name):
        return Path("/fake/spacy/__init__.py"), True

    async def spawn(*_args, **kwargs):
        captured.append(dict(kwargs["env"]))
        return _FakeProcess()

    for name in _CREDENTIAL_ENV:
        monkeypatch.setenv(name, f"foreign-{name.lower()}")
    monkeypatch.setattr(_native_worker, "_locate_source_module", locate)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    previous_multiplex = is_multiplex_active()
    set_multiplex_active(True)
    home_token = set_hermes_home_override(tmp_path / "unscoped")
    worker = _native_worker.NativeWorker("spacy")
    try:
        await worker._start()
        await worker.close()
    finally:
        reset_hermes_home_override(home_token)
        set_multiplex_active(previous_multiplex)

    assert len(captured) == 1
    assert all(name not in captured[0] for name in _CREDENTIAL_ENV)
    assert captured[0]["HERMES_HOME"] == str(tmp_path / "unscoped")
