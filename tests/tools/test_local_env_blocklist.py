import os

import pytest
from blockbuster import BlockBuster
from pyleak import no_event_loop_blocking, no_task_leaks
from pyleak.eventloop import LeakAction

from tools.environments.local import (
    LocalEnvironment,
    _is_hermes_internal_secret,
    _sanitize_subprocess_env,
)


def test_provider_credentials_are_removed():
    result = _sanitize_subprocess_env(
        {"PATH": "/bin", "OPENAI_API_KEY": "secret", "USER_VALUE": "ok"}
    )
    assert result["PATH"] == "/bin"
    assert result["USER_VALUE"] == "ok"
    assert "OPENAI_API_KEY" not in result


def test_dynamic_auxiliary_and_relay_secrets_are_removed():
    assert _is_hermes_internal_secret("AUXILIARY_VISION_API_KEY")
    assert _is_hermes_internal_secret("GATEWAY_RELAY_MAIN_SECRET")
    result = _sanitize_subprocess_env(
        {
            "AUXILIARY_VISION_API_KEY": "secret",
            "GATEWAY_RELAY_MAIN_SECRET": "secret",
            "GATEWAY_RELAY_URL": "https://example.test",
        }
    )
    assert "AUXILIARY_VISION_API_KEY" not in result
    assert "GATEWAY_RELAY_MAIN_SECRET" not in result
    assert result["GATEWAY_RELAY_URL"] == "https://example.test"


@pytest.mark.asyncio
async def test_local_environment_preserves_public_constructor_and_native_async_io(
    tmp_path,
):
    environment = LocalEnvironment(
        cwd=str(tmp_path),
        timeout=10,
        env={
            "HERMES_LOCAL_ENV_PROBE": "visible",
            "OPENAI_API_KEY": "must-not-leak",
        },
    )

    async with (
        no_event_loop_blocking(action=LeakAction.RAISE, threshold=0.1),
        no_task_leaks(action=LeakAction.RAISE),
    ):
        blocker = BlockBuster()
        blocker.activate()
        try:
            result = await environment.execute(
                'printf "%s" "$HERMES_LOCAL_ENV_PROBE"'
            )
        finally:
            blocker.deactivate()

    assert environment.cwd == os.path.normpath(str(tmp_path))
    assert environment.timeout == 10
    assert result == {"output": "visible", "returncode": 0}
    assert "OPENAI_API_KEY" not in environment.env


def test_local_environment_keeps_upstream_defaults(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    environment = LocalEnvironment()

    assert environment.cwd == os.path.normpath(str(tmp_path))
    assert environment.timeout == 60
