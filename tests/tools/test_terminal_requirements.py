"""Terminal requirement checks keep their upstream behavior asynchronously."""

import pytest

from tools.terminal_tool import check_terminal_requirements


@pytest.mark.asyncio
async def test_local_terminal_requirements(monkeypatch) -> None:
    monkeypatch.setenv("TERMINAL_ENV", "local")
    assert await check_terminal_requirements() is True
