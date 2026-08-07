"""Verify that terminal command timeouts preserve partial output."""
import pytest

from tools.environments.local import LocalEnvironment


class TestTimeoutPreservesPartialOutput:
    """When a command times out, any output captured before the deadline
    should be included in the result — not discarded."""

    @pytest.mark.asyncio
    async def test_timeout_includes_partial_output(self, tmp_path):
        """A command that prints then sleeps past the deadline should
        return both the printed text and the timeout notice."""
        env = LocalEnvironment(str(tmp_path))
        result = await env.execute("echo 'hello from test' && sleep 30", timeout=0.1)

        assert result["returncode"] == 124
        assert "hello from test" in result["output"]
        assert "timed out" in result["output"].lower()

    @pytest.mark.asyncio
    async def test_timeout_with_no_output(self, tmp_path):
        """A command that produces nothing before timeout should still
        return a clean timeout message."""
        env = LocalEnvironment(str(tmp_path))
        result = await env.execute("sleep 30", timeout=0.1)

        assert result["returncode"] == 124
        assert "timed out" in result["output"].lower()
        assert not result["output"].startswith("\n")
