"""Unit tests for the verification-loop policy (agent/verify_hooks.py).

The `pre_verify` user-hook aggregation lives in `hermes_cli.plugins`
(`get_pre_verify_continue_message`) and is tested in
`tests/hermes_cli/test_plugins.py`, alongside `get_pre_tool_call_block_message`.
"""

from __future__ import annotations

import pytest

from agent import verify_hooks

pytestmark = pytest.mark.asyncio


class TestMaxVerifyNudges:
    async def test_default_when_unset(self):
        assert (
            await verify_hooks.max_verify_nudges({})
            == verify_hooks.DEFAULT_MAX_VERIFY_NUDGES
        )
        assert (
            await verify_hooks.max_verify_nudges({"agent": {}})
            == verify_hooks.DEFAULT_MAX_VERIFY_NUDGES
        )

    async def test_reads_and_coerces(self):
        assert await verify_hooks.max_verify_nudges(
            {"agent": {"max_verify_nudges": 5}}
        ) == 5
        assert await verify_hooks.max_verify_nudges(
            {"agent": {"max_verify_nudges": "2"}}
        ) == 2
        assert await verify_hooks.max_verify_nudges(
            {"agent": {"max_verify_nudges": -1}}
        ) == 0

    async def test_bad_value_falls_back(self):
        assert (
            await verify_hooks.max_verify_nudges(
                {"agent": {"max_verify_nudges": "x"}}
            )
            == verify_hooks.DEFAULT_MAX_VERIFY_NUDGES
        )


class TestCodingVerifyGuidance:
    async def test_enabled_by_default(self):
        assert (
            await verify_hooks.coding_verify_guidance({})
            == verify_hooks.CODING_VERIFY_GUIDANCE
        )
        assert (
            await verify_hooks.coding_verify_guidance({"agent": {}})
            == verify_hooks.CODING_VERIFY_GUIDANCE
        )

    async def test_reads_truthy_config(self):
        cfg = {"agent": {"verify_guidance": "yes"}}
        assert await verify_hooks.coding_verify_guidance(
            cfg
        ) == verify_hooks.CODING_VERIFY_GUIDANCE

    async def test_opt_out_via_config(self):
        off = {"agent": {"verify_guidance": False}}
        assert await verify_hooks.coding_verify_guidance(off) is None
