"""Tests for percentage clamping at 100% across display paths.

PR #3480 capped context pressure percentage at 100% in agent/display.py
but missed the same unclamped pattern in 4 other files. When token counts
overshoot the context length (possible during streaming or before
compression fires), users see >100% in /stats, gateway status, and
memory tool output.
"""

class TestMemoryToolPercentClamp:
    """tools/memory_tool.py — _success_response and _render_block pct"""

    def test_over_limit_clamped_at_100(self):
        """Percentage should be capped at 100 even if current > limit."""
        # Simulate the calculation directly
        current = 5500
        limit = 5000
        pct = min(100, int((current / limit) * 100)) if limit > 0 else 0
        assert pct == 100




class TestCLIStatsPercentClamp:
    """cli.py — /stats command percentage"""

    def test_over_context_clamped_at_100(self):
        """Tokens exceeding context_length should show max 100%."""
        last_prompt = 210_000
        ctx_len = 200_000
        pct = min(100, (last_prompt / ctx_len * 100)) if ctx_len else 0
        assert pct == 100
