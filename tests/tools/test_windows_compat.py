"""Tests for Windows compatibility of process management code.

Verifies that os.setsid and os.killpg are never called unconditionally,
and that each module uses a platform guard before invoking POSIX-only functions.
"""

import ast
import pytest
from pathlib import Path

# Files that must have Windows-safe process management
GUARDED_FILES = [
    "tools/environments/local.py",
    "tools/process_registry.py",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _get_preexec_fn_values(filepath: Path) -> list:
    """Find all preexec_fn= keyword arguments in Popen calls."""
    source = filepath.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(filepath))
    values = []
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "preexec_fn":
            values.append(ast.dump(node.value))
    return values


class TestNoUnconditionalSetsid:
    """preexec_fn must never be a bare os.setsid reference."""

    @pytest.mark.parametrize("relpath", GUARDED_FILES)
    def test_preexec_fn_is_guarded(self, relpath):
        filepath = PROJECT_ROOT / relpath
        if not filepath.exists():
            pytest.skip(f"{relpath} not found")
        values = _get_preexec_fn_values(filepath)
        for val in values:
            # A bare os.setsid would be: Attribute(value=Name(id='os'), attr='setsid')
            assert "attr='setsid'" not in val or "IfExp" in val or "None" in val, (
                f"{relpath} has unconditional preexec_fn=os.setsid"
            )


class TestStartNewSession:
    """All guarded files must use start_new_session=True instead of preexec_fn."""

    @pytest.mark.parametrize("relpath", GUARDED_FILES)
    def test_uses_start_new_session(self, relpath):
        """Each guarded file must use start_new_session=True for process isolation."""
        filepath = PROJECT_ROOT / relpath
        if not filepath.exists():
            pytest.skip(f"{relpath} not found")
        source = filepath.read_text(encoding="utf-8")
        # Files should use start_new_session=True, not preexec_fn
        assert "preexec_fn" not in source, (
            f"{relpath} still uses preexec_fn; use start_new_session=True instead"
        )
        assert "start_new_session" in source, (
            f"{relpath} missing start_new_session=True in Popen call"
        )


class TestIsWindowsConstant:
    """Each guarded file must define _IS_WINDOWS."""

    @pytest.mark.parametrize("relpath", GUARDED_FILES)
    def test_has_is_windows(self, relpath):
        filepath = PROJECT_ROOT / relpath
        if not filepath.exists():
            pytest.skip(f"{relpath} not found")
        source = filepath.read_text(encoding="utf-8")
        assert "_IS_WINDOWS" in source or "os.name" in source, (
            f"{relpath} missing _IS_WINDOWS platform guard"
        )


class TestKillpgGuarded:
    """os.killpg must always be behind a platform check."""

    @pytest.mark.parametrize("relpath", GUARDED_FILES)
    def test_no_unguarded_killpg(self, relpath):
        filepath = PROJECT_ROOT / relpath
        if not filepath.exists():
            pytest.skip(f"{relpath} not found")
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
        parents: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent

        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "os"
                and node.func.attr in {"killpg", "getpgid"}
            ):
                continue

            guards: list[str] = []
            ancestor = parents.get(node)
            while ancestor is not None:
                if isinstance(ancestor, ast.If):
                    guards.append(ast.unparse(ancestor.test))
                ancestor = parents.get(ancestor)
            assert any(
                "_IS_WINDOWS" in guard or "os.name" in guard
                for guard in guards
            ), (
                f"{relpath}:{node.lineno} has no platform guard around "
                f"os.{node.func.attr}"
            )
