"""Static invariant tests for native-async lock boundaries."""

from __future__ import annotations

import ast
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_DIRECTORIES = (
    "agent",
    "tools",
    "plugins",
    "hermes_cli",
    "gateway",
    "providers",
)
_SOURCE_FILES = (
    "run_agent.py",
    "model_tools.py",
    "mini_swe_runner.py",
    "batch_runner.py",
    "hermes_state.py",
    "hermes_state_portability.py",
    "trajectory_compressor.py",
)


class _AwaitFinder(ast.NodeVisitor):
    """Find awaits executed in the current lexical block only."""

    def __init__(self) -> None:
        self.found = False

    def visit_Await(self, node: ast.Await) -> None:  # noqa: N802 - ast API
        self.found = True

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:  # noqa: N802 - ast API
        self.found = True

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:  # noqa: N802 - ast API
        self.found = True

    def visit_AsyncFunctionDef(  # noqa: N802 - ast API
        self,
        node: ast.AsyncFunctionDef,
    ) -> None:
        return

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802 - ast API
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802 - ast API
        return


def _contains_await(statements: list[ast.stmt]) -> bool:
    finder = _AwaitFinder()
    for statement in statements:
        finder.visit(statement)
    return finder.found


def _target_name(target: ast.expr) -> str | None:
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return None


def _is_sync_lock_factory(value: ast.expr | None) -> bool:
    if not isinstance(value, ast.Call):
        return False
    factory = ast.unparse(value.func)
    if factory in {"threading.Lock", "threading.RLock", "Lock", "RLock"}:
        return True
    if factory != "field":
        return False
    for keyword in value.keywords:
        if keyword.arg == "default_factory" and ast.unparse(keyword.value) in {
            "threading.Lock",
            "threading.RLock",
            "Lock",
            "RLock",
        }:
            return True
    return False


def _sync_lock_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and _is_sync_lock_factory(node.value):
            names.update(
                name
                for target in node.targets
                if (name := _target_name(target)) is not None
            )
        elif isinstance(node, ast.AnnAssign) and _is_sync_lock_factory(node.value):
            if name := _target_name(node.target):
                names.add(name)
    return names


def _source_paths() -> list[Path]:
    paths = [
        path
        for directory in _SOURCE_DIRECTORIES
        for path in (_PROJECT_ROOT / directory).rglob("*.py")
    ]
    paths.extend(_PROJECT_ROOT / filename for filename in _SOURCE_FILES)
    return sorted(paths)


def test_synchronous_locks_do_not_span_awaits() -> None:
    violations: list[str] = []
    for path in _source_paths():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        lock_names = _sync_lock_names(tree)
        if not lock_names:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.With, ast.AsyncWith)):
                continue
            held_locks = {
                name
                for item in node.items
                if (name := _target_name(item.context_expr)) in lock_names
            }
            if held_locks and _contains_await(node.body):
                relative_path = path.relative_to(_PROJECT_ROOT)
                violations.append(
                    f"{relative_path}:{node.lineno}: {', '.join(sorted(held_locks))}"
                )

    assert not violations, (
        "threading.Lock/RLock must not span await boundaries:\n"
        + "\n".join(violations)
    )
