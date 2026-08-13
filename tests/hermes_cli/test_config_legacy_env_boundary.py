"""Boundary tests for legacy synchronous config dotenv helpers.

The CLI migration/reload writer graph is not shipped as a runtime entry point.
Retained coroutine paths must use their existing native-async dotenv readers
instead of calling these legacy helpers from the event loop.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from unittest.mock import AsyncMock

import aiofiles
import pytest
from blockbuster import BlockBuster


_LEGACY_SYNC_ENV_HELPERS = {
    "get_env_value",
    "load_env",
    "migrate_config",
    "reload_env",
    "sanitize_env_file",
}


def _retained_runtime_paths(root: Path) -> list[Path]:
    return [
        root / "run_agent.py",
        root / "batch_runner.py",
        root / "mini_swe_runner.py",
        root / "trajectory_compressor.py",
        root / "hermes_state.py",
        root / "hermes_state_common.py",
        root / "hermes_state_portability.py",
        root / "hermes_state_schema.py",
        root / "model_tools.py",
        root / "toolsets.py",
        *(root / "agent").rglob("*.py"),
        *(root / "gateway").rglob("*.py"),
        *(root / "tools").rglob("*.py"),
        *(root / "plugins").rglob("*.py"),
        *(root / "providers").rglob("*.py"),
        *(
            path
            for path in (root / "hermes_cli").rglob("*.py")
            if path.name not in {"config.py", "config_migrations.py"}
        ),
    ]


def _async_reaches_legacy_config_helper(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    direct_aliases = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module in {"hermes_cli.config", "tools.xai_http"}
        for alias in node.names
        if (
            node.module == "hermes_cli.config"
            and alias.name in _LEGACY_SYNC_ENV_HELPERS
        )
        or (node.module == "tools.xai_http" and alias.name == "get_env_value")
    }
    legacy_module_aliases = {
        alias.asname or alias.name.rsplit(".", 1)[-1]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name in {"hermes_cli.config", "tools.xai_http"}
    }
    legacy_module_aliases.update(
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module in {"hermes_cli", "tools"}
        for alias in node.names
        if (node.module, alias.name)
        in {("hermes_cli", "config"), ("tools", "xai_http")}
    )

    local_functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    owners: dict[str, str | None] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            local_functions[node.name] = node
            owners[node.name] = None
        elif isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qualified = f"{node.name}.{child.name}"
                    local_functions[qualified] = child
                    owners[qualified] = node.name
    direct_legacy: dict[str, list[tuple[int, str]]] = {}
    local_calls: dict[str, set[str]] = {}
    for name, function in local_functions.items():
        direct: set[tuple[int, str]] = set()
        edges: set[str] = set()
        for node in ast.walk(function):
            if (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id in direct_aliases
            ):
                direct.add((node.lineno, node.id))
            if (
                isinstance(node, ast.Attribute)
                and node.attr in _LEGACY_SYNC_ENV_HELPERS
                and isinstance(node.value, ast.Name)
                and node.value.id in legacy_module_aliases | {"config", "xai_http"}
            ):
                direct.add((node.lineno, node.attr))
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                if node.func.id in local_functions:
                    edges.add(node.func.id)
            elif (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in {"self", "cls"}
                and owners[name] is not None
            ):
                qualified = f"{owners[name]}.{node.func.attr}"
                if qualified in local_functions:
                    edges.add(qualified)
        direct_legacy[name] = sorted(direct)
        local_calls[name] = edges

    violations: list[str] = []
    for name, function in local_functions.items():
        if not isinstance(function, ast.AsyncFunctionDef):
            continue
        stack = [name]
        visited: set[str] = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            for lineno, helper in direct_legacy[current]:
                violations.append(
                    f"{path.name}:{function.lineno}:{name} -> "
                    f"{current}:{lineno}:{helper}"
                )
            stack.extend(local_calls[current] - visited)
    return violations


def test_retained_async_call_graph_does_not_reach_legacy_sync_env_helpers() -> None:
    root = Path(__file__).resolve().parents[2]
    violations = [
        violation
        for path in _retained_runtime_paths(root)
        for violation in _async_reaches_legacy_config_helper(path)
    ]
    assert violations == []


def test_legacy_and_retained_env_helpers_are_explicitly_separated() -> None:
    from hermes_cli import config

    for name in _LEGACY_SYNC_ENV_HELPERS:
        assert not inspect.iscoroutinefunction(getattr(config, name))
    assert inspect.iscoroutinefunction(config.get_env_value_prefer_dotenv)


def test_legacy_load_env_cache_key_prevents_cross_profile_value_reuse(
    tmp_path, monkeypatch
) -> None:
    from hermes_cli import config

    home_a = tmp_path / "a"
    home_b = tmp_path / "b"
    home_a.mkdir()
    home_b.mkdir()
    (home_a / ".env").write_text("TOKEN=profile-a\n", encoding="utf-8")
    (home_b / ".env").write_text("TOKEN=profile-b\n", encoding="utf-8")
    config.invalidate_env_cache()
    try:
        monkeypatch.setenv("HERMES_HOME", str(home_a))
        assert config.load_env() == {"TOKEN": "profile-a"}
        monkeypatch.setenv("HERMES_HOME", str(home_b))
        assert config.load_env() == {"TOKEN": "profile-b"}
    finally:
        config.invalidate_env_cache()


@pytest.mark.asyncio
async def test_retained_dotenv_consumers_never_call_legacy_helpers(
    tmp_path, monkeypatch
) -> None:
    from agent import credential_pool, web_search_provider
    from hermes_cli import config
    from tools import tts_tool
    from tools.environments import docker
    from tools.xai_http import resolve_xai_http_credentials

    (tmp_path / ".env").write_text(
        "DEEPSEEK_API_KEY=deepseek-dotenv\n"
        "TOKEN=dotenv-token\n"
        "XAI_API_KEY=xai-dotenv\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("TOKEN", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("retained runtime called legacy sync config helper")

    for name in _LEGACY_SYNC_ENV_HELPERS:
        monkeypatch.setattr(config, name, forbidden)

    monkeypatch.setattr(
        credential_pool,
        "_load_auth_store",
        AsyncMock(return_value={}),
    )
    oauth_pool = AsyncMock()
    oauth_pool.select.return_value = None
    oauth_pool.try_refresh_matching.return_value = None
    monkeypatch.setattr(
        credential_pool,
        "load_pool",
        AsyncMock(return_value=oauth_pool),
    )

    # Warm aiofiles' executor before BlockBuster starts inspecting thread
    # startup; the audited operations themselves still run under BlockBuster.
    async with aiofiles.open(tmp_path / ".env", encoding="utf-8") as handle:
        await handle.read(0)

    blocker = BlockBuster()
    blocker.activate()
    try:
        assert await config.get_env_value_prefer_dotenv("TOKEN") == "dotenv-token"
        assert (await docker._load_hermes_env_vars())["TOKEN"] == "dotenv-token"
        assert await web_search_provider.get_provider_env("TOKEN") == "dotenv-token"
        assert await tts_tool.get_env_value("TOKEN") == "dotenv-token"
        entries = []
        changed, sources = await credential_pool._seed_from_env(
            "deepseek",
            entries,
        )
        assert changed is True
        assert sources == {"env:DEEPSEEK_API_KEY"}
        assert (await resolve_xai_http_credentials())["api_key"] == "xai-dotenv"
    finally:
        blocker.deactivate()
