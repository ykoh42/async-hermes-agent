"""Native-async service orchestration for LSP clients."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any
from collections.abc import Callable

from agent.lsp import eventlog
from agent.lsp.client import DIAGNOSTICS_DOCUMENT_WAIT, LSPClient
from agent.lsp.servers import (
    ServerContext,
    ServerDef,
    find_server_for_file,
    language_id_for,
)
from agent.lsp.workspace import clear_cache, resolve_workspace_for_file


logger = logging.getLogger("agent.lsp.manager")

DEFAULT_IDLE_TIMEOUT = 600
MIN_IDLE_TIMEOUT = 30


class LSPService:
    """Process-wide LSP service owned by the caller's event loop."""

    def __init__(
        self,
        *,
        enabled: bool,
        wait_mode: str,
        wait_timeout: float,
        install_strategy: str,
        binary_overrides: dict[str, list[str]] | None = None,
        env_overrides: dict[str, dict[str, str]] | None = None,
        init_overrides: dict[str, dict[str, Any]] | None = None,
        disabled_servers: list[str] | None = None,
        idle_timeout: float = DEFAULT_IDLE_TIMEOUT,
    ) -> None:
        self._enabled = enabled
        self._wait_mode = wait_mode if wait_mode in {"document", "full"} else "document"
        self._wait_timeout = wait_timeout
        self._install_strategy = install_strategy
        self._binary_overrides = binary_overrides or {}
        self._env_overrides = env_overrides or {}
        self._init_overrides = init_overrides or {}
        self._disabled_servers = set(disabled_servers or [])
        self._idle_timeout = idle_timeout

        self._clients: dict[tuple[str, str], LSPClient] = {}
        self._broken: set[tuple[str, str]] = set()
        self._spawning: dict[tuple[str, str], asyncio.Future[LSPClient | None]] = {}
        self._spawn_tasks: set[asyncio.Task[Any]] = set()
        self._last_used: dict[tuple[str, str], float] = {}
        self._idle_reaper_task: asyncio.Task[None] | None = None
        self._shutdown_task: asyncio.Task[None] | None = None
        self._delta_baseline: dict[str, list[dict[str, Any]]] = {}
        self._started = False
        self._closed = False

    @classmethod
    async def create_from_config(cls) -> LSPService | None:
        """Build a service from the async read-only Hermes configuration."""
        try:
            from hermes_cli.config import load_config_readonly

            cfg = await load_config_readonly()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.debug("LSP config load failed: %s", exc)
            return None

        lsp_cfg = (cfg.get("lsp") or {}) if isinstance(cfg, dict) else {}
        if not isinstance(lsp_cfg, dict):
            lsp_cfg = {}

        enabled = bool(lsp_cfg.get("enabled", True))
        wait_mode = lsp_cfg.get("wait_mode", "document")
        wait_timeout = float(lsp_cfg.get("wait_timeout", DIAGNOSTICS_DOCUMENT_WAIT))
        install_strategy = lsp_cfg.get("install_strategy", "auto")
        try:
            idle_timeout = float(lsp_cfg.get("idle_timeout", DEFAULT_IDLE_TIMEOUT))
        except (TypeError, ValueError):
            idle_timeout = DEFAULT_IDLE_TIMEOUT
        if 0 < idle_timeout < MIN_IDLE_TIMEOUT:
            idle_timeout = MIN_IDLE_TIMEOUT

        servers_cfg = lsp_cfg.get("servers") or {}
        disabled: list[str] = []
        binary_overrides: dict[str, list[str]] = {}
        env_overrides: dict[str, dict[str, str]] = {}
        init_overrides: dict[str, dict[str, Any]] = {}
        if isinstance(servers_cfg, dict):
            for name, sub in servers_cfg.items():
                if not isinstance(sub, dict):
                    continue
                if sub.get("disabled"):
                    disabled.append(name)
                command = sub.get("command")
                if isinstance(command, list) and command:
                    binary_overrides[name] = command
                env = sub.get("env")
                if isinstance(env, dict):
                    env_overrides[name] = {key: str(value) for key, value in env.items()}
                init = sub.get("initialization_options")
                if isinstance(init, dict):
                    init_overrides[name] = init

        return cls(
            enabled=enabled,
            wait_mode=wait_mode,
            wait_timeout=wait_timeout,
            install_strategy=install_strategy,
            binary_overrides=binary_overrides,
            env_overrides=env_overrides,
            init_overrides=init_overrides,
            disabled_servers=disabled,
            idle_timeout=idle_timeout,
        )

    def is_active(self) -> bool:
        """Return True iff this service should be consulted at all."""
        return self._enabled and not self._closed

    def _has_owned_resources(self) -> bool:
        """Return whether this service is bound to its creating event loop."""
        return self._started

    async def _ensure_started(self) -> None:
        if self._started or self._closed or not self._enabled:
            return
        self._started = True
        await self._start_idle_reaper()

    async def enabled_for(self, file_path: str) -> bool:
        """Return True iff LSP should run for this specific file."""
        if not self.is_active():
            return False
        server = find_server_for_file(file_path)
        if server is None or server.server_id in self._disabled_servers:
            return False
        workspace_root, gated_in = await resolve_workspace_for_file(file_path)
        if not (workspace_root and gated_in):
            return False
        try:
            server_root = await server.resolve_root(file_path, workspace_root) or workspace_root
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            server_root = workspace_root
        return (server.server_id, server_root) not in self._broken

    async def snapshot_baseline(self, file_path: str) -> None:
        """Snapshot current diagnostics before a write for delta filtering."""
        if not await self.enabled_for(file_path):
            return
        await self._ensure_started()
        absolute_path = os.path.abspath(file_path)
        try:
            timeout = max(8.0, self._wait_timeout + 3.0)
            diagnostics = await asyncio.wait_for(
                self._snapshot_async(file_path), timeout=timeout
            )
            self._delta_baseline[absolute_path] = diagnostics or []
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.debug("baseline snapshot failed for %s: %s", file_path, exc)
            await self._mark_broken_for_file(file_path, exc)
            self._delta_baseline[absolute_path] = []

    async def get_diagnostics_sync(  # noqa: ASYNC109 - upstream API names timeout
        self,
        file_path: str,
        *,
        delta: bool = True,
        timeout: float | None = None,
        line_shift: Callable[[int], int | None] | None = None,
    ) -> list[dict[str, Any]]:
        """Open ``file_path``, wait for fresh diagnostics, and return them."""
        if not await self.enabled_for(file_path):
            return []
        await self._ensure_started()
        server = find_server_for_file(file_path)
        server_id = server.server_id if server else "?"
        try:
            wait_timeout = timeout if timeout is not None else self._wait_timeout + 2.0
            diagnostics = await asyncio.wait_for(
                self._open_and_wait_async(file_path), timeout=wait_timeout
            )
        except asyncio.CancelledError:
            raise
        except TimeoutError as exc:
            eventlog.log_timeout(server_id, file_path)
            logger.debug("LSP diagnostics timeout for %s: %s", file_path, exc)
            await self._mark_broken_for_file(file_path, exc)
            return []
        except Exception as exc:  # noqa: BLE001
            eventlog.log_server_error(server_id, file_path, exc)
            logger.debug("LSP diagnostics fetch failed for %s: %s", file_path, exc)
            await self._mark_broken_for_file(file_path, exc)
            return []

        if diagnostics is None:
            eventlog.log_timeout(server_id, file_path, kind="fresh diagnostics")
            return []

        absolute_path = os.path.abspath(file_path)
        if delta:
            baseline = self._delta_baseline.get(absolute_path) or []
            if baseline:
                if line_shift is not None:
                    from agent.lsp.range_shift import shift_baseline

                    baseline = shift_baseline(baseline, line_shift)
                seen = {_diag_key(item) for item in baseline}
                diagnostics = [
                    item for item in diagnostics if _diag_key(item) not in seen
                ]
            try:
                fresh = await asyncio.wait_for(
                    self._current_diags_async(file_path), timeout=2.0
                )
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                fresh = []
            if fresh:
                self._delta_baseline[absolute_path] = fresh

        if diagnostics:
            eventlog.log_diagnostics(server_id, file_path, len(diagnostics))
        else:
            eventlog.log_clean(server_id, file_path)
        return diagnostics

    async def _mark_broken_for_file(
        self, file_path: str, exc: BaseException
    ) -> None:
        server = find_server_for_file(file_path)
        if server is None:
            return
        workspace_root, gated = await resolve_workspace_for_file(file_path)
        if not (workspace_root and gated):
            return
        try:
            server_root = await server.resolve_root(file_path, workspace_root) or workspace_root
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            server_root = workspace_root
        key = (server.server_id, server_root)
        already_broken = key in self._broken
        self._broken.add(key)
        client = self._clients.pop(key, None)
        self._last_used.pop(key, None)
        if client is not None:
            try:
                await asyncio.wait_for(client.shutdown(), timeout=1.0)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                pass
        if not already_broken:
            eventlog.log_spawn_failed(server.server_id, server_root, exc)

    async def shutdown(self) -> None:
        """Tear down all clients and child tasks before returning."""
        if self._shutdown_task is None:
            self._shutdown_task = asyncio.create_task(
                self._shutdown_async(), name="hermes-lsp-shutdown"
            )
        cleanup_task = self._shutdown_task
        cancellation: asyncio.CancelledError | None = None
        while True:
            try:
                await asyncio.shield(cleanup_task)
                break
            except asyncio.CancelledError as exc:  # noqa: ASYNC103 - re-raised below
                if cleanup_task.cancelled():
                    raise
                if cancellation is None:
                    cancellation = exc
            except Exception as exc:
                if cancellation is not None:
                    raise cancellation from exc
                raise
        if cancellation is not None:
            raise cancellation

    async def _snapshot_async(self, file_path: str) -> list[dict[str, Any]]:
        client = await self._get_or_spawn(file_path)
        if client is None:
            return []
        try:
            version = await client.open_file(
                file_path, language_id=language_id_for(file_path)
            )
            fresh = await client.wait_for_diagnostics(
                file_path, version, mode=self._wait_mode
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.debug("snapshot open/wait failed: %s", exc)
            return []
        self._touch(client)
        if not fresh:
            return []
        return list(client.diagnostics_for(file_path, fresh_only=True))

    async def _open_and_wait_async(
        self, file_path: str
    ) -> list[dict[str, Any]] | None:
        client = await self._get_or_spawn(file_path)
        if client is None:
            return None
        try:
            version = await client.open_file(
                file_path, language_id=language_id_for(file_path)
            )
            await client.save_file(file_path)
            fresh = await client.wait_for_diagnostics(
                file_path,
                version,
                mode=self._wait_mode,
                timeout=self._wait_timeout,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.debug("open/wait failed for %s: %s", file_path, exc)
            return None
        self._touch(client)
        if not fresh:
            return None
        return list(client.diagnostics_for(file_path, fresh_only=True))

    async def _current_diags_async(self, file_path: str) -> list[dict[str, Any]]:
        workspace_root, gated = await resolve_workspace_for_file(file_path)
        server = find_server_for_file(file_path)
        if not (workspace_root and gated and server):
            return []
        client = self._clients.get((server.server_id, workspace_root))
        if client is None:
            return []
        return list(client.diagnostics_for(file_path, fresh_only=True))

    async def _get_or_spawn(self, file_path: str) -> LSPClient | None:
        server = find_server_for_file(file_path)
        if server is None:
            return None
        if server.server_id in self._disabled_servers:
            eventlog.log_disabled(server.server_id, file_path, "disabled in config")
            return None
        workspace_root, gated = await resolve_workspace_for_file(file_path)
        if not (workspace_root and gated):
            eventlog.log_no_project_root(server.server_id, file_path)
            return None
        server_root = await server.resolve_root(file_path, workspace_root)
        if server_root is None:
            eventlog.log_disabled(
                server.server_id, file_path, "exclude marker hit (server gated off)"
            )
            return None

        key = (server.server_id, server_root)
        if key in self._broken:
            return None
        client = self._clients.get(key)
        if client is not None and client.is_running:
            self._last_used[key] = time.time()
            eventlog.log_active(server.server_id, server_root)
            return client
        spawning = self._spawning.get(key)
        if spawning is not None:
            try:
                return await asyncio.shield(spawning)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                return None

        spawn_future = asyncio.get_running_loop().create_future()
        self._spawning[key] = spawn_future
        spawn_task = asyncio.create_task(
            self._spawn_client(key, server, server_root, spawn_future),
            name=f"hermes-lsp-spawn-{server.server_id}",
        )
        self._spawn_tasks.add(spawn_task)
        spawn_task.add_done_callback(self._spawn_tasks.discard)
        try:
            return await asyncio.shield(spawn_future)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            return None

    async def _spawn_client(
        self,
        key: tuple[str, str],
        server: ServerDef,
        server_root: str,
        spawn_future: asyncio.Future[LSPClient | None],
    ) -> None:
        client: LSPClient | None = None
        try:
            context = ServerContext(
                workspace_root=server_root,
                install_strategy=self._install_strategy,
                binary_overrides=self._binary_overrides,
                env_overrides=self._env_overrides,
                init_overrides=self._init_overrides,
            )
            spec = await server.build_spawn(server_root, context)
            if spec is None:
                eventlog.log_server_unavailable(server.server_id, server.server_id)
                self._broken.add(key)
                return
            client = LSPClient(
                server_id=server.server_id,
                workspace_root=spec.workspace_root,
                command=spec.command,
                env=spec.env,
                cwd=spec.cwd,
                initialization_options=spec.initialization_options,
                seed_diagnostics_on_first_push=(
                    spec.seed_diagnostics_on_first_push or server.seed_first_push
                ),
            )
            await client.start()
            if self._closed:
                await client.shutdown()
                return
            self._clients[key] = client
            self._last_used[key] = time.time()
            eventlog.log_active(server.server_id, server_root)
            if not spawn_future.done():
                spawn_future.set_result(client)
        except asyncio.CancelledError:
            if client is not None:
                await client.shutdown()
            raise
        except Exception as exc:  # noqa: BLE001
            eventlog.log_spawn_failed(server.server_id, server_root, exc)
            self._broken.add(key)
        finally:
            if not spawn_future.done():
                spawn_future.set_result(None)
            if self._spawning.get(key) is spawn_future:
                self._spawning.pop(key, None)

    def _touch(self, client: LSPClient) -> None:
        key = (client.server_id, client.workspace_root)
        if key in self._clients:
            self._last_used[key] = time.time()

    async def _start_idle_reaper(self) -> None:
        if self._idle_timeout > 0 and self._idle_reaper_task is None:
            self._idle_reaper_task = asyncio.create_task(
                self._idle_reaper_loop(),
                name="hermes-lsp-idle-reaper",
            )

    async def _idle_reaper_loop(self) -> None:
        interval = min(60.0, self._idle_timeout)
        while True:
            await asyncio.sleep(interval)
            try:
                await self._reap_idle_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.debug("LSP idle reaper sweep error: %s", exc)

    async def _reap_idle_once(self) -> None:
        cutoff = time.time() - self._idle_timeout
        idle_keys = [
            key for key in self._clients if self._last_used.get(key, 0) < cutoff
        ]
        clients = [self._clients.pop(key) for key in idle_keys]
        for key in idle_keys:
            self._last_used.pop(key, None)
        if clients:
            eventlog.log_reaped(
                [(client.server_id, client.workspace_root) for client in clients],
                self._idle_timeout,
            )
            await asyncio.gather(
                *(client.shutdown() for client in clients), return_exceptions=True
            )

    async def _shutdown_async(self) -> None:
        if self._closed:
            return
        self._closed = True
        reaper = self._idle_reaper_task
        self._idle_reaper_task = None
        if reaper is not None:
            reaper.cancel()
            await asyncio.gather(reaper, return_exceptions=True)
        current_task = asyncio.current_task()
        spawn_tasks = [
            task
            for task in self._spawn_tasks
            if task is not current_task and not task.done()
        ]
        for task in spawn_tasks:
            task.cancel()
        if spawn_tasks:
            await asyncio.gather(*spawn_tasks, return_exceptions=True)
        self._spawn_tasks.clear()
        clients = list(self._clients.values())
        self._clients.clear()
        self._broken.clear()
        self._last_used.clear()
        spawning = list(self._spawning.values())
        for future in spawning:
            if not future.done():
                future.set_result(None)
        self._spawning.clear()
        await asyncio.gather(
            *(client.shutdown() for client in clients), return_exceptions=True
        )
        clear_cache()

    def get_status(self) -> dict[str, Any]:
        """Return an in-memory status snapshot."""
        return {
            "enabled": self._enabled,
            "wait_mode": self._wait_mode,
            "wait_timeout": self._wait_timeout,
            "install_strategy": self._install_strategy,
            "clients": [
                {
                    "server_id": key[0],
                    "workspace_root": key[1],
                    "state": client.state,
                    "running": client.is_running,
                }
                for key, client in self._clients.items()
            ],
            "broken": list(self._broken),
            "disabled_servers": sorted(self._disabled_servers),
        }


def _diag_key(d: dict[str, Any]) -> str:
    """Content equality key used for cross-edit delta filtering."""
    range_value = d.get("range") or {}
    start = range_value.get("start") or {}
    end = range_value.get("end") or {}
    code = d.get("code")
    if code is not None and not isinstance(code, str):
        code = str(code)
    return "\x00".join(
        [
            str(d.get("severity") or 1),
            str(code or ""),
            str(d.get("source") or ""),
            str(d.get("message") or "").strip(),
            f"{start.get('line', 0)}:{start.get('character', 0)}-"
            f"{end.get('line', 0)}:{end.get('character', 0)}",
        ]
    )


__all__ = ["LSPService"]
