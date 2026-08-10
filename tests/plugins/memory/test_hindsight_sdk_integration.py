"""Real hindsight-client transport integration over a local TCP server."""

from __future__ import annotations

import json

import aiofiles
import pytest
from pyleak import no_event_loop_blocking, no_task_leaks
from pyleak.eventloop import LeakAction

pytest.importorskip("hindsight_client")
web = pytest.importorskip("aiohttp.web")

from plugins.memory.hindsight import (  # noqa: E402
    HindsightMemoryProvider,
    _append_capability_cache,
)

pytestmark = pytest.mark.asyncio


async def test_real_sdk_retain_recall_reflect_and_close(tmp_path, monkeypatch):
    requests: list[tuple[str, str, dict]] = []

    async def version(request):  # noqa: ASYNC124
        return web.json_response({"version": "0.6.1"})

    async def retain(request):
        body = await request.json()
        requests.append((request.method, request.path, body))
        return web.json_response(
            {
                "success": True,
                "bank_id": request.match_info["bank_id"],
                "items_count": len(body["items"]),
                "async": body["async"],
                "operation_id": "op-real",
            }
        )

    async def recall(request):
        body = await request.json()
        requests.append((request.method, request.path, body))
        return web.json_response(
            {
                "results": [
                    {
                        "id": "memory-1",
                        "text": "The user chose Seoul.",
                        "type": "observation",
                    }
                ]
            }
        )

    async def reflect(request):
        body = await request.json()
        requests.append((request.method, request.path, body))
        return web.json_response({"text": "The user prefers Seoul."})

    async def operation(request):  # noqa: ASYNC124
        return web.json_response(
            {
                "operation_id": request.match_info["operation_id"],
                "status": "completed",
            }
        )

    app = web.Application()
    app.router.add_get("/version", version)
    app.router.add_post("/v1/default/banks/{bank_id}/memories", retain)
    app.router.add_post(
        "/v1/default/banks/{bank_id}/memories/recall",
        recall,
    )
    app.router.add_post("/v1/default/banks/{bank_id}/reflect", reflect)
    app.router.add_get(
        "/v1/default/banks/{bank_id}/operations/{operation_id}",
        operation,
    )
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    socket = site._server.sockets[0]
    base_url = f"http://127.0.0.1:{socket.getsockname()[1]}"

    config_path = tmp_path / "hindsight" / "config.json"
    await aiofiles.os.makedirs(config_path.parent, exist_ok=True)
    async with aiofiles.open(config_path, "w", encoding="utf-8") as handle:
        await handle.write(
            json.dumps(
                {
                    "mode": "local_external",
                    "api_url": base_url,
                    "bank_id": "integration-bank",
                    "retain_async": True,
                    "prefetch_retain_drain_timeout": 2,
                }
            )
        )
    monkeypatch.setattr(
        "plugins.memory.hindsight.get_hermes_home",
        lambda: tmp_path,
    )
    _append_capability_cache.clear()

    async with (
        no_event_loop_blocking(action=LeakAction.RAISE, threshold=0.1),
        no_task_leaks(action=LeakAction.RAISE),
    ):
        provider = HindsightMemoryProvider()
        try:
            await provider.initialize(
                "integration-session",
                hermes_home=str(tmp_path),
                platform="service",
            )
            await provider.sync_turn(
                "I chose Seoul.",
                "I will remember that.",
            )
            await provider._retain_queue.join()
            recalled = json.loads(
                await provider.handle_tool_call(
                    "hindsight_recall",
                    {"query": "Which city?"},
                )
            )
            reflected = json.loads(
                await provider.handle_tool_call(
                    "hindsight_reflect",
                    {"query": "Summarize my preference."},
                )
            )

            assert recalled == {"result": "1. The user chose Seoul."}
            assert reflected == {"result": "The user prefers Seoul."}
            retain_body = next(
                body
                for method, path, body in requests
                if path.endswith("/memories")
            )
            assert retain_body["async"] is True
            assert retain_body["items"][0]["document_id"] == "integration-session"
            assert retain_body["items"][0]["update_mode"] == "append"
            assert retain_body["items"][0]["tags"] == [
                "session:integration-session"
            ]
        finally:  # noqa: ASYNC102 - test cleanup must always close both owners
            await provider.shutdown()
            await runner.cleanup()
            _append_capability_cache.clear()

        assert provider._client is None
