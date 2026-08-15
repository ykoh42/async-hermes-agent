"""Regression tests for parallel image-generation tool batches."""

import json
from types import SimpleNamespace

from agent import tool_executor
from agent.tool_dispatch_helpers import _plan_tool_batch_segments


def _tool_call(name: str, args: dict, call_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(args),
        ),
    )


def test_image_generate_batch_routes_to_concurrent_executor():
    assistant_message = SimpleNamespace(
        tool_calls=[
            _tool_call("image_generate", {"prompt": "variation one"}, "img_1"),
            _tool_call("image_generate", {"prompt": "variation two"}, "img_2"),
        ],
    )

    segments = _plan_tool_batch_segments(assistant_message.tool_calls)
    assert segments[0][0] == "parallel"
    assert len(segments[0][1]) == 2


def test_image_generate_parallel_worker_cap_defaults_to_four():
    runnable_calls = [
        (
            0,
            _tool_call("image_generate", {"prompt": "one"}, "img_1"),
            "image_generate",
            {},
        ),
        (
            1,
            _tool_call("image_generate", {"prompt": "two"}, "img_2"),
            "image_generate",
            {},
        ),
        (
            2,
            _tool_call("image_generate", {"prompt": "three"}, "img_3"),
            "image_generate",
            {},
        ),
        (
            3,
            _tool_call("image_generate", {"prompt": "four"}, "img_4"),
            "image_generate",
            {},
        ),
        (
            4,
            _tool_call("image_generate", {"prompt": "five"}, "img_5"),
            "image_generate",
            {},
        ),
    ]

    assert tool_executor._max_workers_for_tool_batch(runnable_calls) == 4


def test_image_generate_parallel_worker_cap_can_be_configured_lower():
    runnable_calls = [
        (
            0,
            _tool_call("image_generate", {"prompt": "one"}, "img_1"),
            "image_generate",
            {},
        ),
        (
            1,
            _tool_call("image_generate", {"prompt": "two"}, "img_2"),
            "image_generate",
            {},
        ),
    ]

    assert tool_executor._max_workers_for_tool_batch(
        runnable_calls,
        {"image_gen": {"max_parallel_requests": 1}},
    ) == 1


def test_image_generate_parallel_worker_cap_accepts_five_tuple_scope_metadata():
    """The segmented native executor appends a scope-block field per call."""
    runnable_calls = [
        (
            0,
            _tool_call("image_generate", {"prompt": "one"}, "img_1"),
            "image_generate",
            {},
            None,
        ),
        (
            1,
            _tool_call("image_generate", {"prompt": "two"}, "img_2"),
            "image_generate",
            {},
            {"path": "/tmp/work"},
        ),
    ]

    assert tool_executor._max_workers_for_tool_batch(runnable_calls) == 2
