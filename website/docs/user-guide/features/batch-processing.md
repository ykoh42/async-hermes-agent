---
title: Batch Processing
description: Generate ordered tool-use trajectories with bounded async concurrency and resumable JSONL output.
sidebar_position: 7
---

# Batch Processing

`BatchRunner` runs a JSONL prompt dataset through the same agent loop used by
interactive library calls. It is a training-data and evaluation harness, not a
fine-tuning trainer.

## Input

Each non-empty JSONL row must contain `prompt`:

```jsonl
{"prompt":"Inspect this Python project and explain its async boundaries."}
{"prompt":"Use the file tools to repair the supplied test fixture."}
```

Malformed rows and rows without `prompt` are skipped. Keep each dataset item
self-contained: batch runs set `skip_context_files=True`, `skip_memory=True`,
and do not attach durable sessions.

## Run programmatically

There is no supported command-line entrypoint in this library distribution.
Construct `BatchRunner` and await its existing `run()` method:

```python
import asyncio
import os

from batch_runner import BatchRunner


async def main() -> None:
    runner = BatchRunner(
        dataset_file="prompts.jsonl",
        batch_size=8,
        run_name="tool-training",
        distribution="terminal_only",
        base_url=os.getenv("MODEL_BASE_URL") or None,
        api_key=os.getenv("MODEL_API_KEY") or None,
        model=os.environ["MODEL_ID"],
        num_workers=4,
        max_iterations=20,
        reasoning_config={"enabled": True, "effort": "low"},
    )
    await runner.run(resume=True)


asyncio.run(main())
```

`run()` returns `None`; its contract is the files written below.

## Concurrency model

Each batch processes its prompts sequentially, preserving shard order. Up to
`num_workers` batches overlap model and tool I/O through asyncio tasks and a
semaphore. `num_workers` itself does not create a batch worker pool. Checkpoint
and shard file operations use the package's current `aiofiles` layer and can
therefore use its executor-backed regular-file implementation.

Cancellation cancels and awaits sibling batch tasks. If cancellation arrives
after a shard-row append has started, that single append completes before
`CancelledError` propagates. Checkpoint publication uses atomic replacement to
avoid half-written resume state.

## Output

Files are written under `data/<run_name>/`:

```text
data/tool-training/
├── batch_0.jsonl
├── batch_1.jsonl
├── trajectories.jsonl
├── checkpoint.json
└── statistics.json
```

- `batch_*.jsonl` are append-only per-batch shards.
- `trajectories.jsonl` merges valid shard rows.
- `checkpoint.json` records completed prompts.
- `statistics.json` summarizes tool and reasoning coverage.

Resume scans existing batch files and matches successful samples by prompt
content before scheduling remaining work. Failed prompts can be retried.

## Trajectory contract

The `conversations` sequence uses ShareGPT-style `from`/`value` entries with
roles such as `system`, `human`, `gpt`, and `tool`. Reasoning, model-emitted tool
calls, ordered observations, and the final answer are preserved. Tool statistics
and error counts are normalized for dataset processing.

Samples with zero recorded assistant reasoning are discarded from trajectory
files and marked complete, so resume does not repeatedly regenerate them.
Invalid JSON and entries containing unknown/hallucinated tool names are excluded
when shards are merged.

For a single conversation, `AIAgent(save_trajectories=True)` appends successful
samples to `trajectory_samples.jsonl` in the current working directory and
failed samples to `failed_trajectories.jsonl`.
