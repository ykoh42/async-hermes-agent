---
sidebar_position: 6
title: "Trajectory Format"
description: "JSONL records produced by individual agent turns and the async batch runner"
---

# Trajectory Format

Async Hermes Agent can convert a completed turn into a text-oriented,
ShareGPT-style conversation for downstream dataset work. It preserves the
sequence needed to study interleaved reasoning and tool use:

1. system tool instructions;
2. human request;
3. assistant reasoning and tool call;
4. tool observation;
5. subsequent reasoning/tool rounds;
6. final assistant answer.

The library generates and serializes trajectories. It does not train, fine-
tune, score, or curate a model.

## Per-turn files

Construct an agent with `save_trajectories=True`:

```python
async with AIAgent(
    ...,
    save_trajectories=True,
) as agent:
    await agent.run_conversation("Research the issue")
```

Completed turns append one record to `trajectory_samples.jsonl` in the current
working directory. Incomplete or failed turns use `failed_trajectories.jsonl`.

```json
{
  "conversations": [],
  "timestamp": "2026-08-08T12:00:00",
  "model": "example/model",
  "completed": true
}
```

Each append is awaited and shielded long enough to finish one JSONL record if
the owning task is cancelled.

## Conversation entries

`conversations` is an ordered list of objects with `from` and `value` fields.

| `from` | Meaning |
| --- | --- |
| `system` | Function-calling instructions and the tool schemas active for the sample |
| `human` | User input, including later genuine user turns if present |
| `gpt` | Assistant reasoning, tool calls, narration, or final answer |
| `tool` | One or more observations corresponding to the prior assistant tool calls |

An abbreviated example:

```json
{
  "conversations": [
    {"from": "system", "value": "...<tools>...</tools>..."},
    {"from": "human", "value": "Inspect the repository"},
    {
      "from": "gpt",
      "value": "<think>\nI need the tree.\n</think>\n<tool_call>\n{\"name\":\"terminal\",\"arguments\":{\"command\":\"git status --short\"}}\n</tool_call>"
    },
    {
      "from": "tool",
      "value": "<tool_response>\n{\"tool_call_id\":\"call_1\",\"name\":\"terminal\",\"content\":\"\"}\n</tool_response>"
    },
    {"from": "gpt", "value": "<think>\n\n</think>\nThe tree is clean."}
  ]
}
```

## Normalization rules

### Reasoning

Native reasoning is placed inside `<think>...</think>`. Existing
`<REASONING_SCRATCHPAD>` blocks are normalized to the same tag. Every `gpt`
entry receives a `<think>` block, which may be empty, so consumers see a stable
shape.

### Tool calls

Each model call is serialized as JSON inside `<tool_call>` tags with `name` and
`arguments`. Arguments are objects rather than JSON-encoded strings.

### Tool observations

Consecutive observations for one assistant turn are collected into one `tool`
entry, each inside its own `<tool_response>` tag. The payload carries
`tool_call_id`, `name`, and `content`.

### Media

Trajectories are text-oriented. Image-bearing tool messages use their text
summary rather than embedding large base64 payloads.

### Ephemeral context

`BatchRunner` disables project context files and persistent memory for each
sample and does not save its optional ephemeral system prompt into the
trajectory. This prevents machine-local context from silently contaminating a
dataset.

## Batch records

`BatchRunner` writes `data/<run_name>/batch_<n>.jsonl` shards. Each successful
record includes:

```json
{
  "prompt_index": 0,
  "conversations": [],
  "metadata": {
    "batch_num": 0,
    "timestamp": "2026-08-08T12:00:00",
    "model": "example/model"
  },
  "completed": true,
  "partial": false,
  "api_calls": 3,
  "toolsets_used": ["file", "terminal"],
  "tool_stats": {},
  "tool_error_counts": {}
}
```

At the end of a run, valid shards are merged into
`data/<run_name>/trajectories.jsonl`. The same directory contains
`checkpoint.json` and `statistics.json`. Resume scans saved prompt content and
does not rely only on a process-local counter.

Samples with no reasoning across all assistant turns are discarded by the
batch path. Invalid JSON rows and entries containing unknown tool names are
filtered during merge and reported in the run output.

## Reading JSONL

```python
import json
from pathlib import Path


records = [
    json.loads(line)
    for line in Path("data/my-run/trajectories.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]
```

Before training, define and test your own acceptance policy for completion,
tool success, reasoning quality, safety, duplication, and data provenance.
Generation success is not evidence that a sample is suitable for training.

See [Batch Processing](/user-guide/features/batch-processing) for runner usage.
