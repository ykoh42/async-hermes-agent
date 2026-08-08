---
sidebar_position: 5
title: "Learning Path"
description: "Choose a documentation path for embedding, serving, or generating training trajectories with Async Hermes Agent."
---

# Learning Path

Start with [Installation](./installation.md), then complete the
[Quickstart](./quickstart.md). Everything below assumes one provider and one
tool-capable model already complete a normal awaited turn.

## By goal

| Goal | Recommended reading |
| --- | --- |
| Embed the agent in Python | [Python Library](../guides/python-library.md) → [Configuration](../user-guide/configuration.md) → [Sessions](../user-guide/sessions.md) |
| Expose it from an async service | [Programmatic Integration](../developer-guide/programmatic-integration.md) → [Security](../user-guide/security.md) → [Session Storage](../developer-guide/session-storage.md) |
| Generate fine-tuning trajectories | [Batch Processing](../user-guide/features/batch-processing.md) → [Trajectory Format](../developer-guide/trajectory-format.md) → [Agent Loop](../developer-guide/agent-loop.md) |
| Add model-facing capabilities | [Tools](../user-guide/features/tools.md) → [Skills](../user-guide/features/skills.md) → [MCP](../user-guide/features/mcp.md) |
| Understand or extend the harness | [Architecture](../developer-guide/architecture.md) → [Agent Loop](../developer-guide/agent-loop.md) → [Toolsets Reference](../reference/toolsets-reference.md) |

## Recommended order for training work

1. Verify one provider can return reasoning and valid tool calls.
2. Inspect the exact tool schemas and enabled toolsets.
3. Run a single trajectory with `save_trajectories=True`.
4. Confirm reasoning → tool call → observation → final answer ordering.
5. Run a small resumable `BatchRunner` job.
6. Add dataset quality gates and model training outside this package.

The harness generates trajectories; it does not decide whether a sample is
correct, safe, diverse, or suitable for a particular fine-tuning objective.

## Recommended order for service work

1. Use one `AIAgent` per ordered conversation.
2. Map application identity to a deliberate session-isolation policy.
3. Register callbacks for interactive tools or disable those tools.
4. Bound concurrent agents and external tool work.
5. Propagate cancellation and always `await agent.close()`.
6. Add authentication, quotas, and HTTP schemas in the host application.

You do not need to read every page. Follow the path matching the boundary your
application owns, then return to the feature and reference sections as needed.
