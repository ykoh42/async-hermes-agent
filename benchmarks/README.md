# Async OpenRouter benchmark

This opt-in benchmark exercises real `AIAgent.run_conversation()` turns with
one agent and one temporary SQLite session per request. It measures latency,
throughput, event-loop lag, process RSS, open file handles, and pending task
deltas while increasing inter-agent concurrency.

The default run is deliberately small: concurrency `1,2,4`, one request per
worker, and at most seven requests. Only `:free` OpenRouter models are accepted
unless `--allow-paid-model` is supplied. A stage with no successful responses
stops the run before the next stage.

```bash
.venv/bin/python benchmarks/openrouter.py \
  --live \
  --env-file ~/Desktop/collie/.env \
  --model openai/gpt-oss-20b:free \
  --output /tmp/async-hermes-openrouter-benchmark.json
```

Free endpoints can queue, disappear, or enforce account-wide limits without
notice. Treat a timeout or provider error as an external-capacity observation,
not automatically as an async runtime regression. The strongest local runtime
signals are event-loop lag, non-zero pending-task deltas, and resource deltas
that continue to grow across repeated runs. The first stage includes cold
client, schema, and logging initialization, so use later-stage deltas to judge
steady-state leakage.

Offline tests cover scheduler overlap, request-budget enforcement, timeouts,
and task cleanup without using credentials or network access:

```bash
.venv/bin/python -m pytest tests/benchmarks/test_openrouter.py -q
```
