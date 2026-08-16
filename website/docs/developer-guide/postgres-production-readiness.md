---
sidebar_position: 5
title: "PostgreSQL Production Readiness"
description: "Evidence and limits for running the PostgreSQL SessionDB with multiple workers"
---

# PostgreSQL Production Readiness

This is a validation report, not a throughput or availability guarantee. The
PostgreSQL backend is intended to be shared by agents in one application
worker. The application owns the worker lifespan and creates one
`hermes_state_postgres.SessionDB` per worker; an individual `AIAgent` borrows
that store and must not close it.

## Reproducible evidence

The integration suite uses a real PostgreSQL service and a credential-free
loopback process boundary. It does not make paid model requests. The CI
matrix runs the same PostgreSQL suite against PostgreSQL 15, 16, 17, and 18
with Python 3.11, 3.12, and 3.13 combinations.

`tests/integration/test_postgres_production_readiness.py` currently verifies:

| Scenario | Evidence collected |
| --- | --- |
| Four OS workers append to one session | 80 committed rows, unique contents, per-worker sequence order, distinct child PIDs, and child wall times |
| Worker termination and cold resume | An interrupted child is reaped; surviving rows remain unique; a new process resumes the same session and adds one marker |
| Backend termination | `pg_terminate_backend()` closes an idle pooled connection and `pool_pre_ping` reconnects the next checkout |
| Pool cleanup | SQLAlchemy pool `checked_out` is zero after each scenario |
| Agent and compaction path | `test_postgres_compaction_e2e.py` uses a real `AIAgent`, loopback streaming provider, PostgreSQL persistence, compaction, and resume |
| Pool exhaustion, cancellation, and schema recovery | The focused `test_postgres_session_db.py` suite exercises bounded timeout, rollback, connection return, migration repair, and read-only behavior |

Run the readiness checks locally with a PostgreSQL DSN:

```bash
HERMES_POSTGRES_TEST_DSN='postgresql+psycopg://user:password@127.0.0.1:5432/hermes' \
  uv run --extra dev --extra postgres pytest -q \
  tests/integration/test_postgres_production_readiness.py \
  tests/integration/test_postgres_compaction_e2e.py
```

Each test uses unique session IDs and fresh child runtime directories. The
children receive only the DSN and the paths required for the test; credentials
are never printed. The parent reaps every child, closes every SQLAlchemy
connection, and checks the pool before returning.

## Operational limits

The possible PostgreSQL connection count is approximately:

```text
worker_count * (pool_size + max_overflow)
```

Choose `pool_size`, `max_overflow`, and `pool_timeout` from the database's
connection budget, not from the number of concurrent HTTP requests. A bounded
pool timeout is preferable to unbounded queueing. Monitor pool checkout wait,
connection errors, transaction rollbacks, statement timeouts, and database
connection saturation in the host application.

PostgreSQL transactions provide cross-process ordering and rollback. They do
not make an application-level request idempotent: if a process dies after a
successful commit but before sending its response, the caller must use its own
request or operation key when retrying. The readiness harness checks duplicate
free test markers, not arbitrary application retries.

The backend reconnect test covers an invalidated connection and a terminated
backend with `pool_pre_ping`. It does not certify a managed failover service,
Aurora/RDS endpoint behavior, DNS convergence, replication lag, or a specific
cloud provider. Validate those properties against the selected provider and
its connection endpoint before production deployment.

SessionDB owns durable session and message rows. Trajectory files, memory
plugin databases, delegated-work records, caches, authentication, HTTP
routing, and FastAPI worker lifecycle remain separate application concerns.
Use a shared durable filesystem or object store for any artifact that must be
visible across replicas; PostgreSQL alone does not synchronize those files.

No fixed requests-per-second or user-count claim is made here. Capacity must
be measured with the target model, schema size, pool configuration, query mix,
and database hardware. A loopback provider is deliberately used for the
database gate so model latency does not hide database contention.
