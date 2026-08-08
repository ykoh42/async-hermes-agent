---
sidebar_position: 4
title: "Updating & Uninstalling"
description: "Move between Async Hermes Agent releases while keeping application state and source changes explicit."
---

# Updating & Uninstalling

Async Hermes Agent does not include the upstream interactive updater. The host
application controls dependency upgrades, rollout, and rollback.

## Update a PyPI installation

Replace `<version>` with the release you have reviewed:

```bash
uv pip install --upgrade "async-hermes-agent==<version>"
```

Pinning the version keeps the agent harness reproducible alongside your model,
prompt, and dataset versions.

## Update a tagged Git installation

Replace `<release>` with the tag you have reviewed:

```bash
uv pip install --upgrade \
  "git+https://github.com/ykoh42/async-hermes-agent.git@<release>"
```

Do not track `main` implicitly in production. Pin a tag or commit.

## Update a source checkout

```bash
git fetch origin --tags
git status --short
git diff --stat HEAD..origin/main
git pull --ff-only origin main
uv sync --locked
```

Inspect local changes before updating. This fork keeps upstream file locations
to make migrations reviewable, but native-async changes can still conflict with
a newer Hermes Agent release.

## Validate after updating

Run all of the following before rollout:

1. `uv lock --check`
2. `uv run ruff check .`
3. the repository's native-async audit from `.github/workflows/ci.yml`
4. `scripts/run_tests.sh -j 4 -- -q`
5. one real provider turn that exercises a tool and persists its trajectory
6. your application's cancellation, session-resume, and shutdown tests

## Roll back

Reinstall the previous tag or deploy the previous lockfile. Session, memory,
skill, and trajectory data live under `HERMES_HOME`; back that directory up
according to your application's retention policy before a schema-sensitive
upgrade.

## Uninstall

```bash
uv pip uninstall async-hermes-agent
```

Uninstalling the package does not remove `HERMES_HOME`. Delete application
state only when you have explicitly decided that sessions, memories, skills,
credentials, and trajectories are no longer needed.
