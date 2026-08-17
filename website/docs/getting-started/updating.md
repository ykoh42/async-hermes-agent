---
sidebar_position: 4
title: "Updating & Uninstalling"
description: "Move between Async Hermes Agent releases while keeping application state and source changes explicit."
---

# Updating & Uninstalling

Async Hermes Agent does not include the upstream interactive updater. The host
application controls dependency upgrades, rollout, and rollback.

## Update a pinned installation

Replace `<version>` with the release you have reviewed:

```bash
uv pip install --upgrade "async-hermes-agent==<version>"
```

Pinning the release keeps the agent harness reproducible alongside your model,
prompt, and dataset versions. If you install from source, pin an immutable tag
or commit rather than tracking `main` implicitly.

### One-time migration from the legacy version scheme

GitHub releases through `0.20.4` used a fork-only version sequence. Starting
with `0.20.2.1`, the first three numeric segments match the upstream Python
package and the fourth is this distribution's revision. This release aligns the
fork with upstream `v2026.8.16`; replace an old exact pin and reinstall it
explicitly:

```bash
uv pip install --reinstall "async-hermes-agent==0.20.2.1"
```

Also update lockfiles, requirements manifests, and direct Git URLs from
`v0.20.4` to `v0.20.2.1`. Fork-only follow-ups increment the final segment,
while a later upstream port changes the first three segments and resets the
revision to `1`.

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
