## What does this change?

<!-- Describe the retained behavior or bug and why this approach is appropriate. -->

## Upstream relationship

<!-- If this ports upstream work, link the release/commit/diff and explain how its intent was preserved. Otherwise write N/A. -->

## Verification

<!-- List the exact commands and any live-provider checks you ran. -->

- [ ] `uv run pytest -q`
- [ ] `uv run ruff check .`
- [ ] `uv build`
- [ ] Additional focused checks:

## Checklist

- [ ] I read the [contributing guide](https://github.com/ykoh42/async-hermes-agent/blob/main/CONTRIBUTING.md).
- [ ] Public module paths, names, arguments, and return shapes remain stable, or the change documents why they cannot.
- [ ] The retained runtime uses native async I/O without sync wrappers or thread fallbacks.
- [ ] Prompt caching, role alternation, tool-result order, and trajectory order remain intact.
- [ ] I added behavior-level tests for the change.
- [ ] I did not add a removed UI, messaging, scheduler, or service surface incidentally.
- [ ] I updated current documentation when behavior or workflow changed.
- [ ] I did not commit credentials, private data, or generated trajectories.

## Notes for reviewers

<!-- Call out cancellation, concurrency, resource-lifecycle, dependency, security, or migration risks. -->
