---
sidebar_position: 3
title: "Platform Support"
description: "Supported Python versions, operating systems, and installation methods for Async Hermes Agent."
---

# Platform Support

Async Hermes Agent is a Python library rather than the complete Hermes desktop,
CLI, or messaging distribution. Platform support therefore follows the Python
runtime and the external tools enabled by your application.

## Supported Python versions

| Python | Status |
| --- | --- |
| 3.11 | Supported and used by CI |
| 3.12 | Supported |
| 3.13 | Supported |
| 3.10 and older | Unsupported |
| 3.14 and newer | Not yet supported by the package constraint |

## Operating systems

The core agent, provider, session, trajectory, skill, and MCP code is intended
for Linux, macOS, and Windows. A capability can impose additional requirements:

- terminal and process behavior follows the host shell and process model;
- browser tools require their external browser runtime;
- stdio MCP servers require their configured command to exist on `PATH`;
- optional cloud providers require their provider extra and credentials;
- filesystem paths and subprocess environments remain platform-specific.

Linux is the authoritative CI environment. Before production deployment on a
different platform, run the repository test suite and an end-to-end model →
tool → observation → final-answer turn on that target.

## Supported installation methods

The currently verified release path is the immutable GitHub tag. Pin the
package version in production:

```bash
uv pip install "git+https://github.com/ykoh42/async-hermes-agent.git@v0.20.4"
```

The package metadata for this release is `async-hermes-agent==0.20.4`.
Use the PyPI path only after `async-hermes-agent` is visible on PyPI and the
clean-install check has passed.

Installing a Git tag or source checkout is also supported. This site does not
claim a Homebrew, desktop-installer, container-image, or system-service
distribution. A host application may package the library in those forms, but
owns that additional support contract.

Continue with [Installation](./installation.md) and the
[Quickstart](./quickstart.md).
