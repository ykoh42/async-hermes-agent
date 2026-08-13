---
title: Work with Skills
description: Create, discover, and load reusable skill documents in the async agent harness.
sidebar_position: 2
---

# Work with Skills

A skill is a directory whose `SKILL.md` gives the model reusable instructions.
Skills are loaded on demand through tools, so a large skill library does not
need to be copied into every model request.

## Create a skill

Create a directory under `$HERMES_HOME/skills`; `HERMES_HOME` defaults to
`~/.hermes`.

```text
~/.hermes/skills/
└── review-python-change/
    ├── SKILL.md
    └── references/
        └── checklist.md
```

Use YAML frontmatter with a concise activation description:

```markdown
---
name: review-python-change
description: Review a Python change for correctness, async safety, and tests.
---

# Review a Python change

1. Read the changed code and its callers.
2. Check coroutine and cancellation behavior.
3. Run the focused tests before reporting findings.
```

The directory name and frontmatter name should be stable, lowercase identifiers.
Put large reference material under `references/`, templates under `templates/`,
scripts under `scripts/`, and other assets under `assets/`. `skill_view` can
load those linked files when needed.

## Add shared skill directories

Point the library at existing repositories without copying them:

```yaml
skills:
  external_dirs:
    - ~/.agents/skills
    - /srv/team-skills
```

Relative entries resolve from `HERMES_HOME`; `~` and environment variables are
expanded. Missing and duplicate directories are ignored. The local skills
directory takes precedence during discovery.

External directories are externally owned. Discovery and reading are safe by
default, while an explicit foreground `skill_manage` call may edit an existing
external skill in place. Use filesystem permissions if those repositories must
remain immutable.

## Enable skill tools

```python
async with AIAgent(..., enabled_toolsets=["skills", "file"]) as agent:
    result = await agent.run_conversation(
        "Find and follow the review-python-change skill for this patch."
    )
```

The `skills` toolset contains:

| Tool | Purpose |
| --- | --- |
| `skills_list` | Discover local and configured external skills |
| `skill_view` | Read a complete `SKILL.md` or one of its supporting files |
| `skill_manage` | Create, patch, edit, delete, or manage supporting files |

Skill reads and writes expose awaited coroutine APIs. They currently use the
package's executor-backed `aiofiles` regular-file layer, so they must not be
described as OS-native file I/O. When trajectory saving is enabled,
list/view/manage calls and their observations retain their normal place in the
trajectory.

## Distribution note

Source checkouts can contain example skills, but the installed library should
not be treated as a comprehensive bundled skill catalog or installer. Deploy the
skills your application intends to expose and version them separately when
reproducibility matters.

Skills are trusted procedural instructions with access to the agent's enabled
tools. Review third-party skills before activation. See
[Skills concepts](../user-guide/features/skills.md) and
[Security](../user-guide/security.md).
