---
title: Skills
description: Add reusable procedural knowledge without growing the core tool schema.
sidebar_position: 3
---

# Skills

Skills are Markdown instruction packages that the model discovers and reads on
demand. They are useful for repeatable workflows, project conventions, and
specialized tool procedures.

## Layout

```text
$HERMES_HOME/skills/<skill-name>/
├── SKILL.md
├── references/    # Optional detailed documentation
├── templates/     # Optional reusable templates
├── scripts/       # Optional helper scripts
└── assets/        # Optional assets
```

`SKILL.md` begins with YAML frontmatter:

```markdown
---
name: release-review
description: Verify a Python package before publishing a release.
---

# Release review

Run focused tests, inspect package contents, and report blockers before publish.
```

Use `skills.external_dirs` in `config.yaml` to discover shared skill roots:

```yaml
skills:
  external_dirs:
    - /srv/team-skills
```

## Tool surface

Enable `enabled_toolsets=["skills"]` to expose:

- `skills_list`, which scans local and external roots;
- `skill_view`, which returns the complete selected document or supporting
  file;
- `skill_manage`, which can create, patch, edit, delete, write, and remove
  skill files.

Skill content is not paginated. The model must receive the complete selected
instruction rather than reading only the first page. Repeated unchanged views
within one task can be deduplicated without changing the earlier content.

## Prompt-cache behavior

The list of available skill metadata can be incorporated into the stable
conversation context, while full instructions are loaded through a tool call.
Creating or editing a skill invalidates the skill snapshot for a later
conversation; it does not rewrite past messages in the current conversation.

## Packaging and trust

Do not assume the Python distribution installs a large upstream skill catalog.
Provision the exact skills needed by the application and version them alongside
the deployment when reproducible trajectories matter.

Skills influence a tool-capable model, so treat third-party skill text and
scripts as trusted code. Filesystem permissions remain the strongest way to
make a shared skill repository read-only.

For a practical walkthrough, see
[Work with Skills](../../guides/work-with-skills.md).
