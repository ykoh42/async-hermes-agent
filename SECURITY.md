# Async Hermes Agent Security Policy

This document covers the native-async library in this repository. The upstream
Hermes Agent product has a different surface and security policy.

## Reporting a vulnerability

Report vulnerabilities privately through
[GitHub Security Advisories](https://github.com/ykoh42/async-hermes-agent/security/advisories/new).
Do not include credentials, private trajectories, or working exploit details in
a public issue.

Include the affected commit or release, Python version, operating system,
configuration relevant to the issue, minimal reproduction, and the security
boundary that is crossed. This project does not operate a bug bounty program.

## Supported versions

Security fixes target the latest release and `main`. Confirm a report against
one of those before submitting it.

## Retained security surface

This package is a library. It retains:

- model-provider network clients;
- terminal, file, code-execution, browser, memory, and related tools;
- MCP client connections and MCP-launched subprocesses;
- plugin and provider discovery;
- skill discovery and loading;
- SQLite sessions, memory files, checkpoints, and trajectory output; and
- concurrent single-agent and batch execution.

It does not provide an HTTP server, authentication layer, messaging gateway,
CLI/TUI, scheduler, dashboard, or desktop application. Applications embedding
the library are responsible for caller authentication, authorization, request
limits, tenant isolation, transport security, and safe exposure of agent
output.

## Trust model

### The model is untrusted

Model output and content returned by web pages, files, tools, skills, plugins,
and MCP servers may be adversarial. Prompt instructions, allowlists, approval
prompts, and string scanners reduce accidents; they are not containment.

### The operating system is the isolation boundary

The default local terminal backend executes with the permissions of the host
process. Selecting a container or remote terminal backend confines operations
that go through that backend, but it does not automatically confine Python
plugins, MCP clients, provider clients, or other in-process code.

For hostile inputs or shared/production services, run the entire embedding
application and its child processes inside an OS-level sandbox with explicit
filesystem, process, and network policy. Expose only the directories and
credentials the agent needs.

### Plugins, skills, and MCP are trusted extensions

Plugins execute Python in the agent process and therefore receive its process
privileges. Review plugin code and dependencies before installation.

Skills can direct the model to execute commands or access external systems.
Review the full skill directory, including scripts and referenced assets, not
only `SKILL.md`.

MCP servers are separate trusted programs or services. They may receive
conversation data and return attacker-controlled content. Pin and review local
server packages, constrain their environment, and use authenticated encrypted
transport for remote servers.

### Credentials

Use `.env` only for secrets and `config.yaml` for behavior. Do not place secrets
in prompts, skills, trajectories, source control, or logs. Give each provider,
tool, plugin, and MCP server the narrowest credential scope possible.

Thread-based compatibility is intentionally not used to hide synchronous
implementations. This is an async correctness guarantee, not a security
boundary.

### Stored data

Sessions, memories, checkpoints, and trajectories can contain prompts,
reasoning, tool arguments, tool observations, file contents, and model output.
Treat the Hermes home and trajectory directories as sensitive data. Apply
appropriate filesystem permissions, retention, encryption, and deletion policy
in the embedding application.

## In-scope reports

Examples include:

- credential disclosure caused by this library;
- authorization or path validation defects that escape a selected backend's
  documented boundary;
- unsafe deserialization, command construction, or SQL construction;
- cross-session state disclosure caused by the library;
- MCP or plugin discovery that loads a different target than the operator
  selected;
- cancellation or concurrency defects that expose another conversation's
  state; and
- dependency vulnerabilities exploitable through a retained runtime path.

## Usually out of scope

The following are not library vulnerabilities by themselves:

- a model following malicious prompt instructions without crossing an OS or
  application authorization boundary;
- commands intentionally approved by the embedding application;
- a malicious plugin, skill, or MCP server explicitly installed by the
  operator;
- credentials or directories deliberately exposed to the agent process; and
- vulnerabilities that exist only in removed upstream surfaces.

These cases may still justify a normal hardening issue if they can be discussed
without publishing sensitive exploit details.

## Dependency and disclosure handling

Dependencies are pinned and reviewed deliberately. A report should identify
the reachable retained path, not only a package advisory. After a fix is
available, maintainers will publish an updated release and credit reporters who
want acknowledgement.
