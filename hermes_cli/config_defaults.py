"""Default configuration data for Hermes Agent.

Pure-data leaf module: DEFAULT_CONFIG and OPTIONAL_ENV_VARS, extracted
verbatim from hermes_cli/config.py. Must not import from hermes_cli.config.
"""

DEFAULT_CONFIG = {
    "model": "",
    "providers": {},
    "fallback_providers": [],
    "credential_pool_strategies": {},
    "toolsets": ["hermes-cli"],
    # SQLite journal mode used by every Hermes database opener. WAL is the
    # normal default; set DELETE for weak-fsync/shared filesystems where WAL is
    # not crash-safe (for example macOS virtiofs, NFS, or SMB).
    "database": {
        "journal_mode": "wal",
        # Optional WAL sizing pragmas, applied when set to integers.
        # None = SQLite defaults (autocheckpoint 1000 pages, no size limit).
        "wal_autocheckpoint": None,
        "journal_size_limit": None,
    },
    "agent": {
        "max_turns": 500,
        # Max app-level retry attempts for API errors (connection drops,
        # provider timeouts, 5xx, etc.) before the agent surfaces the
        # failure.  The OpenAI SDK already does its own low-level retries
        # (max_retries=2 default) for transient network errors; this is
        # the Hermes-level retry loop that wraps the whole call.  Lower
        # this to 1 if you use fallback providers and want fast failover
        # on flaky primaries; raise it if you prefer to tolerate longer
        # provider hiccups on a single provider.
        "api_max_retries": 3,
        "service_tier": "",
        # Tool-use enforcement: injects system prompt guidance that tells the
        # model to actually call tools instead of describing intended actions.
        # Values: "auto" (default — applies to gpt/codex models), true/false
        # (force on/off for all models), or a list of model-name substrings
        # to match (e.g. ["gpt", "codex", "gemini", "qwen"]).
        "tool_use_enforcement": "auto",
        # Intent-ack continuation: when the model opens a turn by narrating an
        # action it will take ("I'll go check the logs...") but emits no tool
        # call, intercept the turn-end, inject a "continue now, execute the
        # tools" nudge, and loop instead of ending the turn (capped at 2 nudges
        # per turn). This is the corrective sibling of tool_use_enforcement (the
        # preventive prompt-side guard). Values: "auto" (default — fires only on
        # the codex_responses api_mode, the historical behavior), true (all
        # api_modes — fixes the Gemini/Claude "stops after stating intent" case),
        # false (never), or a list of model-name substrings to match.
        "intent_ack_continuation": "auto",
        # Universal "finish the job" guidance — short prompt block applied to
        # all models that targets two cross-family failure modes: (1) stopping
        # after a stub instead of finishing the artifact, (2) fabricating
        # plausible-looking output when a real path is blocked.  Costs ~80
        # tokens in the cached system prompt.  Set False to disable globally.
        "task_completion_guidance": True,
        # Universal parallel-tool-call guidance — short prompt block applied to
        # all models that tells the model to batch independent tool calls
        # (reads, searches, web fetches, read-only commands) into one turn
        # instead of one call per turn.  The runtime already runs independent
        # calls concurrently, so this just steers the model to produce the
        # batch — cutting round-trips and the resent-context cost that
        # compounds over a long conversation.  Costs ~70 tokens in the cached
        # system prompt.  Set False to disable globally.
        "parallel_tool_call_guidance": True,
        "environment_probe": True,
        # Local-environment toolchain probe — surfaces Python/pip/uv/PEP-668
        # state in the system prompt when something non-default is detected
        # (e.g. python3 has no pip module, pip→python version mismatch, PEP
        # 668 enforcement without uv).  Costs zero tokens when the env is
        # clean (probe emits nothing).  Skipped for remote terminal backends
        # (docker/modal/ssh — they have their own probe).  Set False to
        # disable entirely.
        # Embedder-supplied environment description appended to the system
        # prompt's environment-hints block. Lets a host that wraps Hermes
        # (sandbox runner, managed platform) explain the runtime environment
        # — proxy, credential handling, mount layout — without editing the
        # identity slot (SOUL.md). Empty by default. The HERMES_ENVIRONMENT_HINT
        # env var overrides this (build-time/container mechanism).
        "environment_hint": "",
        # Coding posture — on interactive coding surfaces (CLI, TUI, desktop
        # app, ACP) in a code workspace, Hermes adds a coding operating brief
        # + a live git/workspace snapshot to the system prompt. See
        # agent/coding_context.py.
        #   "auto" (default) — prompt-only posture when the surface is
        #                      interactive AND cwd is a code workspace.
        #                      Toolsets are never touched; messaging platforms
        #                      unaffected.
        #   "focus"          — auto + collapse the toolset to the lean coding
        #                      set (+ enabled MCP servers) + demote non-coding
        #                      skill categories to names-only in the prompt's
        #                      skill index. Explicit opt-in.
        #   "on"             — force the prompt posture everywhere.
        #   "off"            — disable entirely.
        "coding_context": "auto",
        # Standing operator instructions for the coding posture. A string (or
        # list of strings) appended to the coding brief as an extra stable
        # system block — pin project-wide workflow rules here instead of editing
        # the shipped brief, e.g. "For UI work, don't run tsc/lint until I
        # approve. Clean the diff before you commit and push." Cache-safe:
        # takes effect next session. Empty by default.
        "coding_instructions": "",
        # Verification closure for code edits. Programmatic callers default to
        # enabled; set false to disable the bounded evidence follow-up.
        "verify_on_stop": "auto",
        "verify_guidance": True,
        "max_verify_nudges": 3,
        # How user-attached images are presented to the main model on each turn.
        #   "auto"   — attach natively when the active model reports
        #              supports_vision=True AND the user hasn't explicitly
        #              configured auxiliary.vision.provider.  Otherwise fall
        #              back to text (vision_analyze pre-analysis).
        #   "native" — always attach natively; non-vision models will either
        #              error at the provider or get a last-chance text fallback
        #              (see run_agent._prepare_messages_for_api).
        #   "text"   — always pre-analyze with vision_analyze and prepend the
        #              description as text; the main model never sees pixels.
        # Affects gateway platforms, the TUI, and CLI /attach.  vision_analyze
        # remains available as a tool regardless of this setting — the routing
        # only controls how inbound user images are presented.
        "image_input_mode": "auto",
        "disabled_toolsets": [],

        # Per-model reasoning effort overrides (spelling-tolerant).
        # Dict mapping model names (any reasonable spelling) to effort levels.
        # Takes precedence over agent.reasoning_effort when the current model
        # matches a key in this dict.
        # Edit directly in config.yaml (no CLI support due to dots in keys).
        "reasoning_overrides": {},
    },

    "terminal": {
        "backend": "local",
        "modal_mode": "auto",
        "cwd": ".",  # Use current directory
        "timeout": 180,
        "daemon_term_grace_seconds": 2.0,
        # Environment variables to pass through to sandboxed execution.
        # Skill-declared required_environment_variables are included
        # automatically; this list is for non-skill use cases.
        "env_passthrough": [],
        "home_mode": "auto",
        "shell_init_files": [],
        "auto_source_bashrc": True,
        "docker_image": "nikolaik/python-nodejs:python3.11-nodejs20",
        "docker_forward_env": [],
        "docker_env": {},
        "singularity_image": "docker://nikolaik/python-nodejs:python3.11-nodejs20",
        "modal_image": "nikolaik/python-nodejs:python3.11-nodejs20",
        "daytona_image": "nikolaik/python-nodejs:python3.11-nodejs20",
        "vercel_runtime": "node24",
        "container_cpu": 1,
        "container_memory": 5120,
        "container_disk": 51200,
        "container_persistent": True,
        "docker_volumes": [],
        "docker_mount_cwd_to_workspace": False,
        "docker_network": True,
        "docker_extra_args": [],
        "docker_shm_size": "1g",
        "docker_run_as_host_user": False,
        "persistent_shell": True,
    },

    # Language Server Protocol — semantic diagnostics from real language
    # servers wired into the post-write checks used by write_file and patch.
    # The subsystem stays dormant outside a detected git worktree.
    "lsp": {
        "enabled": True,
        "wait_mode": "document",
        "wait_timeout": 5.0,
        # Missing servers may be installed into <HERMES_HOME>/lsp/bin.
        "install_strategy": "auto",
        # Set to 0 to keep spawned servers for the service lifetime.
        "idle_timeout": 600.0,
        # Per-server disabled/command/env/initialization_options overrides.
        "servers": {},
    },

    "web": {
        "backend": "",           # shared fallback — applies to both search and extract
        "search_backend": "",    # per-capability override for web_search (e.g. "searxng")
        "extract_backend": "",   # per-capability override for web_extract (e.g. "native")
        "extract_char_limit": 15000,  # per-page char budget for web_extract; larger pages truncate + store full text in cache/web
    },

    "browser": {
        "inactivity_timeout": 120,
        "command_timeout": 30,  # Timeout for browser commands in seconds (screenshot, navigate, etc.)
        "record_sessions": False,  # Auto-record browser sessions as WebM videos
        "headed": False,  # Local mode: launch Chromium with a visible window (also skips per-turn cleanup so the window persists between turns; idle reaper still applies)
        "allow_private_urls": False,  # Allow navigating to private/internal IPs (localhost, 192.168.x.x, etc.)
        # Browser engine for local mode.  Passed as ``--engine <value>`` to
        # agent-browser v0.25.3+.
        # "auto"       — use Chrome (default, don't pass --engine at all)
        # "lightpanda" — use Lightpanda (1.3-5.8x faster navigation, no screenshots)
        # "chrome"     — explicitly request Chrome
        # Also settable via AGENT_BROWSER_ENGINE env var.
        "engine": "auto",
        "auto_local_for_private_urls": True,  # When a cloud provider is set, auto-spawn local Chromium for LAN/localhost URLs instead of sending them to the cloud
        "cdp_url": "",  # Optional persistent CDP endpoint for attaching to an existing Chromium/Chrome
        "allow_unsafe_evaluate": False,  # Legacy override: when true, browser_console(expression=...) bypasses the restrict_evaluate denylist entirely
        "restrict_evaluate": False,  # Opt-in denylist blocking sensitive JS primitives (cookies/storage/clipboard/network/form values) in browser_console(expression=...)
        # CDP supervisor — dialog + frame detection via a persistent WebSocket.
        # Active only when a CDP-capable backend is attached (Browserbase or
        # local Chrome via /browser connect). See
        # website/docs/developer-guide/browser-supervisor.md.
        "dialog_policy": "must_respond",  # must_respond | auto_dismiss | auto_accept
        "dialog_timeout_s": 300,  # Safety auto-dismiss after N seconds under must_respond
        "camofox": {
            # When true, Hermes sends a stable profile-scoped userId to Camofox
            # so the server maps it to a persistent Firefox profile automatically.
            # When false (default), each session gets a random userId (ephemeral).
            "managed_persistence": False,
            # Optional externally managed Camofox identity. Useful when another
            # app owns the visible browser and Hermes should operate in it.
            "user_id": "",
            "session_key": "",
            # Rehydrate tab_id from Camofox before creating a new tab.
            "adopt_existing_tab": False,
            # Docker Camofox opens page URLs from inside the container. Enable
            # this to rewrite loopback page URLs (localhost/127.0.0.1/::1) to a
            # host alias while leaving CAMOFOX_URL itself unchanged.
            "rewrite_loopback_urls": False,
            "loopback_host_alias": "host.docker.internal",
        },
    },

    # Filesystem checkpoints — automatic snapshots before destructive file ops.
    # When enabled, the agent takes a snapshot of the working directory once
    # per conversation turn (on first write_file/patch call).  Use /rollback
    # to restore.
    #
    # Defaults changed in v2 (single shared shadow store, real pruning):
    #   - enabled: True -> False   (opt-in; most users never use /rollback)
    #   - max_snapshots: 50 -> 20  (now actually enforced via ref rewrite)
    #   - auto_prune:   False -> True (orphans/stale pruned automatically)
    # Opt in via ``hermes chat --checkpoints`` or set enabled=True here.
    "checkpoints": {
        "enabled": False,
        # Max checkpoints to keep per working directory.  Pre-v2 this only
        # limited the `/rollback` listing; v2 actually rewrites the ref and
        # garbage-collects older commits.
        "max_snapshots": 20,
        # Hard ceiling on total ``~/.hermes/checkpoints/`` size (MB).  When
        # exceeded, the oldest checkpoint per project is dropped in a
        # round-robin pass until total size falls under the cap.
        # 0 disables the size cap.
        "max_total_size_mb": 500,
        # Skip any single file larger than this when staging a checkpoint.
        # Prevents accidental snapshotting of datasets, model weights, and
        # other large generated assets.  0 disables the filter.
        "max_file_size_mb": 10,
        # Auto-maintenance: hermes sweeps the checkpoint base at startup
        # (at most once per ``min_interval_hours``) and:
        #   * deletes project entries whose last_touch is older than
        #     ``retention_days``
        #   * GCs the single shared store to reclaim unreachable objects
        #   * enforces ``max_total_size_mb`` across remaining projects
        #   * deletes ``legacy-*`` archives older than ``retention_days``
        #
        # NOTE: this automatic sweep never deletes "orphan" entries (workdir
        # no longer found on disk). A missing workdir at startup is
        # ambiguous — it can mean the project was deleted, or that an
        # external volume / network share / VPN is simply not mounted yet —
        # and this sweep runs unattended, so it must never guess. Orphan
        # cleanup is only available via the explicit
        # ``hermes checkpoints prune`` command (add ``--keep-orphans`` to
        # skip it), where a human is looking at the output.
        "auto_prune": True,
        "retention_days": 7,
        "min_interval_hours": 24,
    },

    # Hard cap (chars) for a single automatic context file such as SOUL.md,
    # AGENTS.md, CLAUDE.md, .hermes.md, or .cursorrules before Hermes applies
    # head/tail truncation. ``null`` (the default) lets the cap scale with the
    # model's context window (floor 20K, ceiling 500K) so large-context models
    # rarely truncate a project doc. Set a positive integer to pin a fixed cap
    # and override the dynamic behavior. Separate from read_file tool limits.
    "context_file_max_chars": None,

    # Maximum characters returned by a single read_file call.  Reads that
    # exceed this are rejected with guidance to use offset+limit.
    # 100K chars ≈ 25–35K tokens across typical tokenisers.
    "file_read_max_chars": 100_000,

    # Seconds to wait at agent-build time for in-flight MCP server discovery
    # to finish before the agent snapshots its tool list. MCP discovery runs
    # in background asyncio tasks; this bounds how long the first agent build
    # awaits a slow/dead server. The wait returns
    # the INSTANT discovery completes, so users with no MCP servers (the common
    # case) or fast servers pay ~0s regardless of this value — the bound is
    # only reached when a server is genuinely still connecting.  The old 0.75s
    # default was a touch short for HTTP/OAuth servers on a cold connect; a
    # modest bump lets more of them land in the FIRST turn's snapshot.  This is
    # only a turn-1 latency/UX knob: a server that misses this window is still
    # picked up automatically on the next turn by the between-turns refresh
    # (see agent/turn_context.py), so correctness never depends on it.  Keep it
    # small so a slow/dead server adds little to first-response latency.
    "mcp_discovery_timeout": 1.5,

    # Single-query (``hermes -q/-z "..."``) variant of mcp_discovery_timeout.
    # In one-shot mode there is only ONE turn, so the between-turns late-binding
    # refresh never runs: a server that misses the small interactive bound is
    # invisible to the LLM for the whole session.  This larger bound gives slow
    # cold-start servers (npx, uvx, remote HTTP) a chance to land in the one
    # tool snapshot.  ``thread.join(timeout)`` returns the instant discovery
    # completes, so reachable servers only wait for their real handshake time
    # while unavailable servers remain bounded.
    "mcp_single_query_discovery_timeout": 15.0,

    # MCP runtime behavior (distinct from the per-server definitions in
    # mcp_servers: and from the auxiliary.mcp side-LLM task settings).
    "mcp": {
        # Auto-reload MCP connections when config.yaml's mcp_servers section
        # changes at runtime (CLI file watcher, default on).
        # Set to false to stop the automatic reload: every automatic reload
        # rebuilds the agent tool surface and INVALIDATES the provider
        # prompt cache (the next message re-sends the full input prefix),
        # which is expensive on long-context / high-reasoning models.
        # When disabled, the watcher still detects the change and prints
        # guidance to apply it deliberately via /reload-mcp.
        "auto_reload_on_config_change": True,
    },

    # Tool-output truncation thresholds. When terminal output or a
    # single read_file page exceeds these limits, Hermes truncates the
    # payload sent to the model (keeping head + tail for terminal,
    # enforcing pagination for read_file). Tuning these trades context
    # footprint against how much raw output the model can see in one
    # shot. Ported from anomalyco/opencode PR #23770.
    #
    # - max_bytes:       terminal_tool output cap, in chars
    #                    (default 50_000 ≈ 12-15K tokens).
    # - max_lines:       read_file pagination cap — the maximum `limit`
    #                    a single read_file call can request before
    #                    being clamped (default 2000).
    # - max_line_length: per-line cap applied when read_file emits a
    #                    line-numbered view (default 2000 chars).
    "tool_output": {
        "max_bytes": 50_000,
        "max_lines": 2000,
        "max_line_length": 2000,
    },

    # Tool loop guardrails nudge models when they repeat failed or
    # non-progressing tool calls. Soft warnings are always-on by default;
    # hard stops are opt-in so interactive CLI/TUI sessions keep flowing.
    "tool_loop_guardrails": {
        "warnings_enabled": True,
        "hard_stop_enabled": False,
        "warn_after": {
            "exact_failure": 2,
            "same_tool_failure": 3,
            "idempotent_no_progress": 2,
        },
        "hard_stop_after": {
            "exact_failure": 5,
            "same_tool_failure": 8,
            "idempotent_no_progress": 5,
        },
        # Per-turn runaway-loop caps (inspired by Claude Code v2.1.212,
        # Week 29, July 2026). Hard ceilings on how many times a runaway-prone
        # tool may be called within a SINGLE agent loop (turn); the counters
        # reset at the start of every turn, so a legitimate multi-turn session
        # is never starved. They are always-on and fire regardless of the
        # warn/hard-stop thresholds above. A single turn issuing dozens of web
        # searches or spawning dozens of subagents is already pathological, so
        # the defaults are low. Set either to 0 to disable that cap (unlimited).
        "loop_caps": {
            "max_web_searches": 50,   # max web_search calls per turn (0 = unlimited)
            "max_subagents": 50,      # max subagents spawned per turn (0 = unlimited)
        },
    },

    "compression": {
        "enabled": True,
        "progress_notices": False,    # opt-in (#52995): when True, routine compression
                                      # progress statuses (compacting/preflight/pre-API/
                                      # idle/retry) are delivered to chat gateway
                                      # platforms instead of being suppressed by the
                                      # gateway noise filter. Default False keeps
                                      # routine compression silent-by-design on chat
                                      # surfaces (server-side logging only). Failure
                                      # notices and manual /compress feedback are
                                      # always visible regardless of this setting.
        "threshold": 0.50,            # compress when context usage exceeds this ratio.
                                      # Models with context windows below 512K are
                                      # floored at 0.75 (raise-only) so compaction
                                      # doesn't fire with half the window still free;
                                      # set this above 0.75 to override the floor.
        "threshold_tokens": None,     # absolute token cap — when set, compression
                                      # triggers at the lower of the ratio-based
                                      # threshold and this token count. Clamped to
                                      # the model's context length at apply-time.
        "target_ratio": 0.20,         # fraction of threshold to preserve as recent tail
        "protect_last_n": 20,         # minimum recent messages to keep uncompressed
        "min_tail_user_messages": 1,  # REAL (actionable) user messages guaranteed to
                                      # survive in the uncompressed tail. 1 = existing
                                      # single last-user anchor (default, behavior-
                                      # preserving); raise to e.g. 3 to keep the last
                                      # 3 real user turns verbatim when bulky tool
                                      # outputs fill the tail token budget.
        "max_attempts": 3,            # compression retry rounds before a turn gives up
                                      # with "max compression attempts reached". Raise
                                      # (e.g. 6) for tool-schema-heavy sessions where 3
                                      # rounds cannot clear the request estimate.
                                      # Validated >= 1, hard-capped at 10.
        "proactive_prune_tokens": 0,  # opt-in trigger (tokens) for the deterministic,
                                      # no-LLM tool-result prune, run independently of
                                      # `threshold` above. On large-window models
                                      # `threshold` (≈50% of the window) rarely fires,
                                      # so old tool output otherwise rides in history
                                      # and is re-sent every turn; a low value like
                                      # 48000 reclaims it early. 0 = off. Recent tail
                                      # protected by `protect_last_n`. Built-in
                                      # compressor only (other engines inherit a no-op).
                                      # NOTE: each committed prune rewrites already-sent
                                      # history, breaking the provider prompt-cache
                                      # prefix — the min_reclaim gate below keeps those
                                      # breaks episodic rather than per-turn.
        "proactive_prune_min_result_chars": 8000,  # the prune's summarize pass only
                                      # touches tool results larger than this (chars);
                                      # clamped to >= 200 so a generated summary can't
                                      # itself be re-summarized.
        "proactive_prune_min_reclaim_tokens": 4096,  # a proactive prune only commits
                                      # when it reclaims at least this many tokens
                                      # (measured on the pruned output). Keeps
                                      # prompt-cache invalidation amortized: one big
                                      # episodic break instead of a tiny break every
                                      # tool iteration. 0 = commit any non-zero prune.
        "micro_compact": False,       # opt-in: after each completed turn, fold the
                                      # oldest un-absorbed exchange into a rolling
                                      # summary, amortizing compression cost instead
                                      # of paying it in one batch stall. Default False
                                      # because a pass rewrites already-sent history
                                      # and so breaks the provider prompt-cache prefix
                                      # EVERY turn — the per-turn cache break that
                                      # `proactive_prune_min_reclaim_tokens` above
                                      # exists to avoid. Enable only when you have
                                      # measured that the amortized stall is worth
                                      # more to you than the cached-prefix discount.
                                      # See docs/micro-compaction.md.
        "micro_compact_every_n_turns": 1,  # cadence: run a pass every Nth completed
                                      # turn. Since each pass costs one prompt-cache
                                      # break, this is the dial for how often that
                                      # cost is paid — 1 reclaims most aggressively
                                      # at one break per turn, 5 trades reclaim rate
                                      # for a fifth of the breaks. Clamped to >= 1.
                                      # Ignored unless `micro_compact` is true.
        "micro_compact_defrag_threshold_tokens": 2000,  # once the rolling summary
                                      # exceeds this many tokens, the next pass
                                      # re-summarizes the summary itself instead of
                                      # letting it grow without bound.
        "hygiene_hard_message_limit": 5000,  # gateway session-hygiene force-compress threshold by message count
        "hygiene_timeout_seconds": 30,  # max seconds gateway waits for pre-agent hygiene compression
                                      # WITHOUT forward progress. The summary call streams, so
                                      # this is an inactivity budget: a slow model still
                                      # producing tokens keeps extending the wait; only a
                                      # silent/hung call is cut off.
        "hygiene_total_ceiling_seconds": 600,  # absolute cap on the hygiene compression wait even
                                      # while tokens are still moving — bounds a degenerate
                                      # trickle stream. Clamped to >= hygiene_timeout_seconds.
        "hygiene_failure_cooldown_seconds": 300,  # skip repeated failed hygiene attempts for this session
        "context_timeout_seconds": 120,  # inactivity budget for in-agent compress_context
                                      # (conversation loop, /compress, preflight, etc.).
                                      # Same progress-aware semantics as hygiene_timeout_seconds:
                                      # streamed summary tokens extend the wait; only a silent
                                      # worker is cut off. 0 = disable the owned wrapper
                                      # (callers that already pass commit_fence, e.g. gateway
                                      # hygiene, never use this path).
        "context_total_ceiling_seconds": 600,  # absolute cap on the *pre-commit*
                                      # in-agent compress_context wait (summary /
                                      # stream phase) even while tokens are still
                                      # moving. Clamped to >= context_timeout_seconds
                                      # when the idle budget is > 0. Guarantee:
                                      # the summary phase is bounded by this
                                      # ceiling; an already-started SessionDB
                                      # commit is never abandoned mid-flight —
                                      # if the commit itself runs past the
                                      # ceiling it is logged (WARNING, then
                                      # ERROR) and surfaced to the user via the
                                      # warning channel while the host keeps
                                      # waiting in bounded increments for the
                                      # commit to finish.
        "protect_first_n": 3,         # non-system head messages always preserved
                                      # verbatim, in ADDITION to the system prompt
                                      # (which is always implicitly protected). Set to
                                      # 0 for long-running rolling-compaction sessions
                                      # where you want nothing pinned except the
                                      # system prompt + rolling summary + recent tail.
        "abort_on_summary_failure": False,  # When True, auto-compression that fails
                                      # to generate a summary (aux LLM errored / returned
                                      # non-JSON / timed out) aborts entirely instead of
                                      # dropping the middle window with a static
                                      # "summary unavailable" placeholder.  Messages are
                                      # preserved unchanged and the session "freezes" at
                                      # its current size until the user runs /compress
                                      # (which bypasses the failure cooldown) or /new.
                                      # Default False matches historical behavior; set to
                                      # True if you'd rather pause than silently lose
                                      # context turns when your aux model is flaky.
        "codex_gpt55_autoraise": True,  # Historical key name kept for compatibility.
                                      # When True, gpt-5.4 / gpt-5.5 / gpt-5.6 on the
                                      # ChatGPT Codex OAuth route raise their compaction
                                      # trigger to 85% (vs the global `threshold` above).
                                      # Codex hard-caps these families at a 272K window, so
                                      # the default 50% would compact at ~136K and waste half
                                      # the usable context. Set to False to opt back down to
                                      # the global threshold (e.g. 0.50) for those Codex
                                      # sessions. Only this exact route is affected —
                                      # gpt-5.4 / 5.5 / 5.6 on OpenAI's direct API,
                                      # OpenRouter, and Copilot keep the global threshold
                                      # regardless.
        "codex_gpt55_autoraise_notice": True,  # Display the one-time Codex gpt-5.4/5.5/5.6
                                      # autoraise banner. Set False to keep the
                                      # 85% threshold autoraise but suppress the
                                      # user-facing notice in CLI/gateway output.
        "in_place": True,             # When True, compaction rewrites the message
                                      # list and rebuilds the system prompt WITHOUT
                                      # rotating the session id — the conversation
                                      # keeps one durable id for its whole life
                                      # (no parent_session_id chain, no `name #N`
                                      # renumbering). Eliminates the session-rotation
                                      # bug cluster (#33618 /goal loss, #14238 lost
                                      # response, #33907 orphans, #45117 search gaps,
                                      # #42228 null cwd) — see #38763. Non-destructive:
                                      # the live context is compacted (lossy for what
                                      # the model reloads), but the pre-compaction
                                      # turns are soft-archived under the same id
                                      # (active=0, compacted=1) — still searchable via
                                      # session_search and recoverable, not deleted.
                                      # Default True since 2107b86024; set False to
                                      # restore the legacy rotating-compaction path.
        "model_thresholds": {},       # Per-model threshold overrides. Keys are
                                      # substring-matched against the model name
                                      # (longest match wins); values replace the
                                      # global `threshold` for that model, e.g.
                                      #   model_thresholds:
                                      #     "glm-5.2": 0.40
                                      #     "claude-sonnet": 0.35
                                      # The small-context floor (0.75 for <512K
                                      # models) still applies on top of overrides
                                      # (raise-only: an override above the floor
                                      # wins; one below it is raised to the floor).
        "idle_compact_after_seconds": 0,  # Opt-in idle compaction (0 = disabled).
                                      # When > 0, a session that resumes after at
                                      # least this many seconds of inactivity
                                      # compacts its accumulated history up front,
                                      # before the first reply — so a long-lived
                                      # thread resumed hours later doesn't re-read
                                      # its full stale context on every turn.
                                      # Time-based; complements (does not replace)
                                      # the size-based `threshold` above. Skipped
                                      # when the context is already at/below the
                                      # post-compression target (threshold ×
                                      # target_ratio) and it honors the same
                                      # failure-cooldown / anti-thrash / per-session
                                      # lock guards as every automatic compaction.
                                      # Example: 1800 = compact after 30 min idle.
    },

    # Anthropic prompt caching (Claude via OpenRouter or native Anthropic API).
    # cache_ttl: "5m" or "1h" (Anthropic-supported tiers). Other non-falsy
    # values are silently ignored. Falsy values (false, null, "off",
    # "disabled", "no", "none") disable prompt caching entirely.
    "prompt_caching": {
        "cache_ttl": "5m",
    },

    # OpenRouter-specific settings.
    # response_cache: enable OpenRouter response caching (X-OpenRouter-Cache header).
    #   When enabled, identical requests return cached responses for free (zero billing).
    #   This is separate from Anthropic prompt caching and works alongside it.
    #   See: https://openrouter.ai/docs/guides/features/response-caching
    # response_cache_ttl: how long cached responses remain valid, in seconds (1-86400).
    #   Default 300 (5 minutes). Only used when response_cache is enabled.
    # min_coding_score: knob for the openrouter/pareto-code router (0.0-1.0).
    #   Only applied when model.model is "openrouter/pareto-code". Higher
    #   values route to stronger (more expensive) coders; lower values open
    #   up cheaper, faster options. Default 0.65 lands on the mid-tier
    #   coder on the current Pareto frontier. Empty string = let OpenRouter
    #   pick the strongest available coder (router's documented default
    #   when the plugins block is omitted).
    #   See: https://openrouter.ai/docs/guides/routing/routers/pareto-router
    "openrouter": {
        "response_cache": True,
        "response_cache_ttl": 300,
        "min_coding_score": 0.65,
    },

    # AWS Bedrock provider configuration.
    # Only used when model.provider is "bedrock".
    "bedrock": {
        "region": "",  # AWS region for Bedrock API calls (empty = AWS_REGION env var → us-east-1)
        "discovery": {
            "enabled": True,           # Auto-discover models via ListFoundationModels
            "provider_filter": [],     # Only show models from these providers (e.g. ["anthropic", "amazon"])
            "refresh_interval": 3600,  # Cache discovery results for this many seconds
        },
        "guardrail": {
            # Amazon Bedrock Guardrails — content filtering and safety policies.
            # Create a guardrail in the Bedrock console, then set the ID and version here.
            # See: https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html
            "guardrail_identifier": "",  # e.g. "abc123def456"
            "guardrail_version": "",     # e.g. "1" or "DRAFT"
            "stream_processing_mode": "async",  # "sync" or "async"
            "trace": "disabled",         # "enabled", "disabled", or "enabled_full"
        },
    },

    # Auxiliary model config — provider:model for each side task.
    # Format: provider is the provider name, model is the model slug.
    # "auto" for provider = auto-detect best available provider.
    # Empty model = use provider's default auxiliary model.
    # All tasks fall back to openrouter:google/gemini-3-flash-preview if
    # the configured provider is unavailable.
    #
    # extra_body: forwarded verbatim as request body fields on every aux call
    # for that task. Use this to set provider-specific knobs (independent of
    # main-agent settings). On OpenRouter you can set provider routing prefs
    # and the Pareto Code coding-score floor here. Example:
    #
    #   auxiliary:
    #     compression:
    #       provider: openrouter
    #       model: openrouter/pareto-code
    #       extra_body:
    #         provider:           # OpenRouter provider routing
    #           order: [anthropic, google]
    #           sort: throughput  # or price | latency
    #         plugins:            # OpenRouter Pareto Code router
    #           - id: pareto-router
    #             min_coding_score: 0.5
    #
    # Each aux task is independent — main-agent provider_routing and
    # openrouter.min_coding_score do NOT propagate to aux calls by design.
    "auxiliary": {
        # Same-provider retries for a transient transport blip (connection
        # reset / timeout / 5xx / 408) on ANY auxiliary call before falling
        # back. Default 2 (→ 3 total attempts), clamped [0,6]. Matters most for
        # pinned calls like MoA reference advisors, where provider fallback is
        # not a meaningful recovery, so an unretried blip silently loses the
        # call.
        "transient_retries": 2,
        # Restrict the auxiliary auto-chain's OpenRouter fallback to free
        # (:free) SKUs. When true, the OpenRouter step is skipped entirely
        # unless the resolved fallback model ends in ":free" — a PAID lane
        # is never engaged for background auxiliary traffic (compression,
        # title generation, session search, vision, web extract) even when
        # OPENROUTER_API_KEY is present. Default false keeps the historical
        # paid fallback for users who want it.
        "free_only": False,
        # Override the auxiliary auto-chain's OpenRouter fallback model
        # (default: google/gemini-3.6-flash, a PAID model). Set e.g.
        # "nvidia/nemotron-3-ultra-550b-a55b:free" together with
        # free_only: true to keep auxiliary traffic free-only. A one-time
        # WARNING is logged whenever a non-":free" model is engaged.
        "openrouter_model": "",
        # Endpoints that reject NON-streaming chat requests outright (e.g.
        # Tencent Copilot returns HTTP 400 "Non-stream chat request is
        # currently not supported"). Auxiliary calls to a matching endpoint
        # are sent with stream=True and aggregated client-side. Entries are
        # case-insensitive substrings matched against the endpoint URL;
        # copilot.tencent.com is always treated as stream-only.
        "stream_only_base_urls": [],
        "vision": {
            "provider": "auto",    # auto | openrouter | nous | codex | custom
            "model": "",           # e.g. "google/gemini-2.5-flash", "gpt-4o"
            "base_url": "",        # direct OpenAI-compatible endpoint (takes precedence over provider)
            "api_key": "",         # API key for base_url (falls back to OPENAI_API_KEY)
            "timeout": 120,        # seconds — LLM API call timeout; vision payloads need generous timeout
            "extra_body": {},      # OpenAI-compatible provider-specific request fields
            "reasoning_effort": "",  # per-task thinking level: none|minimal|low|medium|high|xhigh|max|ultra (empty = provider default)
            "download_timeout": 30,  # seconds — image HTTP download timeout; increase for slow connections
        },
        "web_extract": {
            "provider": "auto",
            "model": "",
            "base_url": "",
            "api_key": "",
            "timeout": 360,        # seconds (6min) — per-attempt LLM summarization timeout; increase for slow local models
            "extra_body": {},
            "reasoning_effort": "",  # per-task thinking level: none|minimal|low|medium|high|xhigh|max|ultra (empty = provider default)
        },
        "compression": {
            "provider": "auto",
            "model": "",
            "base_url": "",
            "api_key": "",
            "timeout": 120,        # seconds — compression summarises large contexts; increase for local models
            "extra_body": {},
            "reasoning_effort": "",  # per-task thinking level: none|minimal|low|medium|high|xhigh|max|ultra (empty = provider default)
        },
        # Note: session_search no longer uses an auxiliary LLM (PR #27590 —
        # single-shape tool returns DB content directly). The old
        # ``auxiliary.session_search.*`` block was removed here. Existing
        # values in user config.yaml files are harmless leftovers and ignored.
        "skills_hub": {
            "provider": "auto",
            "model": "",
            "base_url": "",
            "api_key": "",
            "timeout": 30,
            "extra_body": {},
            "reasoning_effort": "",  # per-task thinking level: none|minimal|low|medium|high|xhigh|max|ultra (empty = provider default)
        },
        "approval": {
            "provider": "auto",
            "model": "",           # fast/cheap model recommended (e.g. gemini-flash, haiku)
            "base_url": "",
            "api_key": "",
            "timeout": 30,
            "extra_body": {},
            "reasoning_effort": "",  # per-task thinking level: none|minimal|low|medium|high|xhigh|max|ultra (empty = provider default)
        },
        "mcp": {
            "provider": "auto",
            "model": "",
            "base_url": "",
            "api_key": "",
            "timeout": 30,
            "extra_body": {},
            "reasoning_effort": "",  # per-task thinking level: none|minimal|low|medium|high|xhigh|max|ultra (empty = provider default)
        },
        "title_generation": {
            "enabled": True,
            "provider": "auto",
            "model": "",
            "base_url": "",
            "api_key": "",
            "timeout": 30,
            "extra_body": {},
            "reasoning_effort": "",  # per-task thinking level: none|minimal|low|medium|high|xhigh|max|ultra (empty = provider default)
            "language": "",
        },
        "memory_query_rewrite": {
            "provider": "auto",
            "model": "",
            "base_url": "",
            "api_key": "",
            "timeout": 8,
            "extra_body": {},
        },
        # Profile describer — auto-generates a 1-2 sentence description
        # of what a profile is good at. Invoked by
        # ``hermes profile describe <name> --auto`` and the dashboard's
        # auto-generate button. Short, cheap call.
        "profile_describer": {
            "provider": "auto",
            "model": "",
            "base_url": "",
            "api_key": "",
            "timeout": 60,
            "extra_body": {},
            "reasoning_effort": "",  # per-task thinking level: none|minimal|low|medium|high|xhigh|max|ultra (empty = provider default)
        },
        # Goal judge — evaluates whether a /goal run's latest response
        # satisfies the goal/contract, and drafts goal contracts. Short
        # structured-JSON calls; a fast cheap model is fine.
        "goal_judge": {
            "provider": "auto",
            "model": "",
            "base_url": "",
            "api_key": "",
            "timeout": 60,
            "extra_body": {},
            "reasoning_effort": "",  # per-task thinking level: none|minimal|low|medium|high|xhigh|max|ultra (empty = provider default)
        },
        # Monitor — urgency/importance classifier used by the important-mail
        # monitor catalog automation (cron/scripts/classify_items.py). Scores
        # candidate items 0-10 against the user's criteria so only above-
        # threshold items get delivered. "auto" = main chat model; override to
        # a cheap fast model (e.g. openrouter google/gemini-3-flash-preview,
        # haiku) since per-item scoring is high-volume and a small model is fine.
        "monitor": {
            "provider": "auto",
            "model": "",
            "base_url": "",
            "api_key": "",
            "timeout": 60,
            "extra_body": {},
            "reasoning_effort": "",  # per-task thinking level: none|minimal|low|medium|high|xhigh|max|ultra (empty = provider default)
        },
        "moa_reference": {
            "provider": "auto",
            "model": "",
            "base_url": "",
            "api_key": "",
            "timeout": 900,
            "extra_body": {},
            # NOTE: no reasoning_effort here by design — MoA reasoning depth is
            # configured PER SLOT in the MoA preset (moa.presets.<name>.
            # reference_models[].reasoning_effort / aggregator.reasoning_effort),
            # not at the auxiliary-task level.
        },
        "moa_aggregator": {
            "provider": "auto",
            "model": "",
            "base_url": "",
            "api_key": "",
            "timeout": 900,
            "extra_body": {},
            # NOTE: no reasoning_effort here by design — see moa_reference above.
        },
    },
    
    "display": {
        # Keep model commentary visible to streaming consumers.
        "show_commentary": True,
        # Surface provider credit thresholds through the existing status callback.
        "credits_notices": True,
        # Verify mutations when a turn fails after changing files.
        "file_mutation_verifier": True,
        # Explain abnormal turn completion instead of returning a silent sentinel.
        "turn_completion_explainer": True,
    },

    "privacy": {
        "redact_pii": False,  # When True, hash user IDs and strip phone numbers from LLM context
    },

    # Context engine -- controls how the context window is managed when
    # approaching the model's token limit.
    # "compressor" = built-in lossy summarization (default).
    # Set to a plugin name to activate an alternative engine (e.g. "lcm"
    # for Lossless Context Management).  The engine must be installed as
    # a plugin in plugins/context_engine/<name>/ or ~/.hermes/plugins/.
    "context": {
        "engine": "compressor",
    },

    # Persistent memory -- bounded curated memory injected into system prompt
    "memory": {
        "memory_enabled": True,
        "user_profile_enabled": True,
        "memory_char_limit": 2200,   # ~800 tokens at 2.75 chars/token
        "user_char_limit": 1375,     # ~500 tokens at 2.75 chars/token
        # External memory provider plugin (empty = built-in only).
        # Bundled providers are "mem0" and "byterover"; user-installed
        # providers are discovered from the profile plugin directory.
        # Only ONE external provider is allowed at a time.
        "provider": "",
    },

    # Subagent delegation — override the provider:model used by delegate_task
    # so child agents can run on a different (cheaper/faster) provider and model.
    # Uses the same runtime provider resolution as CLI/gateway startup, so all
    # configured providers (OpenRouter, Nous, Z.ai, Kimi, etc.) are supported.
    "delegation": {
        "model": "",       # e.g. "google/gemini-3-flash-preview" (empty = inherit parent model)
        "provider": "",    # e.g. "openrouter" (empty = inherit parent provider + credentials)
        "base_url": "",    # direct OpenAI-compatible endpoint for subagents
        "api_key": "",     # API key for delegation.base_url (falls back to OPENAI_API_KEY)
        "api_mode": "",    # wire protocol for delegation.base_url: "chat_completions",
                           # "codex_responses", or "anthropic_messages". Empty = auto-detect
                           # from URL (e.g. /anthropic suffix → anthropic_messages). Set this
                           # explicitly for non-standard endpoints the heuristic can't detect.
        # When delegate_task narrows child toolsets explicitly, preserve any
        # MCP toolsets the parent already has enabled. On by default so
        # narrowing (e.g. toolsets=["web","browser"]) expresses "I want these
        # extras" without silently stripping MCP tools the parent already has.
        # Set to false for strict intersection.
        "inherit_mcp_toolsets": True,
        "max_iterations": 50,  # per-subagent iteration cap (each subagent gets its own budget,
                               # independent of the parent's max_iterations)
        # Subagent summaries return to the parent's context verbatim. A batch
        # fan-out (N children) returns N summaries at once, which can exceed
        # the parent's context window and trigger a compression/429 death
        # spiral. delegate_task sizes each summary against the parent's
        # remaining context headroom (split across the batch); when it must
        # trim, the full text is spilled to ~/.hermes/cache/delegation/
        # (mounted into remote backends) and the in-context summary becomes a
        # head+tail window plus a footer with the exact read_file offset to
        # page the omitted middle — the same convention web_extract uses for
        # large pages. Nothing is lost. max_summary_chars is a hard per-summary
        # character ceiling layered on top of that dynamic budget
        # (belt-and-suspenders for models that ignore the "be concise"
        # instruction). 0 disables the hard ceiling; the dynamic headroom
        # budget still applies.
        "max_summary_chars": 24000,

        "child_timeout_seconds": 0,  # optional wall-clock cap per child agent. 0 (default)
                                     # = no timeout: children fail only from real errors
                                     # (API, tools, iteration budget), never a delegation
                                     # stopwatch. Set a positive number of seconds
                                     # (floor 30s) to enforce a hard cap.
        "reasoning_effort": "",  # subagent effort: "ultra", "max", "xhigh", "high",
                                 # "medium", "low", "minimal", "none" (empty = inherit)
        "max_concurrent_children": 3,  # unified concurrency cap: max parallel children per batch
                                       # AND max concurrent background (background=true)
                                       # delegation units. New async dispatches beyond the cap
                                       # fall back to synchronous execution. Floor of 1, no ceiling.
                                       # (Replaces the deprecated max_async_children.)
        # Orchestrator role controls (see tools/delegate_tool.py:_get_max_spawn_depth
        # and _get_orchestrator_enabled).  Floored at 1, no upper ceiling —
        # raise deliberately, each level multiplies API cost.
        "max_spawn_depth": 1,        # depth (1 = flat [default], 2 = orchestrator→leaf, 3+ = deeper)
        "orchestrator_enabled": True,  # kill switch for role="orchestrator"
        # When a subagent hits a dangerous-command approval prompt, the parent's
        # prompt_toolkit TUI owns stdin — a thread-local input() call from the
        # subagent worker would deadlock the parent UI. To avoid the deadlock,
        # subagent threads ALWAYS resolve approvals non-interactively:
        #   false (default) → auto-deny with a logger.warning audit line (safe)
        #   true             → auto-approve "once" with a logger.warning audit line
        # Flip to true only if you trust delegated work to run dangerous cmds
        # without human review (cron pipelines, batch automation, etc.).
        "subagent_auto_approve": False,
    },

    # Ephemeral prefill messages file — JSON list of {role, content} dicts
    # injected at the start of every API call for few-shot priming.
    # Never saved to sessions, logs, or trajectories.
    "prefill_messages_file": "",

    # Goals — persistent cross-turn goals (Ralph-style loop).
    # After every turn, a lightweight judge call asks the auxiliary model
    # whether the active /goal is satisfied by the assistant's last
    # response. If not, Hermes feeds a continuation prompt back into the
    # same session and keeps working until the goal is done, the turn
    # budget is exhausted, or the user pauses/clears it. Judge failures
    # fail OPEN (continue) so a flaky judge never wedges progress — the
    # turn budget is the real backstop.
    "goals": {
        # Max continuation turns before Hermes auto-pauses the goal and
        # asks the user to /goal resume. Protects against judge false
        # negatives (goal actually done but judge says continue) and
        # unbounded model spend on fuzzy / unachievable goals.
        "max_turns": 20,
    },

    # Mixture of Agents — named presets used by /moa. A preset is an execution
    # mode around the main model, not a provider/model itself: references +
    # aggregator synthesize private guidance before each main-model iteration.
    "moa": {
        "default_preset": "default",
        "active_preset": "",
        # When true, every MoA turn that runs the reference fan-out writes the
        # FULL turn (each reference's exact input messages + output + usage/cost,
        # and the aggregator's exact input + output) to a JSONL file at
        # <hermes_home>/moa-traces/<session_id>.jsonl. Off by default — turn it
        # on to audit / improve MoA behavior from real runs. Set trace_dir to
        # override the output directory.
        "save_traces": False,
        "trace_dir": "",
        # Privacy redaction filter for advisor (reference) outputs. Advisors
        # can echo PII from the conversation (emails, formatted phone numbers)
        # and credential shapes into reference blocks, traces, and the
        # aggregator prompt. Modes ('' = off, the default):
        #   "display" — redact user-visible surfaces only (reference blocks
        #               shown in the UI + saved MoA trace records); the
        #               aggregator still sees raw advisor text.
        #   "full"    — additionally redact the advisor text injected into
        #               the aggregator prompt (issue #59959).
        "privacy_filter": "",
        "presets": {
            "default": {
                "reference_models": [
                    {"provider": "openai-codex", "model": "gpt-5.5"},
                    {"provider": "openrouter", "model": "deepseek/deepseek-v4-pro"},
                ],
                "aggregator": {"provider": "openrouter", "model": "anthropic/claude-opus-4.8"},
                "max_tokens": 4096,
                "enabled": True,
            }
        },
    },

    # Skills — external skill directories for sharing skills across tools/agents.
    # Each path is expanded (~, ${VAR}) and resolved.  Read-only — skill creation
    # always goes to ~/.hermes/skills/.
    "skills": {
        "external_dirs": [],   # e.g. ["~/.agents/skills", "/shared/team-skills"]
        # Substitute ${HERMES_SKILL_DIR} and ${HERMES_SESSION_ID} in SKILL.md
        # content with the absolute skill directory and the active session id
        # before the agent sees it.  Lets skill authors reference bundled
        # scripts without the agent having to join paths.
        "template_vars": True,
        # Pre-execute inline shell snippets written as !`cmd` in SKILL.md
        # body.  Their stdout is inlined into the skill message before the
        # agent reads it, so skills can inject dynamic context (dates, git
        # state, detected tool versions, …).  Off by default because any
        # content from the skill author runs on the host without approval;
        # only enable for skill sources you trust.
        "inline_shell": False,
        # Timeout (seconds) for each !`cmd` snippet when inline_shell is on.
        "inline_shell_timeout": 10,
    },

    # IANA timezone (e.g. "Asia/Kolkata", "America/New_York").
    # Empty string means use server-local time.
    "timezone": "",

    # Approval mode for dangerous commands:
    #   manual — always prompt the user
    #   smart  — use auxiliary LLM to auto-approve low-risk commands (default)
    #   off    — skip all approval prompts (equivalent to --yolo)
    #
    # timeout — seconds to wait for the user's approve/deny before failing
    # closed (deny).
    "approvals": {
        "mode": "smart",
        "timeout": 300,
        # Operator-customizable policy text for smart approvals. When
        # non-empty, this is appended to the smart-approval guardian's
        # SYSTEM prompt (trusted channel) as additional rules — e.g.
        # "Always ESCALATE commands touching /etc" or "APPROVE docker
        # compose restarts under ~/deploys". Inspired by ChatGPT Work's
        # customizable auto-review guardian policy.
        "smart_policy": "",
        # Consecutive-denial circuit breaker for smart approvals: after this
        # many guardian DENY verdicts in a row within one session, the deny
        # message returned to the model escalates to a hard-stop instruction
        # (report to the user / ask for manual run or /approve) instead of a
        # plain "Do NOT retry". Any approval resets the count. 0 disables.
        # Inspired by ChatGPT Work's auto-review circuit breaker.
        "denial_breaker_threshold": 3,
        # User-defined deny rules: fnmatch globs matched against terminal
        # commands. A match blocks the command unconditionally — BEFORE the
        # --yolo / /yolo / mode=off bypass — making this the user-editable
        # counterpart to the code-shipped hardline blocklist. Patterns are
        # case-insensitive and must be quoted in YAML when they start with
        # * or contain {}/!/: sequences. Example:
        #   deny:
        #     - "git push --force*"
        #     - "*curl*|*sh*"
        "deny": [],
    },

    # Permanently allowed dangerous command patterns (added via "always" approval)
    "command_allowlist": [],
    # Per-platform system-prompt hint overrides for embedding applications.
    # Each key is a platform name; the value is either:
    #   { "append": "extra text" }   — keep the default hint, append text
    #   { "replace": "full text" }   — substitute the default hint entirely
    #   "extra text"                 — shorthand for { "append": ... }
    # `replace` wins over `append` if both are given. Example:
    #   platform_hints:
    #     web:
    #       append: >
    #         When tabular output would be useful, invoke the
    #         table_formatting skill instead of emitting a Markdown table.
    "platform_hints": {},

    # Shell-script hooks — declarative bridge that invokes shell scripts
    # on plugin-hook events (pre_tool_call, post_tool_call, pre_llm_call,
    # subagent_stop, etc.).  Each entry maps an event name to a list of
    # {matcher, command, timeout} dicts.  First registration of a new
    # command prompts the user for consent; subsequent runs reuse the
    # stored approval from ~/.hermes/shell-hooks-allowlist.json.
    # See `website/docs/user-guide/features/hooks.md` for schema + examples.
    "hooks": {},

    # Auto-accept shell-hook registrations without a TTY prompt.  Also
    # toggleable per-invocation via --accept-hooks or HERMES_ACCEPT_HOOKS=1.
    # Non-interactive embedders need this to pick up newly-added hooks.
    "hooks_auto_accept": False,
    # Custom personalities — add your own entries here
    # Supports string format: {"name": "system prompt"}
    # Or dict format: {"name": {"description": "...", "system_prompt": "...", "tone": "...", "style": "..."}}
    "personalities": {},

    "security": {
        "allow_private_urls": False,  # Allow requests to private/internal IPs (for OpenWrt, proxies, VPNs)
        "redact_secrets": True,
        "website_blocklist": {
            "enabled": False,
            "domains": [],
            "shared_files": [],
        },
        # Acknowledged supply-chain security advisories. Each entry is the
        # ID of an advisory the user has read and acted on (uninstalled the
        # compromised package, rotated credentials). Acked advisories no
        # longer trigger the startup banner. Add via `hermes doctor --ack
        # <id>`; remove by editing the list directly. See
        # ``hermes_cli/security_advisories.py`` for the catalog.
        "acked_advisories": [],
    },

    # Tool Search (progressive disclosure for large tool surfaces).
    # When the model is connected to many MCP servers or non-core plugin
    # tools, their JSON schemas can consume a substantial fraction of the
    # context window on every turn. When enabled, those tools are replaced
    # in the model-facing tools array with three bridge tools —
    # tool_search / tool_describe / tool_call — and surfaced on demand.
    #
    # Core Hermes tools (terminal, read_file, write_file, patch,
    # search_files, todo, memory, browser_*, etc.) are NEVER deferred.
    # See tools/tool_search.py for full design notes and the
    # openclaw-tool-search-report PDF in this PR for the rationale.
    "tools": {
        "tool_search": {
            # Tiered disclosure: any deferrable (MCP/plugin) tool activates
            # the bridge; the listing then scales with catalog size.
            #   Tier 0 — no MCP/plugin tools: everything stays eager.
            #   Tier 1 — catalog listing fits the budget: bridge + skills-style
            #     name+description manifest (degrades to names-only).
            #   Tier 2 — per-tool listing over budget even names-only (e.g.
            #     Cloudflare's ~3,300-tool flat API surface): bare bridge +
            #     a one-line-per-server summary (name + tool count) so the
            #     model knows which domains are reachable; individual tools
            #     discoverable through tool_search only.
            # "auto"/"on" — activate when at least one deferrable tool exists.
            # "off" — disable entirely. Tools-array assembly is a pass-through.
            "enabled": "auto",
            # Listing budget as a percentage of the active model's context
            # length. Effective budget = min(this % of context,
            # listing_max_tokens). Range 0..100.
            "threshold_pct": 5,
            # When the model calls tool_search without a ``limit`` argument,
            # how many hits to return. Range 1..max_search_limit.
            "search_default_limit": 5,
            # Hard upper bound the model can request via ``limit``. Range 1..50.
            "max_search_limit": 20,
            # Skills-style catalog listing embedded in the tool_search bridge
            # description: every deferred tool's name + first sentence of its
            # description (≤60 chars), grouped by MCP server / toolset. Keeps
            # capabilities discoverable while schemas stay deferred.
            # "auto" (default) — include when the listing fits the budget
            #   (falls back to names-only, then to the bare tier-2 bridge).
            # "on"  — same rendering, but explicit intent to always list.
            # "off" — always the bare bridge (tier 2 for every catalog).
            "listing": "auto",
            # Absolute cap on the embedded listing in tokens (chars/4
            # estimate), regardless of context size. Range 200..60000.
            "listing_max_tokens": 4000,
        },
    },

    # Logging — controls file logging to ~/.hermes/logs/.
    # agent.log captures INFO+ (all agent activity); errors.log captures WARNING+.
    "logging": {
        "level": "INFO",       # Minimum level for agent.log: DEBUG, INFO, WARNING
        "max_size_mb": 5,      # Max size per log file before rotation
        "backup_count": 3,     # Number of rotated backup files to keep
    },

    # Remotely-hosted model catalog manifest.  When enabled, the CLI fetches
    # curated model lists for OpenRouter and Nous Portal from this URL,
    # falling back to the in-repo snapshot on network failure.  Lets us
    # update model picker lists without shipping a hermes-agent release.
    # The default URL is served by the docs site GitHub Pages deploy.
    "model_catalog": {
        "enabled": True,
        "url": "https://hermes-agent.nousresearch.com/docs/api/model-catalog.json",
        # Disk cache TTL in hours.  Beyond this, the CLI refetches on the
        # next /model or `hermes model` invocation; network failures
        # silently fall back to the stale cache.
        "ttl_hours": 1,
        # Optional per-provider override URLs for third parties that want
        # to self-host their own curation list using the same schema.
        # Example:
        #   providers:
        #     openrouter:
        #       url: https://example.com/my-curation.json
        "providers": {},
    },

    # Network settings — workarounds for connectivity issues.
    "network": {
        # Force IPv4 connections.  On servers with broken or unreachable IPv6,
        # Python tries AAAA records first and hangs for the full TCP timeout
        # before falling back to IPv4.  Set to true to skip IPv6 entirely.
        "force_ipv4": False,
    },

    # Session storage — controls automatic cleanup of ~/.hermes/state.db.
    # state.db accumulates every session, message, tool call, and FTS5 index
    # entry forever.  Without auto-pruning, a heavy user (gateway + cron)
    # reports 384MB+ databases with 68K+ messages, which slows down FTS5
    # inserts, /resume listing, and insights queries.
    "sessions": {
        # When true, prune ended sessions inactive for retention_days once
        # per (roughly) min_interval_hours at CLI/gateway/cron startup.
        # Activity is the latest message timestamp, falling back to creation
        # time for empty sessions. Active sessions are always preserved.
        # Default false: session history is valuable for search recall, and
        # silently deleting it could surprise users.  Opt in explicitly.
        "auto_prune": False,
        # How many inactive days of ended-session history to keep. Matches
        # the default of ``hermes sessions prune``.
        "retention_days": 90,
        # When true, auto-archive (soft-hide, never delete) sessions that
        # haven't been touched in ``auto_archive_days`` days, once per
        # (roughly) min_interval_hours.  "Touched" is last activity, not
        # creation, so an old-but-recently-used session is spared.  Pinned
        # sessions are always exempt.  Off by default — opt in explicitly.
        "auto_archive": False,
        # Idle threshold (days of no activity) before auto-archive hides a
        # session.  Only applies when auto_archive is true.
        "auto_archive_days": 3,
        # VACUUM after a prune that actually deleted rows.  SQLite does not
        # reclaim disk space on DELETE — freed pages are just reused on
        # subsequent INSERTs — so without VACUUM the file stays bloated
        # even after pruning.  VACUUM blocks writes for a few seconds per
        # 100MB, so it only runs at startup, and only when prune deleted
        # ≥1 session.
        "vacuum_after_prune": True,
        # Minimum days between successful VACUUM rewrites. Pruning can still
        # run on its normal cadence while SQLite reuses the freed pages.
        "min_vacuum_interval_days": 30,
        # Minimum hours between auto-maintenance runs (avoids repeating
        # the sweep on every CLI invocation).  Tracked via state_meta in
        # state.db itself, so it's shared across all processes.
        "min_interval_hours": 24,
        # Legacy per-session JSON snapshot writer.  When true, the agent
        # rewrites ``~/.hermes/sessions/session_{sid}.json`` on every turn
        # boundary with the full message list.  state.db is canonical and
        # has every field the snapshot stored (plus per-message timestamps
        # and token counts), so this is off by default — the snapshots had
        # no consumer outside their own overwrite guard and accumulated
        # GBs of disk on heavy users.  Opt in only if you have an external
        # tool that consumes the JSON files directly.
        "write_json_snapshots": False,
        # Search-index (FTS) storage optimization — the compact v23 layout
        # that drops duplicate content copies and stops trigram-indexing tool
        # output (typically reclaims ~60%+ of state.db on heavy users). It is
        # OPT-IN: existing databases keep their working legacy index until the
        # user runs `hermes sessions optimize-storage`, because the rebuild is
        # disk-heavy and long on large DBs (see that command's disk preflight).
        #
        #   "advise" (default): `hermes update` prints a one-line notice with
        #     the reclaimable size and the command, when a legacy index is
        #     detected. Nothing is changed automatically.
        #   "require": the notice is shown as a REQUIRED upgrade (firmer copy),
        #     and future tooling may gate on it. Flip this default in a future
        #     release when we're ready to make the v23 layout mandatory — the
        #     command, progress bar, and resumability are already in place, so
        #     enforcement is a copy/gating change, not new migration code.
        #   "off": suppress the notice entirely.
        "fts_optimize_notice": "advise",
        # CJK-bigram search index (messages_fts_cjk, cjk_unicode61 loadable
        # tokenizer). When the extension is built (native/fts5_cjk/build.sh →
        # ~/.hermes/lib/libfts5_cjk.so), 1-2 char CJK terms (일본, 项目, ...)
        # get index-speed exact matching instead of LIKE full-table scans.
        # True (default): use the index when the extension is present; the
        # setting is inert when it isn't. False: never load the extension or
        # serve the cjk index. Bridged to HERMES_CJK_FTS (internal carrier).
        "cjk_fts": True,
        # Slow session-search log threshold in milliseconds: searches at or
        # above it log one INFO line with the routing path taken (fts_cjk /
        # fts5 / trigram / like_scan) so latency regressions stay
        # attributable per query shape. 0 logs every search. Bridged to
        # HERMES_SEARCH_SLOW_MS (internal carrier).
        "search_slow_ms": 1000,
    },

    # =========================================================================
    # External secret sources
    # =========================================================================
    # Pull credentials from external secret managers at process startup
    # rather than storing them in ~/.hermes/.env.
    "secrets": {
        # Optional explicit ordering of enabled secret sources.  When
        # omitted, sources run in registration order (bundled first,
        # then plugin-registered).  Regardless of this list, "mapped"
        # sources (explicit VAR→ref bindings, e.g. a future 1Password
        # env: map) always take precedence over "bulk" sources
        # (project dumps like Bitwarden BSM), and the first source to
        # claim a var wins — later claims are skipped with a warning.
        # Example: sources: [onepassword, bitwarden]
        # "sources": [],
        "bitwarden": {
            # Master switch.  When false, BSM is never contacted and the
            # bws binary is never auto-installed — same as not having
            # this section at all.
            "enabled": False,
            # Name of the env var that holds the Bitwarden machine-account
            # access token.  This is the one bootstrap secret; it lives
            # in ~/.hermes/.env (or your shell) and never in config.yaml.
            "access_token_env": "BWS_ACCESS_TOKEN",
            # UUID of the BSM project to sync from.
            "project_id": "",
            # Seconds to reuse a fresh disk/memory cache entry before contacting
            # Bitwarden again. 0 disables normal fresh-cache reuse.
            "cache_ttl_seconds": 300,
            # Optional encrypted last-good fallback for network/timeout outages.
            # When enabled, successful BWS fetches write AES-GCM encrypted cache
            # material under ~/.hermes/cache/. If a later startup cannot reach
            # Bitwarden due to NETWORK/TIMEOUT, Hermes may use this encrypted
            # cache for up to max_stale_seconds. Auth failures do not fall back.
            "encrypted_cache": {
                "enabled": False,
                "max_stale_seconds": 0,
            },
            # When True, BSM values overwrite existing env vars.  Default
            # True because the point of using BSM is centralized rotation —
            # if .env had the final say, rotating in Bitwarden wouldn't
            # take effect until you also cleared the matching .env line.
            "override_existing": True,
            # When True, the bws binary is auto-downloaded into
            # ~/.hermes/bin/ on first use.  When False you must install
            # bws yourself and have it on PATH.
            "auto_install": True,
            # Bitwarden region / self-hosted endpoint.  Empty string
            # means use the bws CLI default (US Cloud,
            # https://vault.bitwarden.com).  Set to
            # https://vault.bitwarden.eu for EU Cloud, or your own URL
            # for self-hosted Bitwarden.  Plumbed into the bws subprocess
            # as BWS_SERVER_URL.  Prompted for during
            # `hermes secrets bitwarden setup`.
            "server_url": "",
        },
        "onepassword": {
            # Master switch.  When false, the op CLI is never invoked —
            # same as not having this section at all.
            "enabled": False,
            # Mapping of env-var name → 1Password secret reference
            # (op://vault/item/field).  Each entry is resolved with a
            # single `op read` at startup.
            "env": {},
            # Optional account shorthand / sign-in address passed as
            # `op read --account <account>`.  Empty = op's default account.
            "account": "",
            # Name of the env var holding a 1Password service-account token
            # for headless auth.  Sourced from ~/.hermes/.env (or the shell)
            # and exported to the op child as OP_SERVICE_ACCOUNT_TOKEN.
            # Leave the var unset to use an interactive/desktop op session.
            "service_account_token_env": "OP_SERVICE_ACCOUNT_TOKEN",
            # Optional absolute path to the op binary.  When set it is used
            # verbatim (PATH is not consulted) — pin this to avoid trusting
            # whatever `op` appears first on PATH.  Empty = resolve via PATH.
            "binary_path": "",
            # Seconds to cache resolved values in-process and on disk.  0
            # disables BOTH cache layers (no values are written to disk).
            "cache_ttl_seconds": 300,
            # When True (default), resolved values overwrite existing env
            # vars so rotating a secret in 1Password takes effect on next
            # start.  Flip to false to let .env / shell exports win locally.
            "override_existing": True,
        },
    },

    # =========================================================================
    # Egress credential-injection proxy (iron-proxy)
    # =========================================================================
    # When enabled, outbound traffic from remote terminal sandboxes (Docker
    # today; Modal/SSH in follow-ups) is routed through a managed iron-proxy
    # subprocess.  The sandbox sees opaque proxy tokens; iron-proxy swaps in
    # real API credentials at the egress boundary.  Compromising the sandbox
    # leaks tokens that only work behind the configured trusted proxy boundary
    # (CA private key + proxy endpoint integrity are part of that boundary).
    #
    # Configure with `hermes egress setup`.  Disabled by default — the rest of
    # Hermes works exactly as before with `enabled: false`.
    "proxy": {
        # Master switch.  When false, iron-proxy is never started, no docker
        # mounts are added, no binaries are auto-installed — feature is a
        # complete no-op.
        "enabled": False,
        # Tunnel listener port.  Sandboxes get `HTTPS_PROXY=http://<host>:<port>`.
        # 9090 is the default; collide-aware setup wizard can reassign.
        "tunnel_port": 9090,
        # Auto-download the pinned iron-proxy binary into ~/.hermes/bin/ on
        # first use.  When false, you must place `iron-proxy` on PATH yourself.
        "auto_install": True,
        # Where iron-proxy looks up the real upstream secrets at egress time.
        # "env"        — process env (default; what bitwarden integration
        #                already populates if you use it)
        # "bitwarden"  — refetch via `bws secret list` on each proxy restart;
        #                rotation in the Bitwarden web app propagates without
        #                touching .env (requires `secrets.bitwarden.enabled`).
        "credential_source": "env",
        # When true, the Docker backend refuses to start a sandbox if the
        # proxy is enabled but not running.  False = fall back to direct
        # outbound with real credentials in the sandbox (the legacy posture).
        "enforce_on_docker": True,
        # NOTE: ``fail_on_uncovered_providers`` was removed.  It gated a
        # refuse-start when Anthropic / Azure OpenAI / Gemini env vars were
        # present — those providers are now first-class swapped providers
        # via per-provider match_headers rules (x-api-key, api-key,
        # x-goog-api-key), so the fail-closed tier is empty.  A leftover
        # key in existing user configs is ignored harmlessly.
        # When credential_source is bitwarden but the BWS access token /
        # project_id is missing OR the bws fetch returns no values for
        # mapped providers, the daemon raises by default.  Set this to
        # True to opt back in to the legacy "silently fall back to host
        # env" behaviour — useful for migrations where the operator wants
        # to switch credential_source to bitwarden but hasn't fully wired
        # BWS yet.  Defaults to false (strict).
        "allow_env_fallback": False,
        # SSRF deny list applied to outbound traffic.  Omit / leave empty
        # to use the safe default: loopback, link-local (incl. cloud
        # metadata IPs at 169.254.169.254), and RFC1918.  Set to an
        # explicit ``[]`` to opt out entirely (only sensible in hermetic
        # tests that need to reach a loopback upstream).
        "upstream_deny_cidrs": None,
        # Extra allowed upstream hosts beyond the bundled defaults (which
        # cover OpenRouter, OpenAI, Anthropic, Google, xAI, Mistral, Groq,
        # Together, DeepSeek, Nous).  Wildcards (`*.foo.com`) are supported.
        "extra_allowed_hosts": [],
    },

    # Google Vertex AI provider (Gemini via the OpenAI-compatible endpoint).
    # Auth is OAuth2 (short-lived access tokens minted from a service-account
    # JSON or Application Default Credentials) — NOT a static API key. The
    # credential *path* is a secret-adjacent pointer and lives in .env
    # (VERTEX_CREDENTIALS_PATH / GOOGLE_APPLICATION_CREDENTIALS); these two
    # settings are non-secret routing config and live here. Both are bridged to
    # the VERTEX_PROJECT_ID / VERTEX_REGION env vars the adapter reads, so an
    # explicit env var still wins over config.yaml.
    "vertex": {
        # GCP project ID. Empty → use the project_id embedded in the service
        # account JSON (or ADC-resolved project).
        "project_id": "",
        # Vertex region. "global" is required for the Gemini 3.x preview models
        # (regional endpoints silently 404 them). Override to a regional value
        # (e.g. "us-central1") only if your models are pinned to a region.
        "region": "global",
    },

    # Config schema version - bump this when adding new required fields
    "_config_version": 33,
}

# Optional environment variables that enhance functionality
OPTIONAL_ENV_VARS = {
    # ── Provider (handled in provider selection, not shown in checklists) ──
    "NOUS_BASE_URL": {
        "description": "Nous Portal base URL override",
        "prompt": "Nous Portal base URL (leave empty for default)",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "OPENROUTER_API_KEY": {
        "description": "OpenRouter API key (for vision, web scraping helpers, and MoA)",
        "prompt": "OpenRouter API key",
        "url": "https://openrouter.ai/keys",
        "password": True,
        "tools": ["vision_analyze"],
        "category": "provider",
        "advanced": True,
    },
    "GOOGLE_API_KEY": {
        "description": "Google AI Studio API key (also recognized as GEMINI_API_KEY)",
        "prompt": "Google AI Studio API key",
        "url": "https://aistudio.google.com/app/apikey",
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "GEMINI_API_KEY": {
        "description": "Google AI Studio API key (alias for GOOGLE_API_KEY)",
        "prompt": "Gemini API key",
        "url": "https://aistudio.google.com/app/apikey",
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "GEMINI_BASE_URL": {
        "description": "Google AI Studio base URL override",
        "prompt": "Gemini base URL (leave empty for default)",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "VERTEX_CREDENTIALS_PATH": {
        "description": "Path to a Google Cloud service account JSON for Vertex AI (Gemini). "
                       "Vertex uses OAuth2, not a static API key — this points at the "
                       "credentials Hermes mints short-lived tokens from. Falls back to "
                       "GOOGLE_APPLICATION_CREDENTIALS, then to ADC (gcloud auth "
                       "application-default login). Set project/region under vertex: in config.yaml.",
        "prompt": "Vertex service account JSON path (leave empty to use ADC / GOOGLE_APPLICATION_CREDENTIALS)",
        "url": "https://cloud.google.com/iam/docs/keys-create-delete",
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "XAI_API_KEY": {
        "description": "xAI API key",
        "prompt": "xAI API key",
        "url": "https://console.x.ai/",
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "XAI_BASE_URL": {
        "description": "xAI base URL override",
        "prompt": "xAI base URL (leave empty for default)",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "NVIDIA_API_KEY": {
        "description": "NVIDIA NIM API key (build.nvidia.com or local NIM endpoint)",
        "prompt": "NVIDIA NIM API key",
        "url": "https://build.nvidia.com/",
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "NVIDIA_BASE_URL": {
        "description": "NVIDIA NIM base URL override (e.g. http://localhost:8000/v1 for local NIM)",
        "prompt": "NVIDIA NIM base URL (leave empty for default)",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "LM_API_KEY": {
        "description": "LM Studio bearer token for auth-enabled local servers",
        "prompt": "LM Studio API key / bearer token",
        "url": None,
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "LM_BASE_URL": {
        "description": "LM Studio base URL override",
        "prompt": "LM Studio base URL (leave empty for default)",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "GLM_API_KEY": {
        "description": "Z.AI / GLM API key (also recognized as ZAI_API_KEY / Z_AI_API_KEY)",
        "prompt": "Z.AI / GLM API key",
        "url": "https://z.ai/",
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "ZAI_API_KEY": {
        "description": "Z.AI API key (alias for GLM_API_KEY)",
        "prompt": "Z.AI API key",
        "url": "https://z.ai/",
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "Z_AI_API_KEY": {
        "description": "Z.AI API key (alias for GLM_API_KEY)",
        "prompt": "Z.AI API key",
        "url": "https://z.ai/",
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "GLM_BASE_URL": {
        "description": "Z.AI / GLM base URL override",
        "prompt": "Z.AI / GLM base URL (leave empty for default)",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "KIMI_API_KEY": {
        "description": "Kimi / Moonshot API key",
        "prompt": "Kimi API key",
        "url": "https://platform.moonshot.cn/",
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "KIMI_BASE_URL": {
        "description": "Kimi / Moonshot base URL override",
        "prompt": "Kimi base URL (leave empty for default)",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "KIMI_CN_API_KEY": {
        "description": "Kimi / Moonshot China API key",
        "prompt": "Kimi (China) API key",
        "url": "https://platform.moonshot.cn/",
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "STEPFUN_API_KEY": {
        "description": "StepFun Step Plan API key",
        "prompt": "StepFun Step Plan API key",
        "url": "https://platform.stepfun.com/",
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "STEPFUN_BASE_URL": {
        "description": "StepFun Step Plan base URL override",
        "prompt": "StepFun Step Plan base URL (leave empty for default)",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "ARCEEAI_API_KEY": {
        "description": "Arcee AI API key",
        "prompt": "Arcee AI API key",
        "url": "https://chat.arcee.ai/",
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "ARCEE_BASE_URL": {
        "description": "Arcee AI base URL override",
        "prompt": "Arcee base URL (leave empty for default)",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "GMI_API_KEY": {
        "description": "GMI Cloud API key",
        "prompt": "GMI Cloud API key",
        "url": "https://www.gmicloud.ai/",
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "GMI_BASE_URL": {
        "description": "GMI Cloud base URL override",
        "prompt": "GMI Cloud base URL (leave empty for default)",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "FIREWORKS_API_KEY": {
        "description": "Fireworks AI API key",
        "prompt": "Fireworks AI API key",
        "url": "https://app.fireworks.ai/settings/users/api-keys",
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "MINIMAX_API_KEY": {
        "description": "MiniMax API key (international)",
        "prompt": "MiniMax API key",
        "url": "https://www.minimax.io/",
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "MINIMAX_BASE_URL": {
        "description": "MiniMax base URL override",
        "prompt": "MiniMax base URL (leave empty for default)",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "MINIMAX_CN_API_KEY": {
        "description": "MiniMax API key (China endpoint)",
        "prompt": "MiniMax (China) API key",
        "url": "https://www.minimaxi.com/",
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "MINIMAX_CN_BASE_URL": {
        "description": "MiniMax (China) base URL override",
        "prompt": "MiniMax (China) base URL (leave empty for default)",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "DEEPSEEK_API_KEY": {
        "description": "DeepSeek API key for direct DeepSeek access",
        "prompt": "DeepSeek API Key",
        "url": "https://platform.deepseek.com/api_keys",
        "password": True,
        "category": "provider",
    },
    "DEEPSEEK_BASE_URL": {
        "description": "Custom DeepSeek API base URL (advanced)",
        "prompt": "DeepSeek Base URL",
        "url": "",
        "password": False,
        "category": "provider",
    },
    "DASHSCOPE_API_KEY": {
        "description": "Alibaba Cloud DashScope API key (Qwen + multi-provider models)",
        "prompt": "DashScope API Key",
        "url": "https://modelstudio.console.alibabacloud.com/",
        "password": True,
        "category": "provider",
    },
    "DASHSCOPE_BASE_URL": {
        "description": "Custom DashScope base URL (default: coding-intl OpenAI-compat endpoint)",
        "prompt": "DashScope Base URL",
        "url": "",
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "HERMES_QWEN_BASE_URL": {
        "description": "Qwen Portal base URL override (default: https://portal.qwen.ai/v1)",
        "prompt": "Qwen Portal base URL (leave empty for default)",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "OPENCODE_ZEN_API_KEY": {
        "description": "OpenCode Zen API key (pay-as-you-go access to curated models)",
        "prompt": "OpenCode Zen API key",
        "url": "https://opencode.ai/auth",
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "OPENCODE_ZEN_BASE_URL": {
        "description": "OpenCode Zen base URL override",
        "prompt": "OpenCode Zen base URL (leave empty for default)",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "OPENCODE_GO_API_KEY": {
        "description": "OpenCode Go API key ($10/month subscription for open models)",
        "prompt": "OpenCode Go API key",
        "url": "https://opencode.ai/auth",
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "OPENCODE_GO_BASE_URL": {
        "description": "OpenCode Go base URL override",
        "prompt": "OpenCode Go base URL (leave empty for default)",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "HF_TOKEN": {
        "description": "Hugging Face token for Inference Providers (20+ open models via router.huggingface.co)",
        "prompt": "Hugging Face Token",
        "url": "https://huggingface.co/settings/tokens",
        "password": True,
        "category": "provider",
    },
    "HF_BASE_URL": {
        "description": "Hugging Face Inference Providers base URL override",
        "prompt": "HF base URL (leave empty for default)",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "OLLAMA_API_KEY": {
        "description": "Ollama Cloud API key (ollama.com — cloud-hosted open models)",
        "prompt": "Ollama Cloud API key",
        "url": "https://ollama.com/settings",
        "password": True,
        "category": "provider",
        "advanced": True,
    },
    "OLLAMA_BASE_URL": {
        "description": "Ollama Cloud base URL override (default: https://ollama.com/v1)",
        "prompt": "Ollama base URL (leave empty for default)",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "XIAOMI_API_KEY": {
        "description": "Xiaomi MiMo API key for MiMo models (mimo-v2.5-pro, mimo-v2.5, mimo-v2-pro, mimo-v2-omni, mimo-v2-flash)",
        "prompt": "Xiaomi MiMo API Key",
        "url": "https://platform.xiaomimimo.com",
        "password": True,
        "category": "provider",
    },
    "XIAOMI_BASE_URL": {
        "description": "Xiaomi MiMo base URL override (default: https://api.xiaomimimo.com/v1)",
        "prompt": "Xiaomi base URL (leave empty for default)",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "UPSTAGE_API_KEY": {
        "description": "Upstage API key for Solar LLM models",
        "prompt": "Upstage API Key",
        "url": "https://console.upstage.ai/api-keys",
        "password": True,
        "category": "provider",
    },
    "UPSTAGE_BASE_URL": {
        "description": "Upstage base URL override (default: https://api.upstage.ai/v1)",
        "prompt": "Upstage base URL (leave empty for default)",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "AWS_REGION": {
        "description": "AWS region for Bedrock API calls (e.g. us-east-1, eu-central-1)",
        "prompt": "AWS Region",
        "url": "https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-regions.html",
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "AWS_PROFILE": {
        "description": "AWS named profile for Bedrock authentication (from ~/.aws/credentials)",
        "prompt": "AWS Profile",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    "AZURE_FOUNDRY_API_KEY": {
        "description": "Azure Foundry API key for custom Azure endpoints",
        "prompt": "Azure Foundry API Key",
        "url": "https://ai.azure.com/",
        "password": True,
        "category": "provider",
    },
    "AZURE_FOUNDRY_BASE_URL": {
        "description": "Azure Foundry base URL (set via 'hermes model' for endpoint-specific config)",
        "prompt": "Azure Foundry base URL",
        "url": None,
        "password": False,
        "category": "provider",
        "advanced": True,
    },
    # ── Tool API keys ──
    "EXA_API_KEY": {
        "description": "Exa API key for AI-native web search and contents",
        "prompt": "Exa API key",
        "url": "https://exa.ai/",
        "tools": ["web_search", "web_extract"],
        "password": True,
        "category": "tool",
    },
    "PARALLEL_API_KEY": {
        "description": "Parallel API key for AI-native web search and extract",
        "prompt": "Parallel API key",
        "url": "https://parallel.ai/",
        "tools": ["web_search", "web_extract"],
        "password": True,
        "category": "tool",
    },
    "FIRECRAWL_API_KEY": {
        "description": "Firecrawl API key for web search and scraping",
        "prompt": "Firecrawl API key",
        "url": "https://firecrawl.dev/",
        "tools": ["web_search", "web_extract"],
        "password": True,
        "category": "tool",
    },
    "FIRECRAWL_API_URL": {
        "description": "Firecrawl API URL for self-hosted instances (optional)",
        "prompt": "Firecrawl API URL (leave empty for cloud)",
        "url": None,
        "password": False,
        "category": "tool",
        "advanced": True,
    },
    "TAVILY_API_KEY": {
        "description": "Tavily API key for AI-native web search and extract",
        "prompt": "Tavily API key",
        "url": "https://app.tavily.com/home",
        "tools": ["web_search", "web_extract"],
        "password": True,
        "category": "tool",
    },
    "SEARXNG_URL": {
        "description": "URL of your SearXNG instance for free self-hosted web search",
        "prompt": "SearXNG URL (e.g. http://localhost:8080)",
        "url": "https://searxng.github.io/searxng/",
        "tools": ["web_search"],
        "password": False,
        "category": "tool",
    },
    "BRAVE_SEARCH_API_KEY": {
        "description": "Brave Search API subscription token (free tier: 2,000 queries/mo)",
        "prompt": "Brave Search subscription token",
        "url": "https://brave.com/search/api/",
        "tools": ["web_search"],
        "password": True,
        "category": "tool",
    },
    "BROWSERBASE_API_KEY": {
        "description": "Browserbase API key for cloud browser (optional — local browser works without this)",
        "prompt": "Browserbase API key",
        "url": "https://browserbase.com/",
        "tools": ["browser_navigate", "browser_click"],
        "password": True,
        "category": "tool",
    },
    "BROWSERBASE_PROJECT_ID": {
        "description": "Browserbase project ID (optional — only needed for cloud browser)",
        "prompt": "Browserbase project ID",
        "url": "https://browserbase.com/",
        "tools": ["browser_navigate", "browser_click"],
        "password": False,
        "category": "tool",
    },
    "BROWSER_USE_API_KEY": {
        "description": "Browser Use API key for cloud browser (optional — local browser works without this)",
        "prompt": "Browser Use API key",
        "url": "https://browser-use.com/",
        "tools": ["browser_navigate", "browser_click"],
        "password": True,
        "category": "tool",
    },
    "FIRECRAWL_BROWSER_TTL": {
        "description": "Firecrawl browser session TTL in seconds (optional, default 300)",
        "prompt": "Browser session TTL (seconds)",
        "tools": ["browser_navigate", "browser_click"],
        "password": False,
        "category": "tool",
    },
    "AGENT_BROWSER_ENGINE": {
        "description": "Browser engine for local mode: auto (default Chrome), lightpanda (faster, no screenshots), chrome",
        "prompt": "Browser engine (auto/lightpanda/chrome)",
        "url": "https://github.com/vercel-labs/agent-browser",
        "tools": ["browser_navigate", "browser_snapshot", "browser_click", "browser_vision"],
        "password": False,
        "category": "tool",
        "advanced": True,
    },
    "CAMOFOX_URL": {
        "description": "Camofox browser server URL for local anti-detection browsing (e.g. http://localhost:9377)",
        "prompt": "Camofox server URL",
        "url": "https://github.com/jo-inc/camofox-browser",
        "tools": ["browser_navigate", "browser_click"],
        "password": False,
        "category": "tool",
    },
    "CAMOFOX_API_KEY": {
        "description": "Optional bearer token sent as Authorization header to a remote/authenticated Camofox server",
        "prompt": "Camofox API key",
        "url": "https://github.com/jo-inc/camofox-browser",
        "tools": ["browser_navigate", "browser_click"],
        "password": True,
        "category": "tool",
        "advanced": True,
    },
    "FAL_KEY": {
        "description": "FAL API key for image and video generation",
        "prompt": "FAL API key",
        "url": "https://fal.ai/",
        "tools": ["image_generate", "video_generate"],
        "password": True,
        "category": "tool",
    },
    "KREA_API_KEY": {
        "description": "Krea API key for Krea 2 image generation (Medium + Large)",
        "prompt": "Krea API key",
        "url": "https://www.krea.ai/settings/api-tokens",
        "tools": ["image_generate"],
        "password": True,
        "category": "tool",
    },
    "GITHUB_TOKEN": {
        "description": "GitHub token for Skills Hub (higher API rate limits, skill publish)",
        "prompt": "GitHub Token",
        "url": "https://github.com/settings/tokens",
        "password": True,
        "category": "tool",
    },

    # ── Mem0 ──
    "MEM0_API_KEY": {
        "description": "Mem0 Platform API key for semantic persistent memory",
        "prompt": "Mem0 API key",
        "url": "https://app.mem0.ai",
        "tools": ["mem0_search"],
        "password": True,
        "category": "tool",
    },

    # ── ByteRover ──
    "BRV_API_KEY": {
        "description": "ByteRover API key (optional, for cloud sync — local-first by default)",
        "prompt": "ByteRover API key",
        "url": "https://app.byterover.dev",
        "tools": ["brv_query"],
        "password": True,
        "category": "tool",
    },

    # ── Agent settings ──
    # NOTE: MESSAGING_CWD was removed here — use terminal.cwd in config.yaml
    # instead.  The gateway reads TERMINAL_CWD (bridged from terminal.cwd).
    "SUDO_PASSWORD": {
        "description": "Sudo password for terminal commands requiring root access; set to an explicit empty string to try empty without prompting",
        "prompt": "Sudo password",
        "url": None,
        "password": True,
        "category": "setting",
    },
    # HERMES_TOOL_PROGRESS_MODE is deprecated — tool progress is configured via
    # display.tool_progress in config.yaml (off|new|all|verbose|log). The
    # gateway still falls back to HERMES_TOOL_PROGRESS_MODE for backward
    # compatibility, so it lives in _EXTRA_ENV_KEYS (known to reload and
    # compatibility paths) but is intentionally NOT listed here:
    # OPTIONAL_ENV_VARS feeds user-facing surfaces (dashboard keys page, setup
    # checklists) and deprecated knobs shouldn't be offered there. The boolean
    # HERMES_TOOL_PROGRESS is fully unsupported since the v12 config support
    # floor retired its only consumer (the v3→4 migration).
    "HERMES_PREFILL_MESSAGES_FILE": {
        "description": "Path to JSON file with ephemeral prefill messages for few-shot priming",
        "prompt": "Prefill messages file path",
        "url": None,
        "password": False,
        "category": "setting",
    },
    "HERMES_EPHEMERAL_SYSTEM_PROMPT": {
        "description": "Ephemeral system prompt injected at API-call time (never persisted to sessions)",
        "prompt": "Ephemeral system prompt",
        "url": None,
        "password": False,
        "category": "setting",
    },
}
