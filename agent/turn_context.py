"""Per-turn setup for ``run_conversation`` (the turn prologue).

``run_conversation`` opened with ~470 lines of straight-line setup before the
tool-calling loop ever started: stdio guarding, runtime-main wiring, retry-counter
resets, user-message sanitization, todo/nudge-counter hydration, system-prompt
restore-or-build, session-row creation (before compression, whose DB writes
reference the row), preflight context compression, the ``pre_llm_call`` plugin
hook, external-memory prefetch, and crash-resilience persistence (last, so the
user row is written once with its final ``api_content`` sidecar).

All of that is *prologue* — it runs once per turn, has no back-references into the
loop, and produces a fixed set of values the loop then consumes. ``TurnContext``
captures those produced values; ``build_turn_context`` performs the setup work and
returns one. ``run_conversation`` is left to unpack the context and run the loop,
shrinking the orchestrator by the full prologue.

The builder still mutates ``agent`` heavily (counters, turn state, cached prompt,
session DB) exactly as the inline code did — those side effects are the point. The
``TurnContext`` it returns carries only the *locals* the loop reads back.

Behavior is identical to the original inline prologue; this is a pure
move-and-name refactor with no semantic change.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from agent.conversation_compression import (
    IDLE_COMPACTION_STATUS_TEMPLATE,
    PREFLIGHT_COMPRESSION_STATUS_TEMPLATE,
    compression_skipped_due_to_lock,
    conversation_history_after_compression,
    _hydrate_persisted_compression_guards,
    recover_rotated_compression_session,
)
from agent.context_engine import automatic_compaction_status_message
from agent.iteration_budget import IterationBudget
from agent.memory_manager import build_memory_context_block
from agent.memory_provider import is_trivial_prompt
from agent.model_metadata import (
    estimate_messages_tokens_rough,
    estimate_request_tokens_rough,
)
from hermes_cli import plugins as _plugins
from tools import hook_output_spill as _hook_output_spill
from tools import mcp_tool as _mcp_tool

logger = logging.getLogger(__name__)


async def _await_engine_result(value: Any) -> Any:
    """Await native-async engines while retaining pure synchronous plugins."""
    if inspect.isawaitable(value):
        return await value
    return value


def compose_user_api_content(
    content: Any,
    ext_prefetch_cache: str,
    plugin_user_context: str,
) -> Optional[str]:
    """Compose the API-bound content of the current turn's user message.

    Sources: memory-manager prefetch + ``pre_llm_call`` plugin context with
    target="user_message" (the default). Both are appended to the *API copy*
    of the user message only — the stored content stays clean.

    This is the single source of that composition. The prologue stamps the
    result onto the live message as ``api_content`` (persisted alongside the
    clean content) and the ``api_messages`` build in ``conversation_loop``
    sends the same helper's output, so the persisted sidecar can never drift
    from the bytes on the wire — which is the whole prompt-cache invariant:
    what turn N sends must be what turn N+1 replays.

    Returns ``None`` when nothing is injected (multimodal/non-string content,
    or no ephemeral context), meaning the message is sent as-is.
    """
    if not isinstance(content, str):
        return None
    injections = []
    if ext_prefetch_cache:
        fenced = build_memory_context_block(ext_prefetch_cache)
        if fenced:
            injections.append(fenced)
    if plugin_user_context:
        injections.append(plugin_user_context)
    if not injections:
        return None
    return content + "\n\n" + "\n\n".join(injections)


def substitute_api_content(api_msg: Dict[str, Any]) -> Optional[str]:
    """Pop the ``api_content`` sidecar and substitute it into ``content``.

    Used at every API-bound message-build site (the ``api_messages`` build in
    ``conversation_loop``, the max-iterations summary in
    ``chat_completion_helpers``, the chat-completions transport). The sidecar
    carries the exact bytes previously sent to the API for this message when
    they differ from the clean stored content; substituting it here keeps the
    provider prompt-cache prefix byte-stable across turns.

    Returns the popped sidecar string (for callers that need the value for
    current-turn composition logic) or ``None`` when absent.
    """
    sidecar = api_msg.pop("api_content", None)
    if (
        isinstance(sidecar, str)
        and sidecar
        and api_msg.get("role") in ("user", "assistant")
    ):
        api_msg["content"] = sidecar
    return sidecar


def drop_stale_api_content(msg: Dict[str, Any]) -> None:
    """Drop the ``api_content`` sidecar from a message whose content was rewritten.

    Called from every content-rewrite path (historical image strip,
    merge-summary-into-tail, consecutive-user repair merge, stale-confirmation
    redaction). Replaying the pre-rewrite sidecar would resend exactly what
    the rewrite removed, so it must be dropped — the cost is one cache
    boundary miss, never wrong content.
    """
    msg.pop("api_content", None)


def reanchor_current_turn_user_idx(messages: List[Any], user_message: Any) -> int:
    """Locate this turn's user message after compaction rebuilt ``messages``.

    Compression replaces list entries with fresh copies (and may append a
    todo-snapshot user message or a restored user turn AFTER the surviving
    copy of the current turn's message), so a pre-compression index is
    meaningless. Prefer the LAST user message whose content exactly matches
    this turn's text — the surviving copy in the common case — so the
    injection stamp and the #48677 persist override can't land on a
    todo-snapshot or historical row. Fall back to the last user message when
    no exact match survives (merge-summary-into-tail rewrites the content but
    the trackers still need a live anchor). Returns -1 when the list has no
    user message at all.
    """
    fallback = -1
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if not (isinstance(msg, dict) and msg.get("role") == "user"):
            continue
        if fallback < 0:
            fallback = i
        if msg.get("content") == user_message:
            return i
    return fallback


def _compression_made_progress(
    orig_len: int, new_len: int, orig_tokens: int, new_tokens: int
) -> bool:
    """Return ``True`` if a compression pass materially reduced the request.

    Compression can succeed by summarising message contents — reducing the
    estimated request token count — without reducing the message row
    count.  Treating row count as the sole progress signal false-positives
    on size-only wins and surfaces a misleading "Cannot compress further"
    failure even when post-compression tokens are well below the model
    context window.  See issue #39548 for an observed case: 220 → 220
    messages, ~288k → ~183k tokens on a 1M-context model still triggered
    auto-reset.

    The token reduction must be *material* (>5%) to count as progress — the
    same floor the overflow-handler retry path uses (conversation_loop.py,
    #39550) — so a sub-5% wobble doesn't keep the multi-pass loop spinning.
    """
    if new_len < orig_len:
        return True
    return orig_tokens > 0 and new_tokens < orig_tokens * 0.95


def _compression_warrants_another_preflight_pass(
    orig_tokens: int, new_tokens: int, threshold_tokens: int
) -> bool:
    """Whether an over-threshold request merits another immediate summary.

    Row-count progress is enough to prove that a compression boundary was real,
    but not enough to justify another expensive pass before trying the provider.
    Continue only when the request remains over threshold *and* the previous pass
    materially reduced its estimated token pressure (>5%).
    """
    return (
        new_tokens >= threshold_tokens
        and orig_tokens > 0
        and new_tokens < orig_tokens * 0.95
    )


def _should_run_preflight_estimate(
    messages: List[Dict[str, Any]],
    protect_first_n: int,
    protect_last_n: int,
    threshold_tokens: int,
) -> bool:
    """Cheap gate for the (expensive) full preflight token estimate.

    Returns ``True`` when either:
      (a) message count exceeds the protected ranges (the historical gate), or
      (b) a cheap char-based estimate already crosses the configured threshold
          — the few-but-huge case from issue #27405 that the count-only gate
          would silently skip (a handful of very large messages never trips
          the count condition, so compression was never attempted and the
          turn hit a hard context-overflow error).

    Branch (b) uses ``estimate_messages_tokens_rough`` (the shared char-based
    estimator) so a single large base64 image isn't mistaken for ~250K tokens.
    It intentionally undercounts vs. the full request estimate — it omits the
    system prompt and tool schemas — because it is only a *hint* deciding
    whether to pay for the authoritative ``estimate_request_tokens_rough``,
    which (together with ``should_compress``) makes the real decision.
    """
    if len(messages) > protect_first_n + protect_last_n + 1:
        return True
    return estimate_messages_tokens_rough(messages) >= threshold_tokens


def _should_idle_compact(
    *,
    enabled: bool,
    idle_after_seconds: int,
    idle_gap_seconds: float,
    tokens: int,
    floor_tokens: int,
    cooldown_active: bool,
) -> bool:
    """Decide whether an idle-triggered compaction should run this turn.

    Idle compaction is opt-in (``idle_after_seconds <= 0`` disables it). It
    fires when a session resumes after a wall-clock gap of at least
    ``idle_after_seconds`` since its last activity, so a long-lived thread
    that is paused and later resumed compacts its accumulated history up
    front instead of re-reading it on every subsequent turn.

    It is orthogonal to the token-threshold trigger: it does NOT require the
    context to exceed ``threshold_tokens``. It still skips work when the
    context is at or below ``floor_tokens`` (the size compaction would reduce
    *to*), so a small idle thread never pays for a summarisation that saves
    nothing, and it defers to an active compression-failure cooldown.

    Pure predicate so the policy is unit-testable without a live agent.
    """
    if not enabled or idle_after_seconds <= 0:
        return False
    if idle_gap_seconds < idle_after_seconds:
        return False
    if cooldown_active:
        return False
    return tokens > floor_tokens


@dataclass
class TurnContext:
    """Values produced by the turn prologue and consumed by the turn loop."""

    # Sanitized inbound message (surrogates stripped).
    user_message: str
    # Clean message preserved for transcripts / memory queries (no nudge injection).
    original_user_message: Any
    # Working message list for this turn (loop appends to it).
    messages: List[Dict[str, Any]]
    # May be reset to None by preflight compression (new session created).
    conversation_history: Optional[List[Dict[str, Any]]]
    # Cached system prompt active for this turn (may be rebuilt by compression).
    active_system_prompt: Optional[str]
    # Task / turn identifiers.
    effective_task_id: str
    turn_id: str
    # Index of the current user turn within ``messages``.
    current_turn_user_idx: int
    # Whether the post-turn memory review should fire.
    should_review_memory: bool = False
    # Context contributed by ``pre_llm_call`` plugins (appended to user message).
    plugin_user_context: str = ""
    # External-memory prefetch result, reused across loop iterations.
    ext_prefetch_cache: str = ""
    # Turn-start preflight already proved an immediate retry ineffective.
    preflight_compression_blocked: bool = False


async def build_turn_context(
    agent,
    user_message: Any,
    system_message: Optional[str],
    conversation_history: Optional[List[Dict[str, Any]]],
    task_id: Optional[str],
    stream_callback,
    persist_user_message: Optional[Any],
    persist_user_timestamp: Optional[float] = None,
    *,
    persist_user_display_kind: Optional[str] = None,
    persist_user_display_metadata: Optional[Dict[str, Any]] = None,
    restore_or_build_system_prompt,
    install_safe_stdio,
    sanitize_surrogates,
    summarize_user_message_for_log,
    set_session_context,
    set_current_write_origin,
    ra,
    moa_active: bool = False,
) -> TurnContext:
    """Run the once-per-turn setup and return the loop's input context.

    The callables/helpers the original prologue referenced from the
    ``conversation_loop`` module are passed in explicitly to keep this module
    free of an import cycle with ``agent.conversation_loop``.
    """
    # Guard stdio against OSError from broken pipes (systemd/headless/daemon).
    install_safe_stdio()

    # Recover a session rotated by another path before binding log/turn ids or
    # copying client-supplied history. Everything in this turn must consistently
    # belong to the canonical child, including observability metadata.
    recovered_history = await recover_rotated_compression_session(agent)
    if recovered_history is not None:
        conversation_history = recovered_history

    # NOTE: the DB session row is created later, AFTER the system prompt is
    # restored/built (see _ensure_db_session() below the system-prompt block).
    # Creating it here — before _cached_system_prompt is populated — inserts a
    # row with system_prompt=NULL on a fresh API/gateway agent that carries
    # client-managed history, which then trips the "stored system prompt is
    # null; rebuilding from scratch" warning and a needless first-turn prefix
    # cache miss. (Issue #45499.)

    # Bind task-local session identity before any provider or tool work.
    from gateway.session_context import set_current_session_id

    set_current_session_id(agent.session_id)

    # Tag log records in this context with the session ID for logging.
    set_session_context(agent.session_id)

    # Bind the skill write-origin ContextVar for this task.
    set_current_write_origin(getattr(agent, "_memory_write_origin", "assistant_tool"))

    # Resolve credentials lazily.  The constructor deliberately keeps auth
    # discovery state-only so auth.json/OAuth I/O cannot block application
    # startup or the event loop.
    ensure_provider_runtime = getattr(agent, "_ensure_provider_runtime", None)
    if ensure_provider_runtime is not None:
        await ensure_provider_runtime()

    # Restore the primary runtime if the previous turn activated fallback.
    await agent._restore_primary_runtime()

    # Tell auxiliary_client what the live main provider/model are for this turn
    # after primary restoration has settled the runtime.
    try:
        from agent.auxiliary_client import set_runtime_main
        set_runtime_main(
            getattr(agent, "provider", "") or "",
            getattr(agent, "model", "") or "",
            requested_provider=getattr(agent, "requested_provider", "") or "",
            base_url=getattr(agent, "base_url", "") or "",
            api_key=getattr(agent, "api_key", "") or "",
            api_mode=getattr(agent, "api_mode", "") or "",
            auth_mode=getattr(agent, "auth_mode", "") or "",
        )
    except Exception:
        pass

    # Between-turns MCP discovery/refresh: a server that finished connecting since
    # the previous turn (slow HTTP/OAuth servers routinely take 2-6s on a cold
    # connect, missing the bounded startup wait) lands in THIS turn's tool
    # snapshot.  This is cache-safe by construction: it runs in the per-turn
    # prologue, before this turn's first API call assembles ``tools=``, so it
    # only ever extends a fresh request prefix — it never mutates the cached
    # prefix of an in-flight turn.  No-op when no MCP servers are registered
    # (the common case, gated by the cheap ``has_registered_mcp_tools`` check)
    # or when the tool set is unchanged (``refresh_agent_mcp_tools`` diffs by
    # name and leaves the snapshot untouched on no-change).
    initial_tool_snapshot = not getattr(
        agent, "_tool_snapshot_initialized", False
    )
    initial_lsp_lease = not getattr(agent, "_lsp_lifecycle_retained", False)
    initial_auxiliary_lease = not getattr(
        agent,
        "_auxiliary_lifecycle_retained",
        False,
    )
    initial_parallel_lease = not getattr(
        agent,
        "_parallel_lifecycle_retained",
        False,
    )
    try:
        if initial_lsp_lease:
            from agent.lsp import _retain_lsp_lifecycle

            await _retain_lsp_lifecycle(agent)
            agent._lsp_lifecycle_retained = True
        if initial_auxiliary_lease:
            from agent.auxiliary_client import _retain_auxiliary_lifecycle

            await _retain_auxiliary_lifecycle(agent)
            agent._auxiliary_lifecycle_retained = True
        if initial_parallel_lease:
            from plugins.web.parallel.provider import _retain_parallel_lifecycle

            await _retain_parallel_lifecycle(agent)
            agent._parallel_lifecycle_retained = True
        await _plugins.discover_plugins()
        if not getattr(agent, "_skip_mcp_refresh", False):
            # The first discovery is intentionally lazy: ``AIAgent.__init__``
            # must not open an MCP transport or synchronously wait on a remote
            # server.  Subsequent turns only refresh the in-memory snapshot.
            if not getattr(agent, "_mcp_discovery_started", False):
                agent._mcp_discovery_started = True
                await _mcp_tool._retain_mcp_lifecycle(agent)
                agent._mcp_lifecycle_retained = True
                await _mcp_tool.discover_mcp_tools()
            if initial_tool_snapshot or _mcp_tool.has_registered_mcp_tools():
                await _mcp_tool.refresh_agent_mcp_tools(
                    agent,
                    quiet_mode=(
                        True
                        if not initial_tool_snapshot
                        else getattr(agent, "quiet_mode", False)
                    ),
                )
                agent._tool_snapshot_initialized = True
    except asyncio.CancelledError:
        if initial_lsp_lease and getattr(
            agent, "_lsp_lifecycle_retained", False
        ):
            from agent.lsp import _release_lsp_lifecycle

            await _release_lsp_lifecycle(agent)  # noqa: ASYNC120
            agent._lsp_lifecycle_retained = False
        if initial_auxiliary_lease and getattr(
            agent,
            "_auxiliary_lifecycle_retained",
            False,
        ):
            from agent.auxiliary_client import _release_auxiliary_lifecycle

            await _release_auxiliary_lifecycle(agent)  # noqa: ASYNC120
            agent._auxiliary_lifecycle_retained = False
        if initial_parallel_lease and getattr(
            agent,
            "_parallel_lifecycle_retained",
            False,
        ):
            from plugins.web.parallel.provider import _release_parallel_lifecycle

            await _release_parallel_lifecycle(agent)  # noqa: ASYNC120
            agent._parallel_lifecycle_retained = False
        if initial_tool_snapshot:
            if getattr(agent, "_mcp_lifecycle_retained", False):
                await _mcp_tool._release_mcp_lifecycle(agent)  # noqa: ASYNC120
                agent._mcp_lifecycle_retained = False
            agent._mcp_discovery_started = False
        raise
    except Exception:
        if initial_lsp_lease and getattr(
            agent, "_lsp_lifecycle_retained", False
        ):
            from agent.lsp import _release_lsp_lifecycle

            await _release_lsp_lifecycle(agent)  # noqa: ASYNC120
            agent._lsp_lifecycle_retained = False
        if initial_auxiliary_lease and getattr(
            agent,
            "_auxiliary_lifecycle_retained",
            False,
        ):
            from agent.auxiliary_client import _release_auxiliary_lifecycle

            await _release_auxiliary_lifecycle(agent)  # noqa: ASYNC120
            agent._auxiliary_lifecycle_retained = False
        if initial_parallel_lease and getattr(
            agent,
            "_parallel_lifecycle_retained",
            False,
        ):
            from plugins.web.parallel.provider import _release_parallel_lifecycle

            await _release_parallel_lifecycle(agent)  # noqa: ASYNC120
            agent._parallel_lifecycle_retained = False
        if initial_tool_snapshot:
            if getattr(agent, "_mcp_lifecycle_retained", False):
                await _mcp_tool._release_mcp_lifecycle(agent)  # noqa: ASYNC120
                agent._mcp_lifecycle_retained = False
            agent._mcp_discovery_started = False
            raise
        logger.debug("between-turns MCP tool refresh skipped", exc_info=True)

    # Sanitize surrogate characters from user input.
    if isinstance(user_message, str):
        user_message = sanitize_surrogates(user_message)
    if isinstance(persist_user_message, str):
        persist_user_message = sanitize_surrogates(persist_user_message)

    # Store the stream callback for the native model transport.
    agent._stream_callback = stream_callback
    agent._persist_user_message_idx = None
    agent._persist_user_message_override = persist_user_message
    agent._persist_user_message_timestamp = persist_user_timestamp
    # Generate unique task_id if not provided to isolate VMs between tasks.
    effective_task_id = task_id or str(uuid.uuid4())
    agent._current_task_id = effective_task_id
    task_ids = getattr(agent, "_task_ids", None)
    if task_ids is None:
        task_ids = set()
        agent._task_ids = task_ids
    task_ids.add(effective_task_id)
    turn_id = (
        f"{agent.session_id or 'session'}:{effective_task_id}:{uuid.uuid4().hex[:8]}"
    )
    agent._current_turn_id = turn_id
    agent._current_api_request_id = ""
    # Tripwire: warn (with both turn ids) when this turn starts before the
    # previous turn's turn-end persist — concurrent turns on one session
    # interleave transcript writes. Cleared in _persist_session.
    from agent.agent_runtime_helpers import note_turn_start
    note_turn_start(agent, turn_id)
    # This diagnostic/accounting field is turn-scoped.  Reset it at the same
    # boundary as the in-flight marker so a cancellation later in the
    # prologue cannot report the previous turn's provider-call count.
    agent._api_call_count = 0

    # Reset retry counters and iteration budget at the start of each turn.
    agent._invalid_tool_retries = 0
    agent._invalid_json_retries = 0
    agent._empty_content_retries = 0
    agent._incomplete_scratchpad_retries = 0
    agent._codex_incomplete_retries = 0
    agent._thinking_prefill_retries = 0
    agent._post_tool_empty_retried = False
    agent._last_content_with_tools = None
    agent._last_content_tools_all_housekeeping = False
    agent._mute_post_response = False
    agent._unicode_sanitization_passes = 0
    agent._tool_guardrails.reset_for_turn()
    agent._tool_guardrail_halt_decision = None
    _reset_consol = getattr(agent._memory_store, "reset_consolidation_failures", None)
    if callable(_reset_consol):
        _reset_consol()
    if agent._memory_store is not None and not getattr(agent, "_memory_loaded", False):
        load_memory = getattr(agent._memory_store, "load_from_disk", None)
        if not callable(load_memory):
            raise RuntimeError(
                "Async AIAgent requires a memory store with async load_from_disk()."
            )
        await load_memory()
        agent._memory_loaded = True
    agent._vision_supported = True

    # Native HTTP clients own their connection health asynchronously.  The
    # former raw-socket probe traversed the synchronous client's pool from the
    # event loop, so it is intentionally absent from the async turn path.
    # Replay compression warning through status_callback for gateway platforms.
    if agent._compression_warning:
        agent._replay_compression_warning()
        agent._compression_warning = None  # send once

    # NOTE: _turns_since_memory is intentionally not reset here.
    agent.iteration_budget = IterationBudget(agent.max_iterations)

    # Log conversation turn start for debugging/observability.
    _preview_text = summarize_user_message_for_log(user_message)
    _msg_preview = (_preview_text[:80] + "...") if len(_preview_text) > 80 else _preview_text
    _msg_preview = _msg_preview.replace("\n", " ")
    logger.info(
        "conversation turn: session=%s model=%s provider=%s platform=%s history=%d msg=%r",
        agent.session_id or "none", agent.model, agent.provider or "unknown",
        agent.platform or "unknown", len(conversation_history or []),
        _msg_preview,
    )

    # Initialize conversation (copy to avoid mutating the caller's list).
    messages = list(conversation_history) if conversation_history else []

    # The CLI may already have staged this input outside the history passed to
    # ``run_conversation``. Reuse it only when its clean transcript text matches
    # this turn; a stale handoff from a failed prior turn must not replace a
    # later, different user input. Voice turns compare against their explicit
    # clean persistence override rather than the API-only prefixed payload.
    pending_cli_message = getattr(agent, "_pending_cli_user_message", None)
    expected_persist_content = (
        persist_user_message if persist_user_message is not None else user_message
    )
    if (
        isinstance(pending_cli_message, dict)
        and pending_cli_message.get("content") == expected_persist_content
    ):
        user_msg = pending_cli_message
        # The CLI-staged value is the clean transcript text. Restore the
        # API-facing variant (for example, a voice-mode prefix) while retaining
        # the same dict and any close-path durable marker.
        user_msg["content"] = user_message
    else:
        user_msg = {"role": "user", "content": user_message}
        if isinstance(pending_cli_message, dict):
            agent._pending_cli_user_message = None

    # Hydrate todo store from conversation history.
    if conversation_history and not agent._todo_store.has_items():
        agent._hydrate_todo_store(conversation_history)

    # Hydrate per-session nudge counters from persisted history (issue #22357).
    if conversation_history and agent._user_turn_count == 0:
        prior_user_turns = sum(
            1 for m in conversation_history if m.get("role") == "user"
        )
        if prior_user_turns > 0:
            agent._user_turn_count = prior_user_turns
            if agent._memory_nudge_interval > 0 and agent._turns_since_memory == 0:
                agent._turns_since_memory = prior_user_turns % agent._memory_nudge_interval

    # Add the current user message after the prompt/session setup has made
    # close persistence safe. The handoff above preserves any marker already
    # stamped by an earlier close flush.
    #
    # A synthesized turn (auto-continue recovery note, delegation completion)
    # declares how it should READ in a transcript. Stamp that on the live
    # message so the crash persist below writes the row already typed. Typing
    # it after the turn instead leaves the row untyped for the whole run — and
    # forever if the turn crashes — so the raw system note paints as a user
    # bubble. The model still receives role/content unchanged; the api_messages
    # build strips both fields from every outgoing copy.
    if persist_user_display_kind:
        user_msg["display_kind"] = persist_user_display_kind
        if persist_user_display_metadata:
            user_msg["display_metadata"] = persist_user_display_metadata

    messages.append(user_msg)
    current_turn_user_idx = len(messages) - 1
    agent._persist_user_message_idx = current_turn_user_idx
    # Publish the live list as soon as the current user row exists.  If the
    # task is cancelled later in the prologue (prompt construction, memory
    # prefetch, plugin hooks, or the crash-safe flush), the outer cancellation
    # finalizer can persist this exact partially prepared transcript instead
    # of falling back to stale messages from the previous turn.
    agent._session_messages = messages

    # Track user turns for memory flush and periodic nudge logic.
    agent._user_turn_count += 1
    # Copilot x-initiator: the first API call of this user turn is
    # user-initiated; tool-loop follow-ups revert to "agent" (#3040).
    agent._is_user_initiated_turn = True

    # Reset the streaming context scrubber at the top of each turn.
    scrubber = getattr(agent, "_stream_context_scrubber", None)
    if scrubber is not None:
        scrubber.reset()
    # Reset the think scrubber for the same reason.
    think_scrubber = getattr(agent, "_stream_think_scrubber", None)
    if think_scrubber is not None:
        think_scrubber.reset()

    # Preserve the original user message (no nudge injection).
    original_user_message = persist_user_message if persist_user_message is not None else user_message

    # Track memory nudge trigger (turn-based, checked here).
    should_review_memory = False
    if (agent._memory_nudge_interval > 0
            and "memory" in agent.valid_tool_names
            and agent._memory_store):
        agent._turns_since_memory += 1
        if agent._turns_since_memory >= agent._memory_nudge_interval:
            should_review_memory = True
            agent._turns_since_memory = 0

    if not agent.quiet_mode:
        _print_preview = summarize_user_message_for_log(user_message)
        agent._safe_print(
            f"💬 Starting conversation: '{_print_preview[:60]}"
            f"{'...' if len(_print_preview) > 60 else ''}'"
        )

    # ── System prompt (cached per session for prefix caching) ──
    if agent._cached_system_prompt is None:
        await restore_or_build_system_prompt(
            agent,
            system_message,
            conversation_history,
        )

    active_system_prompt = agent._cached_system_prompt

    # Create the DB session row now that _cached_system_prompt is populated, so
    # the persisted snapshot is written non-NULL on the first turn (Issue
    # #45499). Idempotent: _ensure_db_session() no-ops once the row exists.
    # Must run BEFORE preflight compression: in-place compaction inserts
    # message rows referencing this session (archive_and_compact), and
    # rotation creates a child with parent_session_id pointing at it — with
    # PRAGMA foreign_keys=ON, a missing parent row fails both INSERTs on a
    # fresh oversized first turn. The user-turn crash persist itself runs
    # LATER (after memory prefetch / pre_llm_call), so the row is written
    # once with its final api_content — both steps take the same per-agent
    # persist lock as CLI close persistence.
    persist_lock = agent._get_session_persist_lock()
    try:
        async with persist_lock:
            await agent._ensure_db_session()
            await agent._persist_pending_billing_route()
            if getattr(agent, "_session_db", None) is not None:
                await _hydrate_persisted_compression_guards(
                    agent.context_compressor,
                    agent._session_db,
                    agent.session_id,
                )
    except Exception:
        logger.warning(
            "Turn-start session row creation failed for session=%s",
            agent.session_id or "none",
            exc_info=True,
        )
    finally:
        # Clear the staged CLI input eagerly (as the pre-refactor code did)
        # so a crash in preflight compression — which runs between this row
        # create and the late crash-persist below — doesn't leave a stale
        # _pending_cli_user_message that the next turn would mistake for a
        # fresh staged input.
        if not isinstance(pending_cli_message, dict) or pending_cli_message.get("_db_persisted"):
            agent._pending_cli_user_message = None

    # ── Idle-triggered compaction (opt-in; ``idle_compact_after_seconds``) ──
    # When a session resumes after a long idle gap, compact the accumulated
    # history up front so the rest of the conversation does not keep re-reading
    # a large stale context on every turn. This fires on elapsed wall-clock time
    # rather than size, so it complements (does not replace) the token-threshold
    # preflight below. ``_last_activity_ts`` is the last time this turn loop did
    # work; nothing has touched it yet this turn, so it measures the gap since
    # the previous turn finished. The cheap gap pre-check gates the (more
    # expensive) token estimate, mirroring ``_should_run_preflight_estimate``.
    _idle_after = getattr(agent, "compression_idle_compact_after_seconds", 0)
    if agent.compression_enabled and _idle_after > 0 and messages:
        _idle_gap = time.time() - getattr(agent, "_last_activity_ts", time.time())
        if _idle_gap >= _idle_after:
            _compressor = agent.context_compressor
            _idle_tokens = estimate_request_tokens_rough(
                messages,
                system_prompt=active_system_prompt or "",
                tools=agent.tools or None,
            )
            # Post-compression target size: don't summarise a thread already
            # below what compaction would reduce it to.
            _idle_floor = int(
                _compressor.threshold_tokens * _compressor.summary_target_ratio
            )
            _idle_cooldown = await _await_engine_result(
                getattr(
                    _compressor,
                    "get_active_compression_failure_cooldown",
                    lambda: None,
                )()
            )
            if _should_idle_compact(
                enabled=agent.compression_enabled,
                idle_after_seconds=_idle_after,
                idle_gap_seconds=_idle_gap,
                tokens=_idle_tokens,
                floor_tokens=_idle_floor,
                cooldown_active=bool(_idle_cooldown),
            ):
                logger.info(
                    "Idle compaction: %ss idle >= %ss, ~%s tokens > %s floor "
                    "(session %s)",
                    int(_idle_gap),
                    _idle_after,
                    f"{_idle_tokens:,}",
                    f"{_idle_floor:,}",
                    agent.session_id or "none",
                )
                _idle_status = automatic_compaction_status_message(
                    _compressor,
                    phase="idle",
                    default_message=IDLE_COMPACTION_STATUS_TEMPLATE.format(
                        idle_seconds=int(_idle_gap), tokens=_idle_tokens
                    ),
                    approx_tokens=_idle_tokens,
                    idle_seconds=int(_idle_gap),
                    model=agent.model,
                )
                if _idle_status:
                    agent._emit_status(_idle_status)
                _idle_input = messages
                messages, active_system_prompt = await agent._compress_context(
                    messages, system_message, approx_tokens=_idle_tokens,
                    task_id=effective_task_id,
                )
                # ``_compress_context`` returns the INPUT list object when it
                # skips (per-session lock held by another path, failure
                # cooldown, anti-thrash breaker, codex-native routing). Only
                # re-baseline + re-anchor after a real compaction — a skip
                # must leave the turn's flush baseline and user-message index
                # untouched.
                if messages is not _idle_input:
                    conversation_history = conversation_history_after_compression(
                        agent, messages, conversation_history
                    )
                    # Compaction rebuilt the list, so the index of this turn's
                    # just-appended user message is stale — re-anchor it the
                    # same way the preflight path does below.
                    current_turn_user_idx = reanchor_current_turn_user_idx(
                        messages, user_message
                    )
                    agent._persist_user_message_idx = current_turn_user_idx

    # ── Preflight context compression ──
    # Gate the (expensive) full token estimate behind a cheap pre-check.
    # See ``_should_run_preflight_estimate`` for the OR semantics that fix
    # issue #27405 (a few very large messages slipping past the count gate).
    _preflight_compressed = False
    _preflight_compression_blocked = False
    agent._turn_received_provider_response = False
    agent._turn_preflight_display_snapshot = None
    if agent.compression_enabled and _should_run_preflight_estimate(
        messages,
        agent.context_compressor.protect_first_n,
        agent.context_compressor.protect_last_n,
        agent.context_compressor.threshold_tokens,
    ):
        _preflight_tokens = estimate_request_tokens_rough(
            messages,
            system_prompt=active_system_prompt or "",
            tools=agent.tools or None,
        )
        _compressor = agent.context_compressor
        # getattr guard: minimal compressor doubles (SimpleNamespace in the
        # engine-preflight tests) and plugin context engines lack this
        # ContextCompressor-only method — absence means no snapshot, and the
        # finalizer's rollback stays disarmed for the turn (display-only).
        _snapshot_fn = getattr(
            _compressor, "snapshot_preflight_display_tokens", None
        )
        if callable(_snapshot_fn):
            _snapshot_val = _snapshot_fn()
            # Type pin: MagicMock compressors return truthy Mock objects —
            # only a real int snapshot may arm the interrupted-turn rollback.
            if isinstance(_snapshot_val, int) and not isinstance(
                _snapshot_val, bool
            ):
                agent._turn_preflight_display_snapshot = _snapshot_val
        _defer_preflight = getattr(
            _compressor,
            "should_defer_preflight_to_real_usage",
            lambda _tokens: False,
        )
        _preflight_deferred = _defer_preflight(_preflight_tokens)
        if not _preflight_deferred:
            _last = _compressor.last_prompt_tokens
            # Do NOT overwrite the -1 sentinel (#36718).
            if _last >= 0 and _preflight_tokens > _last:
                _compressor.last_prompt_tokens = _preflight_tokens

        _compression_cooldown = await _await_engine_result(
            getattr(
                _compressor,
                "get_active_compression_failure_cooldown",
                lambda: None,
            )()
        )

        _should_compress_now = False
        _compress_block_reason = None
        if _preflight_deferred:
            logger.info(
                "Skipping preflight compression: rough estimate ~%s >= %s, "
                "but last real provider prompt was %s after compression",
                f"{_preflight_tokens:,}",
                f"{_compressor.threshold_tokens:,}",
                f"{_compressor.last_real_prompt_tokens:,}",
            )
        elif _compression_cooldown:
            logger.info(
                "Skipping preflight compression: same-session cooldown active "
                "(~%s seconds remaining, session %s)",
                int(_compression_cooldown.get("remaining_seconds", 0.0)),
                agent.session_id or "none",
            )
            if _preflight_tokens >= _compressor.threshold_tokens:
                # Context is over threshold but compression is blocked by the
                # summary-LLM cooldown — surface a warning (see block below).
                _cooldown_secs = _compression_cooldown.get("remaining_seconds", 0.0)
                _compress_block_reason = f"cooldown:{_cooldown_secs:.0f}"
        else:
            _should_compress_now = await _await_engine_result(
                _compressor.should_compress(_preflight_tokens)
            )
            if not _should_compress_now:
                # Context is over threshold but compression is blocked
                # (summary-LLM cooldown or anti-thrashing). Ask should_compress_info
                # for the human-readable reason so we can surface a warning below.
                # getattr guard: minimal compressor doubles (SimpleNamespace in
                # the engine-preflight tests) and older plugin engines lack the
                # method — absence means no block reason, no warning.
                _info = getattr(_compressor, "should_compress_info", None)
                if callable(_info):
                    try:
                        _compress_block_reason = (
                            await _await_engine_result(_info(_preflight_tokens))
                        )[1]
                    except Exception:
                        _compress_block_reason = None
        if _should_compress_now:
            _preflight_compressed = True
            # Compression is actually running (block cleared / was never
            # blocked) — reset the dedup so a future blocked-over-threshold
            # turn can warn again. Real session boundary.
            # getattr guard: test doubles built via object.__new__ lack the
            # method (gateway test-double pitfall) — treat absence as no-op.
            _clear_warn = getattr(agent, "_clear_context_overflow_warn", None)
            if callable(_clear_warn):
                _clear_warn()
            logger.info(
                "Preflight compression: ~%s tokens >= %s threshold (model %s, ctx %s)",
                f"{_preflight_tokens:,}",
                f"{_compressor.threshold_tokens:,}",
                agent.model,
                f"{_compressor.context_length:,}",
            )
            _preflight_status = automatic_compaction_status_message(
                _compressor,
                phase="preflight",
                default_message=PREFLIGHT_COMPRESSION_STATUS_TEMPLATE.format(
                    tokens=_preflight_tokens,
                    threshold=_compressor.threshold_tokens,
                ),
                approx_tokens=_preflight_tokens,
                threshold_tokens=_compressor.threshold_tokens,
                context_length=_compressor.context_length,
                model=agent.model,
            )
            if _preflight_status:
                agent._emit_status(_preflight_status)
            # Preflight passes honor the same configured per-turn cap
            # (compression.max_attempts) as the loop's compression sites;
            # default 3 preserves the prior hardcoded behavior.
            _max_preflight_passes = max(
                1, int(getattr(agent, "max_compression_attempts", 3) or 3)
            )
            for _pass in range(_max_preflight_passes):
                _orig_len = len(messages)
                _orig_tokens = _preflight_tokens
                _preflight_input = messages
                messages, active_system_prompt = await agent._compress_context(
                    messages, system_message, approx_tokens=_preflight_tokens,
                    task_id=effective_task_id,
                )
                if (
                    messages is _preflight_input
                    and compression_skipped_due_to_lock(agent)
                ):
                    # #69870 lock-skip: another path holds this session's
                    # compression lock, so the pass no-oped. That is a
                    # temporary DEFER, not proof the transcript cannot
                    # compress — do NOT arm the insufficient-progress
                    # blocker (the loop's error handlers must keep their
                    # provider-proven retry budget) and stop preflight
                    # passes for this turn; the lock winner is shrinking
                    # the same session concurrently.
                    logger.info(
                        "Preflight compression deferred: compression lock "
                        "held by another path (session %s)",
                        agent.session_id or "none",
                    )
                    break
                # Re-estimate now so size-only compression (same row count,
                # lower token count — e.g. summarising tool outputs) is
                # recognised as progress instead of being misread as
                # "Cannot compress further". Fixes #39548.
                _preflight_tokens = estimate_request_tokens_rough(
                    messages,
                    system_prompt=active_system_prompt or "",
                    tools=agent.tools or None,
                )
                if not _compression_made_progress(
                    _orig_len, len(messages), _orig_tokens, _preflight_tokens
                ):
                    _preflight_compression_blocked = True
                    break  # Cannot compress further: neither rows nor tokens moved
                conversation_history = conversation_history_after_compression(
                    agent, messages, conversation_history
                )
                agent._empty_content_retries = 0
                agent._thinking_prefill_retries = 0
                agent._last_content_with_tools = None
                agent._last_content_tools_all_housekeeping = False
                agent._mute_post_response = False
                if not await _await_engine_result(
                    _compressor.should_compress(_preflight_tokens)
                ):
                    break
                if not _compression_warrants_another_preflight_pass(
                    _orig_tokens,
                    _preflight_tokens,
                    _compressor.threshold_tokens,
                ):
                    _preflight_compression_blocked = True
                    logger.warning(
                        "Preflight compression made insufficient progress: "
                        "~%s -> ~%s request tokens; skipping additional passes",
                        f"{_orig_tokens:,}",
                        f"{_preflight_tokens:,}",
                    )
                    break
        elif _compress_block_reason:
            # Context is already over the compression threshold, but compression
            # is blocked (summary LLM cooldown or anti-thrashing). Without a
            # signal the session keeps growing until the model silently stops
            # answering — the conversation hits the hard provider token limit
            # with no explanation. Surface a deduped warning so the user can
            # take action (/new or /compress) instead of hitting a silent hang.
            agent._warn_context_overflow_blocked(
                _compress_block_reason,
                _preflight_tokens,
                _compressor.threshold_tokens,
            )
        else:
            # Sub-threshold and unblocked — allow the overflow warning to fire
            # again next time the context is over threshold but blocked.
            # getattr guard: test doubles built via object.__new__ lack the
            # method (gateway test-double pitfall) — treat absence as no-op.
            _clear_warn = getattr(agent, "_clear_context_overflow_warn", None)
            if callable(_clear_warn):
                _clear_warn()
            # Engine maintenance only when NO skip-branch fired: a failure
            # cooldown or deferred estimate must keep the engine hook
            # un-consulted (#20316 contract — the cooldown
            # exists precisely because compression recently failed).
            if _compression_cooldown or _preflight_deferred:
                _engine_preflight = None
            else:
                _engine_preflight = getattr(
                    _compressor, "should_compress_preflight", None
                )
            # ── Engine-driven sub-threshold preflight maintenance (#20316) ──
            # None of the threshold-path branches fired (not deferred, no
            # failure cooldown, and should_compress() said
            # the request is under pressure). Context engines that override
            # ``should_compress_preflight()`` (e.g. LCM-style incremental
            # leaf-chunk compaction) can still request deferred maintenance
            # below the token threshold. The default
            # ``ContextEngine.should_compress_preflight()`` returns False, so
            # the built-in ``ContextCompressor`` path is byte-identical.
            #
            # Attempt-cap integration: the engine gets exactly ONE
            # ``compress()`` pass per turn. It is mutually exclusive with the
            # threshold multi-pass loop above (if/elif), so turn-start
            # preflight passes stay bounded by the resolved
            # ``compression.max_attempts`` cap (floor 1) in every case.
            #
            # No-op-blocking integration: a sub-threshold engine pass that
            # no-ops says nothing about over-threshold compressibility, so it
            # must neither set nor clear ``_preflight_compression_blocked``
            # (#64382) — and being in the ``else`` arm it can never run after
            # the threshold loop has proven a retry ineffective.
            # (resolved above, gated on no skip-branch having fired)
            _wants_engine_preflight = False
            if callable(_engine_preflight):
                try:
                    _wants_engine_preflight = bool(_engine_preflight(messages))
                except Exception as _preflight_exc:
                    # A buggy engine must never break an otherwise-healthy
                    # turn: swallow at debug level and skip maintenance.
                    logger.debug(
                        "should_compress_preflight raised %s; skipping "
                        "engine-driven preflight maintenance",
                        _preflight_exc,
                    )
                    _wants_engine_preflight = False
            if _wants_engine_preflight:
                logger.info(
                    "Engine-driven preflight maintenance: %s requested "
                    "compress() at ~%s tokens (below %s threshold)",
                    getattr(_compressor, "name", type(_compressor).__name__),
                    f"{_preflight_tokens:,}",
                    f"{getattr(_compressor, 'threshold_tokens', 0):,}",
                )
                _engine_input = messages
                messages, active_system_prompt = await agent._compress_context(
                    messages, system_message, approx_tokens=_preflight_tokens,
                    task_id=effective_task_id,
                )
                # ``_compress_context`` returns the INPUT list object on every
                # skip path (per-session lock held elsewhere, cooldown,
                # anti-thrash breaker, codex-native routing) and an engine may
                # legitimately no-op. Only re-baseline the flush history and
                # re-anchor the user row after a REAL compaction — a skip must
                # leave the turn's bookkeeping untouched.
                if messages is not _engine_input:
                    _preflight_compressed = True
                    conversation_history = conversation_history_after_compression(
                        agent, messages
                    )
                    agent._empty_content_retries = 0
                    agent._thinking_prefill_retries = 0
                    agent._last_content_with_tools = None
                    agent._last_content_tools_all_housekeeping = False
                    agent._mute_post_response = False

    if _preflight_compressed:
        # Compression rebuilt the list (tail messages are fresh compaction
        # copies), so the pre-compression index of this turn's user message
        # is stale. Re-anchor both index trackers: the api_content stamp
        # below, the loop's injection site, and the flush's persist-override
        # row (#48677) must all target the surviving dict, not a stale
        # position. Exact-content match first so a todo-snapshot user message
        # appended after the tail can't steal the anchor.
        current_turn_user_idx = reanchor_current_turn_user_idx(
            messages, user_message
        )
        agent._persist_user_message_idx = current_turn_user_idx

    plugin_user_context = ""

    # Plugins may contribute request-only context for this turn. Keep it on
    # the current user message rather than the system prompt: the latter is a
    # stable, cached prefix for the lifetime of the conversation.
    #
    # The async lifecycle dispatcher rejects synchronous callbacks instead of
    # running them in a worker. Do not hide that capability error here; users
    # must either migrate the plugin hook or disable it.
    try:
        from hermes_cli.lifecycle import invoke_hook
        from hermes_cli.plugins import _PluginContractError

        pre_llm_results = await invoke_hook(
            "pre_llm_call",
            session_id=agent.session_id,
            task_id=effective_task_id,
            turn_id=turn_id,
            user_message=original_user_message,
            conversation_history=list(messages),
            is_first_turn=not bool(conversation_history),
            model=agent.model,
            platform=getattr(agent, "platform", None) or "",
            parent_session_id=getattr(agent, "_parent_session_id", None) or "",
            sender_id=getattr(agent, "_user_id", None) or "",
        )
    except _PluginContractError:
        raise
    except Exception as exc:
        logger.warning("pre_llm_call hook failed: %s", exc)
    else:
        context_parts = []
        spill_config = await _hook_output_spill.get_spill_config()
        for result in pre_llm_results:
            piece = ""
            if isinstance(result, dict) and result.get("context"):
                piece = str(result["context"])
            elif isinstance(result, str) and result.strip():
                piece = result
            if piece:
                context_parts.append(
                    await _hook_output_spill.spill_if_oversized(
                        piece,
                        session_id=agent.session_id,
                        source="plugin hook",
                        config=spill_config,
                    )
                )
        if context_parts:
            plugin_user_context = "\n\n".join(context_parts)

    # Per-turn file-mutation verifier state.
    agent._turn_failed_file_mutations = {}
    agent._turn_file_mutation_paths = set()
    agent._verification_stop_nudges = 0
    agent._pre_verify_nudges = 0

    # Clear stale task-local tool state while preserving a pre-start interrupt.
    interrupt_event = getattr(agent, "_interrupt_event", None)
    if interrupt_event is None:
        interrupt_event = asyncio.Event()
        agent._interrupt_event = interrupt_event
    ra()._set_interrupt(False, interrupt_event)
    if agent._interrupt_requested:
        ra()._set_interrupt(True, interrupt_event)
    else:
        agent._interrupt_message = None

    # Notify memory providers before recall so cadence-sensitive providers see
    # the current turn number before deciding whether to inject context.
    memory_manager = getattr(agent, "_memory_manager", None)
    if memory_manager:
        try:
            turn_message = (
                original_user_message
                if isinstance(original_user_message, str)
                else ""
            )
            await memory_manager.on_turn_start(
                agent._user_turn_count,
                turn_message,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("Memory provider on_turn_start failed", exc_info=True)

    # External memory provider: prefetch once before the tool loop. Trivial
    # prompts carry no useful recall signal and intentionally skip injection.
    ext_prefetch_cache = ""
    if memory_manager:
        try:
            query = (
                original_user_message
                if isinstance(original_user_message, str)
                else ""
            )
            if not is_trivial_prompt(query):
                ext_prefetch_cache = (
                    await memory_manager.prefetch_all(query)
                ) or ""
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("Memory provider prefetch failed", exc_info=True)

    # ── api_content sidecar: persist what you send ──
    # The prefetch/plugin context above is injected into the API copy of this
    # turn's user message, never into the stored content — so on the next
    # turn the message would replay WITHOUT the injection, diverging the
    # request prefix at this point and re-prefilling everything after it
    # (the whole previous turn's assistant/tool chain). Stamp the exact
    # API-bound bytes on the live dict, only when they differ from the clean
    # content, so the crash persist below writes both in the same row and
    # replay can reproduce the sent prefix byte-for-byte. Guarded by the
    # same predicate the api_messages build uses, so the stamped bytes are
    # exactly the bytes the loop sends. codex_app_server turns bypass the
    # api_messages build entirely (the codex thread gets the plain user
    # message), so stamping there would persist bytes that were never sent.
    # MoA turns append per-call aggregated
    # reference context to the same API
    # copy AFTER this composition, so the stamped bytes would never match the
    # wire either — skip the stamp rather than persist provably wrong "exact
    # sent bytes" (MoA keeps its pre-sidecar cache behavior).
    if (
        not moa_active
        and getattr(agent, "api_mode", None) != "codex_app_server"
        and 0 <= current_turn_user_idx < len(messages)
        and messages[current_turn_user_idx].get("role") == "user"
    ):
        _turn_user_msg = messages[current_turn_user_idx]
        _api_content = compose_user_api_content(
            _turn_user_msg.get("content", ""), ext_prefetch_cache, plugin_user_context
        )
        if _api_content is not None and _api_content != _turn_user_msg.get("content"):
            _turn_user_msg["api_content"] = _api_content
            # In-place preflight compaction has ALREADY inserted this turn's
            # user row (archive_and_compact runs before prefetch/pre_llm_call
            # can compose the sidecar), and the crash persist below identity-
            # skips every compacted dict (they are all in the rebound
            # conversation_history) — so the stamp would never reach the DB.
            # Backfill it onto the freshly-inserted row directly. Rotation
            # mode needs nothing here: its compacted copies flush to the
            # child session after this stamp.
            if _preflight_compressed and bool(
                getattr(agent, "_last_compaction_in_place", False)
            ):
                if getattr(agent, "_session_db", None) is not None:
                    try:
                        await agent._session_db.set_latest_user_api_content(
                            agent.session_id,
                            _turn_user_msg.get("content"),
                            _api_content,
                        )
                    except Exception:
                        logger.warning(
                            "in-place compaction api_content backfill failed "
                            "for session=%s",
                            agent.session_id or "none",
                            exc_info=True,
                        )

    # Crash-resilience: persist the inbound user turn before the first LLM
    # call. Runs after preflight compression (which rewrites history anyway)
    # and after prefetch/pre_llm_call, so the user row is written once with
    # its final api_content instead of being re-written mid-turn.
    try:
        # Retry row creation immediately before the crash-safe flush if the
        # earlier turn-start attempt failed transiently.  ``_persist_session``
        # owns the transcript lock, so do not acquire it here as well.
        await agent._ensure_db_session()
        await agent._persist_session(messages, conversation_history)
    except Exception:
        logger.warning(
            "Early turn-start session persistence failed for session=%s",
            agent.session_id or "none",
            exc_info=True,
        )
    finally:
        # Keep an unmarked staged input available to a later close retry if the
        # normal persistence attempt failed. Once the marker is present, the
        # close path must no longer treat it as a pre-worker UI input.
        if not isinstance(pending_cli_message, dict) or pending_cli_message.get("_db_persisted"):
            agent._pending_cli_user_message = None

    return TurnContext(
        user_message=user_message,
        original_user_message=original_user_message,
        messages=messages,
        conversation_history=conversation_history,
        active_system_prompt=active_system_prompt,
        effective_task_id=effective_task_id,
        turn_id=turn_id,
        current_turn_user_idx=current_turn_user_idx,
        should_review_memory=should_review_memory,
        plugin_user_context=plugin_user_context,
        ext_prefetch_cache=ext_prefetch_cache,
        preflight_compression_blocked=_preflight_compression_blocked,
    )
