"""OpenAI-compatible shim that forwards Hermes requests to ``copilot --acp``."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import subprocess
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator

import aiofiles
import aiofiles.os
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

from agent.file_safety import get_read_block_error, get_write_denied_error
from agent.redact import redact_sensitive_text
from tools.environments.local import hermes_subprocess_env

ACP_MARKER_BASE_URL = "acp://copilot"
_DEFAULT_TIMEOUT_SECONDS = 900.0

_TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL
)
_TOOL_CALL_JSON_RE = re.compile(
    r'\{\s*"id"\s*:\s*"[^"]+"\s*,\s*"type"\s*:\s*"function"'
    r'\s*,\s*"function"\s*:\s*\{.*?\}\s*\}',
    re.DOTALL,
)

_DEPRECATION_REQUIRED = ("gh-copilot",)
_DEPRECATION_MARKERS = (
    "has been deprecated",
    "no commands will be executed",
)


def _is_gh_copilot_deprecation_message(stderr_text: str) -> bool:
    """Return whether stderr is the deprecated gh-copilot extension banner."""
    lowered = stderr_text.lower()
    if not any(required in lowered for required in _DEPRECATION_REQUIRED):
        return False
    return any(marker in lowered for marker in _DEPRECATION_MARKERS)


def _resolve_command() -> str:
    return (
        os.getenv("HERMES_COPILOT_ACP_COMMAND", "").strip()
        or os.getenv("COPILOT_CLI_PATH", "").strip()
        or "copilot"
    )


def _resolve_args() -> list[str]:
    raw = os.getenv("HERMES_COPILOT_ACP_ARGS", "").strip()
    if not raw:
        return ["--acp", "--stdio"]
    return shlex.split(raw)


async def _resolve_home_dir() -> str:
    """Return a stable HOME for child ACP processes."""
    home = os.environ.get("HOME", "").strip()
    if home:
        return home
    expanded = await aiofiles.os.wrap(os.path.expanduser)("~")
    if expanded and expanded != "~":
        return expanded
    try:
        import pwd

        get_home = aiofiles.os.wrap(lambda: pwd.getpwuid(os.getuid()).pw_dir)
        resolved = (await get_home()).strip()
        if resolved:
            return resolved
    except Exception:
        pass
    return "/tmp"


async def _build_subprocess_env() -> dict[str, str]:
    env = await hermes_subprocess_env(inherit_credentials=True)
    env["HOME"] = await _resolve_home_dir()
    from hermes_constants import apply_subprocess_home_env

    await apply_subprocess_home_env(env)
    return env


def _jsonrpc_error(message_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": message_id,
        "error": {"code": code, "message": message},
    }


def _permission_denied(message_id: Any) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": message_id,
        "result": {"outcome": {"outcome": "cancelled"}},
    }


def _format_messages_as_prompt(
    messages: list[dict[str, Any]],
    model: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
) -> str:
    sections: list[str] = [
        "You are being used as the active ACP agent backend for Hermes.",
        "Use ACP capabilities to complete tasks.",
        "IMPORTANT: If you take an action with a tool, you MUST output tool calls "
        "using <tool_call>{...}</tool_call> blocks with JSON exactly in OpenAI "
        "function-call shape.",
        "If no tool is needed, answer normally.",
    ]
    if model:
        sections.append(f"Hermes requested model hint: {model}")

    if isinstance(tools, list) and tools:
        tool_specs: list[dict[str, Any]] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            function = tool.get("function") or {}
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            tool_specs.append(
                {
                    "name": name.strip(),
                    "description": function.get("description", ""),
                    "parameters": function.get("parameters", {}),
                }
            )
        if tool_specs:
            sections.append(
                "Available tools (OpenAI function schema). When using a tool, "
                "emit ONLY <tool_call>{...}</tool_call> with one JSON object "
                "containing id/type/function{name,arguments}. arguments must be "
                "a JSON string.\n"
                + json.dumps(tool_specs, ensure_ascii=False)
            )

    if tool_choice is not None:
        sections.append(
            f"Tool choice hint: {json.dumps(tool_choice, ensure_ascii=False)}"
        )

    transcript: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "unknown").strip().lower()
        if role not in {"system", "user", "assistant", "tool"}:
            role = "context"
        rendered = _render_message_content(message.get("content"))
        if not rendered:
            continue
        label = {
            "system": "System",
            "user": "User",
            "assistant": "Assistant",
            "tool": "Tool",
            "context": "Context",
        }.get(role, role.title())
        transcript.append(f"{label}:\n{rendered}")

    if transcript:
        sections.append("Conversation transcript:\n\n" + "\n\n".join(transcript))
    sections.append("Continue the conversation from the latest user request.")
    return "\n\n".join(
        section.strip() for section in sections if section and section.strip()
    )


def _render_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        if "text" in content:
            return str(content.get("text") or "").strip()
        if "content" in content and isinstance(content.get("content"), str):
            return str(content.get("content") or "").strip()
        return json.dumps(content, ensure_ascii=True)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n".join(parts).strip()
    return str(content).strip()


def _build_openai_tool_call(
    *, call_id: str, name: str, arguments: str
) -> ChatCompletionMessageToolCall:
    return ChatCompletionMessageToolCall(
        id=call_id,
        call_id=call_id,
        response_item_id=None,
        type="function",
        function=Function(name=name, arguments=arguments),
    )


async def _completion_to_stream_chunks(
    completion: SimpleNamespace,
) -> AsyncIterator[SimpleNamespace]:
    """Yield a one-shot ACP response as OpenAI-style async stream chunks."""
    choice = completion.choices[0]
    message = choice.message
    tool_call_deltas = None
    if message.tool_calls:
        tool_call_deltas = [
            SimpleNamespace(
                index=index,
                id=getattr(tool_call, "id", None),
                type=getattr(tool_call, "type", "function"),
                function=SimpleNamespace(
                    name=getattr(tool_call.function, "name", None),
                    arguments=getattr(tool_call.function, "arguments", None),
                ),
            )
            for index, tool_call in enumerate(message.tool_calls)
        ]

    yield SimpleNamespace(
        choices=[
            SimpleNamespace(
                index=0,
                delta=SimpleNamespace(
                    role="assistant",
                    content=message.content or None,
                    tool_calls=tool_call_deltas,
                    reasoning_content=message.reasoning_content,
                    reasoning=message.reasoning,
                ),
                finish_reason=choice.finish_reason,
            )
        ],
        model=completion.model,
        usage=None,
    )
    yield SimpleNamespace(choices=[], model=completion.model, usage=completion.usage)


def _extract_tool_calls_from_text(
    text: str,
) -> tuple[list[ChatCompletionMessageToolCall], str]:
    if not isinstance(text, str) or not text.strip():
        return [], ""

    extracted: list[ChatCompletionMessageToolCall] = []
    consumed_spans: list[tuple[int, int]] = []

    def _try_add_tool_call(raw_json: str) -> None:
        try:
            obj = json.loads(raw_json)
        except Exception:
            return
        if not isinstance(obj, dict):
            return
        function = obj.get("function")
        if not isinstance(function, dict):
            return
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            return
        arguments = function.get("arguments", "{}")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        call_id = obj.get("id")
        if not isinstance(call_id, str) or not call_id.strip():
            call_id = f"acp_call_{len(extracted) + 1}"
        extracted.append(
            _build_openai_tool_call(
                call_id=call_id,
                name=name.strip(),
                arguments=arguments,
            )
        )

    for match in _TOOL_CALL_BLOCK_RE.finditer(text):
        _try_add_tool_call(match.group(1))
        consumed_spans.append((match.start(), match.end()))
    if not extracted:
        for match in _TOOL_CALL_JSON_RE.finditer(text):
            _try_add_tool_call(match.group(0))
            consumed_spans.append((match.start(), match.end()))
    if not consumed_spans:
        return extracted, text.strip()

    consumed_spans.sort()
    merged: list[tuple[int, int]] = []
    for start, end in consumed_spans:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    parts: list[str] = []
    cursor = 0
    for start, end in merged:
        if cursor < start:
            parts.append(text[cursor:start])
        cursor = max(cursor, end)
    if cursor < len(text):
        parts.append(text[cursor:])
    cleaned = "\n".join(
        part.strip() for part in parts if part and part.strip()
    ).strip()
    return extracted, cleaned


async def _ensure_path_within_cwd(path_text: str, cwd: str) -> Path:
    candidate = Path(path_text)
    if not candidate.is_absolute():
        raise PermissionError("ACP file-system paths must be absolute.")
    resolved = await aiofiles.os.wrap(candidate.resolve)()
    root = await aiofiles.os.wrap(Path(cwd).resolve)()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PermissionError(
            f"Path '{resolved}' is outside the session cwd '{root}'."
        ) from exc
    return resolved


class _ACPChatCompletions:
    def __init__(self, client: "CopilotACPClient"):
        self._client = client

    async def create(self, **kwargs: Any) -> Any:
        return await self._client._create_chat_completion(**kwargs)


class _ACPChatNamespace:
    def __init__(self, client: "CopilotACPClient"):
        self.completions = _ACPChatCompletions(client)


class CopilotACPClient:
    """Minimal native-async OpenAI-client-compatible facade for Copilot ACP."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        acp_command: str | None = None,
        acp_args: list[str] | None = None,
        acp_cwd: str | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        **_: Any,
    ):
        self.api_key = api_key or "copilot-acp"
        self.base_url = base_url or ACP_MARKER_BASE_URL
        self._default_headers = dict(default_headers or {})
        self._acp_command = acp_command or command or _resolve_command()
        self._acp_args = list(acp_args or args or _resolve_args())
        # CWD resolution is a filesystem boundary.  Keep construction
        # state-only and resolve it at the first awaited subprocess boundary
        # instead of calling ``os.getcwd()`` from an async request.
        self._acp_cwd = str(acp_cwd) if acp_cwd else None
        self.chat = _ACPChatNamespace(self)
        self.is_closed = False
        self._active_process: asyncio.subprocess.Process | None = None
        self._active_process_lock = asyncio.Lock()

    async def close(self) -> None:
        async with self._active_process_lock:
            process = self._active_process
            self._active_process = None
            self.is_closed = True
        if process is None or process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=2)
        except asyncio.CancelledError:
            if process.returncode is None:
                process.kill()
            await process.wait()
            raise
        except asyncio.TimeoutError:
            if process.returncode is None:
                process.kill()
            await process.wait()

    async def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        stream: bool = False,
        **_: Any,
    ) -> Any:
        prompt_text = _format_messages_as_prompt(
            messages or [], model=model, tools=tools, tool_choice=tool_choice
        )
        if timeout is None:
            effective_timeout = _DEFAULT_TIMEOUT_SECONDS
        elif isinstance(timeout, (int, float)):
            effective_timeout = float(timeout)
        else:
            candidates = [
                getattr(timeout, attribute, None)
                for attribute in ("read", "write", "connect", "pool", "timeout")
            ]
            numeric = [
                float(value)
                for value in candidates
                if isinstance(value, (int, float))
            ]
            effective_timeout = (
                max(numeric) if numeric else _DEFAULT_TIMEOUT_SECONDS
            )

        response_text, reasoning_text = await self._run_prompt(
            prompt_text, timeout_seconds=effective_timeout
        )
        tool_calls, cleaned_text = _extract_tool_calls_from_text(response_text)
        usage = SimpleNamespace(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        )
        assistant_message = SimpleNamespace(
            content=cleaned_text,
            tool_calls=tool_calls,
            reasoning=reasoning_text or None,
            reasoning_content=reasoning_text or None,
            reasoning_details=None,
        )
        finish_reason = "tool_calls" if tool_calls else "stop"
        completion = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=assistant_message,
                    finish_reason=finish_reason,
                )
            ],
            usage=usage,
            model=model or "copilot-acp",
        )
        if stream:
            return _completion_to_stream_chunks(completion)
        return completion

    async def _run_prompt(
        self, prompt_text: str, *, timeout_seconds: float
    ) -> tuple[str, str]:
        cwd = self._acp_cwd or await aiofiles.os.getcwd()
        resolved_cwd = await aiofiles.os.wrap(Path(cwd).resolve)()
        self._acp_cwd = str(resolved_cwd)
        try:
            process = await asyncio.create_subprocess_exec(
                self._acp_command,
                *self._acp_args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._acp_cwd,
                env=await _build_subprocess_env(),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Could not start Copilot ACP command '{self._acp_command}'. "
                "Install GitHub Copilot CLI or set "
                "HERMES_COPILOT_ACP_COMMAND/COPILOT_CLI_PATH."
            ) from exc

        if process.stdin is None or process.stdout is None or process.stderr is None:
            process.kill()
            await process.wait()
            raise RuntimeError("Copilot ACP process did not expose stdio pipes.")

        async with self._active_process_lock:
            self._active_process = process
            self.is_closed = False

        stderr_tail: deque[str] = deque(maxlen=40)

        async def read_stderr() -> None:
            while line := await process.stderr.readline():
                stderr_tail.append(line.decode("utf-8", errors="replace").rstrip("\n"))

        stderr_task = asyncio.create_task(read_stderr())
        next_id = 0

        async def request(
            method: str,
            params: dict[str, Any],
            *,
            text_parts: list[str] | None = None,
            reasoning_parts: list[str] | None = None,
        ) -> Any:
            nonlocal next_id
            next_id += 1
            request_id = next_id
            payload = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            }
            process.stdin.write((json.dumps(payload) + "\n").encode())
            await process.stdin.drain()

            deadline = asyncio.get_running_loop().time() + timeout_seconds
            while asyncio.get_running_loop().time() < deadline:
                remaining = deadline - asyncio.get_running_loop().time()
                try:
                    raw_line = await asyncio.wait_for(
                        process.stdout.readline(), timeout=min(0.1, remaining)
                    )
                except asyncio.TimeoutError:
                    continue
                if not raw_line:
                    if process.returncode is None:
                        await asyncio.sleep(0)
                        continue
                    break
                line = raw_line.decode("utf-8", errors="replace")
                try:
                    message = json.loads(line)
                except Exception:
                    message = {"raw": line.rstrip("\n")}

                if await self._handle_server_message(
                    message,
                    process=process,
                    cwd=self._acp_cwd,
                    text_parts=text_parts,
                    reasoning_parts=reasoning_parts,
                ):
                    continue
                if message.get("id") != request_id:
                    continue
                if "error" in message:
                    error = message.get("error") or {}
                    raise RuntimeError(
                        f"Copilot ACP {method} failed: "
                        f"{error.get('message') or error}"
                    )
                return message.get("result")

            if process.returncode is not None:
                await stderr_task
            stderr_text = "\n".join(stderr_tail).strip()
            if process.returncode is not None and stderr_text:
                if _is_gh_copilot_deprecation_message(stderr_text):
                    raise RuntimeError(
                        "Hermes ACP mode requires the NEW GitHub Copilot CLI "
                        "(github.com/github/copilot-cli), but the binary it just "
                        "spawned is the deprecated `gh copilot` extension.\n\n"
                        "Install the new CLI:\n"
                        "  npm install -g @github/copilot\n"
                        "  # then verify with: copilot --help\n\n"
                        "If `copilot` already resolves to the new CLI but you "
                        "still see this, point Hermes at it explicitly:\n"
                        "  export HERMES_COPILOT_ACP_COMMAND=/path/to/new/copilot"
                        "\n\nAlternative: use the `copilot` provider directly.\n\n"
                        f"Original error:\n{stderr_text}"
                    )
                raise RuntimeError(
                    f"Copilot ACP process exited early: {stderr_text}"
                )
            raise TimeoutError(
                f"Timed out waiting for Copilot ACP response to {method}."
            )

        try:
            await request(
                "initialize",
                {
                    "protocolVersion": 1,
                    "clientCapabilities": {
                        "fs": {"readTextFile": True, "writeTextFile": True}
                    },
                    "clientInfo": {
                        "name": "hermes-agent",
                        "title": "Hermes Agent",
                        "version": "0.0.0",
                    },
                },
            )
            session = await request(
                "session/new", {"cwd": self._acp_cwd, "mcpServers": []}
            ) or {}
            session_id = str(session.get("sessionId") or "").strip()
            if not session_id:
                raise RuntimeError("Copilot ACP did not return a sessionId.")
            text_parts: list[str] = []
            reasoning_parts: list[str] = []
            await request(
                "session/prompt",
                {
                    "sessionId": session_id,
                    "prompt": [{"type": "text", "text": prompt_text}],
                },
                text_parts=text_parts,
                reasoning_parts=reasoning_parts,
            )
            return "".join(text_parts), "".join(reasoning_parts)
        finally:
            await self.close()
            await stderr_task

    async def _handle_server_message(
        self,
        msg: dict[str, Any],
        *,
        process: asyncio.subprocess.Process,
        cwd: str,
        text_parts: list[str] | None,
        reasoning_parts: list[str] | None,
    ) -> bool:
        method = msg.get("method")
        if not isinstance(method, str):
            return False

        if method == "session/update":
            params = msg.get("params") or {}
            update = params.get("update") or {}
            kind = str(update.get("sessionUpdate") or "").strip()
            content = update.get("content") or {}
            chunk_text = (
                str(content.get("text") or "") if isinstance(content, dict) else ""
            )
            if kind == "agent_message_chunk" and chunk_text and text_parts is not None:
                text_parts.append(chunk_text)
            elif (
                kind == "agent_thought_chunk"
                and chunk_text
                and reasoning_parts is not None
            ):
                reasoning_parts.append(chunk_text)
            return True

        if process.stdin is None:
            return True
        message_id = msg.get("id")
        params = msg.get("params") or {}

        if method == "session/request_permission":
            response = _permission_denied(message_id)
        elif method == "fs/read_text_file":
            try:
                path = await _ensure_path_within_cwd(
                    str(params.get("path") or ""), cwd
                )
                block_error = await get_read_block_error(str(path))
                if block_error:
                    raise PermissionError(block_error)
                try:
                    async with aiofiles.open(path, encoding="utf-8") as handle:
                        content = await handle.read()
                except FileNotFoundError:
                    content = ""
                line = params.get("line")
                limit = params.get("limit")
                if isinstance(line, int) and line > 1:
                    lines = content.splitlines(keepends=True)
                    start = line - 1
                    end = (
                        start + limit
                        if isinstance(limit, int) and limit > 0
                        else None
                    )
                    content = "".join(lines[start:end])
                if content:
                    content = redact_sensitive_text(content, force=True)
                response = {
                    "jsonrpc": "2.0",
                    "id": message_id,
                    "result": {"content": content},
                }
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                response = _jsonrpc_error(message_id, -32602, str(exc))
        elif method == "fs/write_text_file":
            try:
                path = await _ensure_path_within_cwd(
                    str(params.get("path") or ""), cwd
                )
                denied = await get_write_denied_error(str(path))
                if denied:
                    raise PermissionError(denied)
                await aiofiles.os.makedirs(path.parent, exist_ok=True)
                async with aiofiles.open(path, "w", encoding="utf-8") as handle:
                    await handle.write(str(params.get("content") or ""))
                response = {"jsonrpc": "2.0", "id": message_id, "result": None}
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                response = _jsonrpc_error(message_id, -32602, str(exc))
        else:
            response = _jsonrpc_error(
                message_id,
                -32601,
                f"ACP client method '{method}' is not supported by Hermes yet.",
            )

        process.stdin.write((json.dumps(response) + "\n").encode())
        await process.stdin.drain()
        return True
