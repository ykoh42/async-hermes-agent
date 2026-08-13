"""
SWE Runner with Hermes Trajectory Format

A native-async library runner that uses Hermes-Agent's retained execution
environment and outputs trajectories in the Hermes-Agent format
compatible with batch_runner.py and trajectory_compressor.py.

Features:
- Uses Hermes-Agent's local, Docker, or Modal environment for command execution
- Outputs trajectories in Hermes format (from/value pairs with <tool_call>/<tool_response> XML)
- Compatible with the trajectory compression pipeline
- Supports batch processing from JSONL prompt files
"""

import asyncio
import inspect
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import aiofiles
import aiofiles.os

from agent.auxiliary_client import (
    OMIT_TEMPERATURE,
    _fixed_temperature_for_model,
    resolve_provider_client,
)
from agent.process_bootstrap import OpenAI as AsyncOpenAI
from agent.secret_scope import get_secret, is_multiplex_active
from agent.tool_dispatch_helpers import make_tool_result_message
from hermes_cli.env_loader import load_hermes_dotenv
from hermes_constants import get_hermes_home


def _effective_temperature_for_model(
    model: str,
    base_url: str | None = None,
) -> float | None:
    """Return a fixed temperature for models with strict sampling contracts.

    Returns ``None`` when the model manages temperature server-side (Kimi);
    callers must omit the ``temperature`` kwarg entirely in that case.
    """
    result = _fixed_temperature_for_model(model, base_url)
    if result is OMIT_TEMPERATURE:
        return None  # caller must omit temperature
    return cast(float | None, result)


async def _finish_owned_task(task: asyncio.Task[Any]) -> Any:
    """Finish owned cleanup before propagating external cancellation."""
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError as error:  # noqa: ASYNC103 - re-raised below
            if task.cancelled():
                raise
            if cancellation is None:
                cancellation = error
        except Exception as error:
            if cancellation is not None:
                raise cancellation from error
            raise
    if cancellation is not None:
        raise cancellation
    return result


def _fallback_provider_api_key() -> str:
    """Preserve upstream's OPENROUTER -> ANTHROPIC -> OPENAI precedence."""
    openrouter_key = get_secret("OPENROUTER_API_KEY")
    if openrouter_key is not None:
        return openrouter_key
    anthropic_key = get_secret("ANTHROPIC_API_KEY")
    if anthropic_key is not None:
        return anthropic_key
    return get_secret("OPENAI_API_KEY", "") or ""


# ============================================================================
# Terminal Tool Definition (matches Hermes-Agent format)
# ============================================================================

TERMINAL_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "terminal",
        "description": """Execute bash commands in a sandboxed environment.

**Environment:**
- Isolated execution environment (local, Docker, or Modal cloud)
- Filesystem persists between tool calls within the same task
- Internet access available

**Command Execution:**
- Provide the command to execute via the 'command' parameter
- Optional 'timeout' parameter in seconds (default: 60)

**Examples:**
- Run command: `{"command": "ls -la"}`
- With timeout: `{"command": "long_task.sh", "timeout": 300}`

**Best Practices:**
- Use non-interactive commands (avoid vim, nano, interactive python)
- Pipe to cat if output might be large
- Install tools with apt-get or pip as needed

**Completion:**
- When task is complete, output: echo "MINI_SWE_AGENT_FINAL_OUTPUT" followed by your result
""",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Command timeout in seconds (default: 60)"
                }
            },
            "required": ["command"]
        }
    }
}


# ============================================================================
# Environment Factory
# ============================================================================

def create_environment(
    env_type: str = "local",
    image: str = "python:3.11-slim",
    cwd: str = "/tmp",
    timeout: int = 60,
    **kwargs
):
    """
    Create an execution environment using Hermes-Agent's retained backend.

    Args:
        env_type: One of "local", "docker", "modal"
        image: Docker/Modal image name (ignored for local)
        cwd: Working directory
        timeout: Default command timeout
        **kwargs: Additional environment-specific options

    Returns:
        Environment instance with execute() and cleanup() methods
    """
    if env_type == "local":
        from tools.environments.local import LocalEnvironment

        return LocalEnvironment(cwd=cwd, timeout=timeout)
    if env_type == "docker":
        from tools.environments.docker import DockerEnvironment

        return DockerEnvironment(image=image, cwd=cwd, timeout=timeout, **kwargs)
    if env_type == "modal":
        from tools.environments.modal import ModalEnvironment

        return ModalEnvironment(image=image, cwd=cwd, timeout=timeout, **kwargs)

    raise ValueError(
        f"Unknown environment type: {env_type}. Use 'local', 'docker', or 'modal'"
    )


# ============================================================================
# Mini-SWE Runner with Hermes Trajectory Format
# ============================================================================

class MiniSWERunner:
    """
    Agent runner that uses Hermes-Agent's built-in execution environments
    and outputs trajectories in Hermes-Agent format.
    """

    def __init__(
        self,
        model: str = "anthropic/claude-sonnet-4.6",
        base_url: str | None = None,  # type: ignore[invalid-parameter-default]
        api_key: str | None = None,  # type: ignore[invalid-parameter-default]
        env_type: str = "local",
        image: str = "python:3.11-slim",
        cwd: str = "/tmp",
        max_iterations: int = 15,
        command_timeout: int = 60,
        verbose: bool = False,
    ):
        """
        Initialize the Mini-SWE Runner.

        Args:
            model: Model name for OpenAI-compatible API
            base_url: API base URL (optional, uses env vars if not provided)
            api_key: API key (optional, uses env vars if not provided)
            env_type: Environment type - "local", "docker", or "modal"
            image: Docker/Modal image (ignored for local)
            cwd: Working directory for commands
            max_iterations: Maximum tool-calling iterations
            command_timeout: Default timeout for commands
            verbose: Enable verbose logging
        """
        self.model = model
        self.max_iterations = max_iterations
        self.command_timeout = command_timeout
        self.verbose = verbose
        self.env_type = env_type
        self.image = image
        self.cwd = cwd

        self.logger = logging.getLogger(__name__)
        self._base_url = base_url
        self._api_key = api_key
        self.client = None
        self._owns_client = False
        self._configuration_loaded = False
        self._run_lock = asyncio.Lock()

        # Environment will be created per-task
        self.env: Any | None = None

        # Tool definition
        self.tools: list[dict[str, Any]] = [TERMINAL_TOOL_DEFINITION]

        print("🤖 Mini-SWE Runner initialized")
        print(f"   Model: {self.model}")
        print(f"   Environment: {self.env_type}")
        if self.env_type != "local":
            print(f"   Image: {self.image}")
        print(f"   Max iterations: {self.max_iterations}")

    async def _ensure_client(self) -> Any:
        """Initialize the native async provider client at an awaited boundary."""
        if self.client is not None:
            return self.client

        if not self._configuration_loaded:
            if not is_multiplex_active():
                project_env = Path(await aiofiles.os.getcwd()) / ".env"
                await load_hermes_dotenv(
                    hermes_home=get_hermes_home(),
                    project_env=project_env,
                )
            self._configuration_loaded = True

        if self._api_key or self._base_url:
            self.client = AsyncOpenAI(
                base_url=self._base_url or "https://openrouter.ai/api/v1",
                api_key=self._api_key or _fallback_provider_api_key(),
            )
            self._owns_client = True
            return self.client

        self.client, _ = await resolve_provider_client("openrouter", model=self.model)
        if self.client is None:
            self.client, _ = await resolve_provider_client("auto", model=self.model)
        if self.client is None:
            self.client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=get_secret("OPENROUTER_API_KEY", "") or "",
            )
        self._owns_client = True
        return self.client

    async def _create_env(self):
        """Create the execution environment."""
        print(f"🔧 Creating {self.env_type} environment...")
        environment = create_environment(
            env_type=self.env_type,
            image=self.image,
            cwd=self.cwd,
            timeout=self.command_timeout
        )
        self.env = environment
        initialize = getattr(environment, "_ensure_initialized", None)
        if initialize is not None:
            if not inspect.iscoroutinefunction(initialize):
                raise RuntimeError(
                    f"{self.env_type} environment initialization is not native async"
                )
            await initialize()
        print("✅ Environment ready")

    async def _cleanup_env(self):
        """Cleanup the execution environment."""
        environment = self.env
        self.env = None
        if environment is None:
            return
        cleanup = getattr(environment, "cleanup", None) or getattr(
            environment, "stop", None
        )
        if cleanup is None:
            return
        if not inspect.iscoroutinefunction(cleanup):
            raise RuntimeError(
                f"{self.env_type} environment cleanup is not native async"
            )
        await cleanup()

    async def _close_owned_client(self) -> None:
        if not self._owns_client or self.client is None:
            return
        client = self.client
        self.client = None
        self._owns_client = False
        close = getattr(client, "close", None)
        if not inspect.iscoroutinefunction(close):
            raise RuntimeError("MiniSWERunner provider client is not native async")
        await close()

    async def _cleanup_task_resources(self) -> None:
        async def _cleanup() -> None:
            try:
                await self._cleanup_env()
            finally:
                await self._close_owned_client()

        await _finish_owned_task(
            asyncio.create_task(_cleanup(), name="mini-swe-runner-cleanup")
        )

    async def _execute_command(
        self,
        command: str,
        timeout: int | None = None,  # type: ignore[invalid-parameter-default]
    ) -> dict[str, Any]:
        """
        Execute a command in the environment.

        Args:
            command: Bash command to execute
            timeout: Optional timeout override

        Returns:
            Dict with 'output' and 'returncode'
        """
        if self.env is None:
            await self._create_env()
        environment = self.env
        if environment is None:
            raise RuntimeError("Execution environment did not initialize")

        try:
            execute = environment.execute
            if not inspect.iscoroutinefunction(execute):
                raise RuntimeError(
                    f"{self.env_type} environment execution is not native async"
                )
            result = await execute(
                command,
                timeout=timeout or self.command_timeout,
            )
            return {
                "output": result.get("output", ""),
                "exit_code": result.get("returncode", 0),
                "error": None
            }
        except asyncio.CancelledError:
            raise
        except Exception as e:
            return {
                "output": "",
                "exit_code": -1,
                "error": str(e)
            }

    def _format_tools_for_system_message(self) -> str:
        """Format tool definitions for the system message."""
        formatted_tools = []
        for tool in self.tools:
            func = tool["function"]
            formatted_tools.append({
                "name": func["name"],
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {}),
                "required": None
            })
        return json.dumps(formatted_tools, ensure_ascii=False)

    def _convert_to_hermes_format(
        self,
        messages: list[dict[str, Any]],
        user_query: str,
        completed: bool
    ) -> list[dict[str, Any]]:
        """
        Convert internal message format to Hermes trajectory format.

        This produces the exact format used by batch_runner.py.
        """
        trajectory = []

        # System message with tool definitions
        system_msg = (
            "You are a function calling AI model. You are provided with function signatures within <tools> </tools> XML tags. "
            "You may call one or more functions to assist with the user query. If available tools are not relevant in assisting "
            "with user query, just respond in natural conversational language. Don't make assumptions about what values to plug "
            "into functions. After calling & executing the functions, you will be provided with function results within "
            "<tool_response> </tool_response> XML tags. Here are the available tools:\n"
            f"<tools>\n{self._format_tools_for_system_message()}\n</tools>\n"
            "For each function call return a JSON object, with the following pydantic model json schema for each:\n"
            "{'title': 'FunctionCall', 'type': 'object', 'properties': {'name': {'title': 'Name', 'type': 'string'}, "
            "'arguments': {'title': 'Arguments', 'type': 'object'}}, 'required': ['name', 'arguments']}\n"
            "Each function call should be enclosed within <tool_call> </tool_call> XML tags.\n"
            "Example:\n<tool_call>\n{'name': <function-name>,'arguments': <args-dict>}\n</tool_call>"
        )

        trajectory.append({"from": "system", "value": system_msg})
        trajectory.append({"from": "human", "value": user_query})

        # Process messages (skip first user message as we already added it)
        i = 1
        while i < len(messages):
            msg = messages[i]

            if msg["role"] == "assistant":
                if "tool_calls" in msg and msg["tool_calls"]:
                    # Assistant message with tool calls
                    content = ""

                    # Add reasoning if present
                    if msg.get("reasoning"):
                        content = f"<think>{msg['reasoning']}</think>"

                    if msg.get("content"):
                        content += msg["content"] + "\n"

                    # Add tool calls in XML format
                    for tool_call in msg["tool_calls"]:
                        if not tool_call or not isinstance(tool_call, dict): continue
                        try:
                            arguments = json.loads(tool_call["function"]["arguments"]) \
                                if isinstance(tool_call["function"]["arguments"], str) \
                                else tool_call["function"]["arguments"]
                        except json.JSONDecodeError:
                            arguments = {}

                        tool_call_json = {
                            "name": tool_call["function"]["name"],
                            "arguments": arguments
                        }
                        content += f"<tool_call>\n{json.dumps(tool_call_json, ensure_ascii=False)}\n</tool_call>\n"

                    trajectory.append({"from": "gpt", "value": content.rstrip()})

                    # Collect subsequent tool responses
                    tool_responses = []
                    j = i + 1
                    while j < len(messages) and messages[j]["role"] == "tool":
                        tool_msg = messages[j]
                        tool_content = tool_msg["content"]

                        # Try to parse as JSON
                        try:
                            if tool_content.strip().startswith(("{", "[")):
                                tool_content = json.loads(tool_content)
                        except (json.JSONDecodeError, AttributeError):
                            pass

                        tool_response = "<tool_response>\n"
                        tool_response += json.dumps({
                            "tool_call_id": tool_msg.get("tool_call_id", ""),
                            "name": msg["tool_calls"][len(tool_responses)]["function"]["name"] \
                                if len(tool_responses) < len(msg["tool_calls"]) else "unknown",
                            "content": tool_content
                        }, ensure_ascii=False)
                        tool_response += "\n</tool_response>"
                        tool_responses.append(tool_response)
                        j += 1

                    if tool_responses:
                        trajectory.append({"from": "tool", "value": "\n".join(tool_responses)})
                        i = j - 1

                else:
                    # Regular assistant message (no tool calls)
                    content = ""
                    if msg.get("reasoning"):
                        content = f"<think>{msg['reasoning']}</think>"
                    content += msg.get("content") or ""
                    trajectory.append({"from": "gpt", "value": content})

            elif msg["role"] == "user":
                trajectory.append({"from": "human", "value": msg["content"]})

            i += 1

        return trajectory

    async def run_task(self, task: str) -> dict[str, Any]:
        """
        Run a single task and return the result with trajectory.

        Args:
            task: The task/prompt to execute

        Returns:
            Dict with trajectory, completion status, and metadata
        """
        async with self._run_lock:
            return await self._run_task(task)

    async def _run_task(self, task: str) -> dict[str, Any]:
        print(f"\n{'='*60}")
        print(f"📝 Task: {task[:80]}{'...' if len(task) > 80 else ''}")
        print(f"{'='*60}")

        # Message history
        messages: list[dict[str, Any]] = [{"role": "user", "content": task}]

        # System prompt for the LLM (ephemeral - not saved to trajectory)
        system_prompt = """You are an AI agent that can execute bash commands to complete tasks.

When you need to run commands, use the 'terminal' tool with your bash command.

**Important:**
- When you have completed the task successfully, run: echo "MINI_SWE_AGENT_FINAL_OUTPUT" followed by a summary
- Be concise and efficient in your approach
- Install any needed tools with apt-get or pip
- Avoid interactive commands (no vim, nano, less, etc.)

Complete the user's task step by step."""

        api_call_count = 0
        completed = False
        final_response = None

        try:
            # Initialize inside the cleanup boundary so cancellation or a
            # failed async startup cannot strand an environment.
            await self._create_env()
            client = await self._ensure_client()
            while api_call_count < self.max_iterations:
                api_call_count += 1
                print(f"\n🔄 API call #{api_call_count}/{self.max_iterations}")

                # Prepare API messages
                api_messages = [{"role": "system", "content": system_prompt}] + [
                    {
                        key: value
                        for key, value in message.items()
                        if key != "reasoning"
                    }
                    for message in messages
                ]

                # Make API call
                try:
                    api_kwargs = {
                        "model": self.model,
                        "messages": api_messages,
                        "tools": self.tools,
                        "timeout": 300.0,
                    }
                    fixed_temperature = _effective_temperature_for_model(
                        self.model,
                        str(getattr(client, "base_url", "") or ""),
                    )
                    if fixed_temperature is not None:
                        api_kwargs["temperature"] = fixed_temperature

                    response = await client.chat.completions.create(**api_kwargs)
                except Exception as e:
                    self.logger.error("API call failed: %s", e)
                    break

                assistant_message = response.choices[0].message

                # Log assistant response
                if assistant_message.content:
                    print(f"🤖 Assistant: {assistant_message.content[:100]}...")

                # Check for tool calls
                if assistant_message.tool_calls:
                    print(f"🔧 Tool calls: {len(assistant_message.tool_calls)}")

                    # Add assistant message with tool calls
                    reasoning = getattr(
                        assistant_message,
                        "reasoning_content",
                        None,
                    ) or getattr(assistant_message, "reasoning", None)
                    assistant_entry: dict[str, Any] = {
                        "role": "assistant",
                        "content": assistant_message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            }
                            for tc in assistant_message.tool_calls
                        ]
                    }
                    if isinstance(reasoning, str) and reasoning:
                        assistant_entry["reasoning"] = reasoning
                    reasoning_details = getattr(
                        assistant_message,
                        "reasoning_details",
                        None,
                    )
                    if reasoning_details is not None:
                        assistant_entry["reasoning_details"] = reasoning_details
                    messages.append(assistant_entry)

                    # Execute each tool call
                    for tc in assistant_message.tool_calls:
                        try:
                            args = json.loads(tc.function.arguments)
                        except json.JSONDecodeError:
                            args = {}

                        command = args.get("command", "echo 'No command provided'")
                        timeout = args.get("timeout", self.command_timeout)

                        print(f"   📞 terminal: {command[:60]}...")

                        # Execute command
                        result = await self._execute_command(command, timeout)

                        # Format result
                        result_json = json.dumps({
                            "content": {
                                "output": result["output"],
                                "exit_code": result["exit_code"],
                                "error": result["error"]
                            }
                        }, ensure_ascii=False)

                        # Check for task completion signal
                        if "MINI_SWE_AGENT_FINAL_OUTPUT" in result["output"]:
                            print("   ✅ Task completion signal detected!")
                            completed = True

                        # Add tool response
                        messages.append(make_tool_result_message(
                            tc.function.name, result_json, tc.id,
                        ))

                        print(f"   ✅ exit_code={result['exit_code']}, output={len(result['output'])} chars")

                    # If task completed, we can stop
                    if completed:
                        final_response = assistant_message.content
                        break

                else:
                    # No tool calls - final response
                    final_response = assistant_message.content or ""
                    assistant_entry = {
                        "role": "assistant",
                        "content": final_response
                    }
                    reasoning = getattr(
                        assistant_message,
                        "reasoning_content",
                        None,
                    ) or getattr(assistant_message, "reasoning", None)
                    if isinstance(reasoning, str) and reasoning:
                        assistant_entry["reasoning"] = reasoning
                    reasoning_details = getattr(
                        assistant_message,
                        "reasoning_details",
                        None,
                    )
                    if reasoning_details is not None:
                        assistant_entry["reasoning_details"] = reasoning_details
                    messages.append(assistant_entry)
                    completed = True
                    print("🎉 Agent finished (no more tool calls)")
                    break

            if api_call_count >= self.max_iterations:
                print(f"⚠️  Reached max iterations ({self.max_iterations})")

        finally:
            # Cleanup environment
            await self._cleanup_task_resources()

        # Convert to Hermes trajectory format
        trajectory = self._convert_to_hermes_format(messages, task, completed)

        return {
            "conversations": trajectory,
            "completed": completed,
            "api_calls": api_call_count,
            "metadata": {
                "model": self.model,
                "env_type": self.env_type,
                "timestamp": datetime.now().isoformat()
            }
        }

    async def run_batch(
        self,
        prompts: list[str],
        output_file: str
    ) -> list[dict[str, Any]]:
        """
        Run multiple tasks and save trajectories to a JSONL file.

        Args:
            prompts: List of task prompts
            output_file: Output JSONL file path

        Returns:
            List of results
        """
        results = []

        print(f"\n📦 Running batch of {len(prompts)} tasks")
        print(f"📁 Output: {output_file}")

        async with aiofiles.open(output_file, "w", encoding="utf-8") as f:
            async def _write_result(value: dict[str, Any]) -> None:
                await f.write(json.dumps(value, ensure_ascii=False) + "\n")
                await f.flush()

            for i, prompt in enumerate(prompts, 1):
                print(f"\n{'='*60}")
                print(f"📋 Task {i}/{len(prompts)}")
                print(f"{'='*60}")

                try:
                    result = await self.run_task(prompt)
                    results.append(result)

                    # Write to file immediately
                    await _finish_owned_task(
                        asyncio.create_task(
                            _write_result(result),
                            name="mini-swe-runner-jsonl-write",
                        )
                    )

                    print(f"✅ Task {i} completed (api_calls={result['api_calls']})")

                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    self.logger.error("Error on task %s: %s", i, e)
                    error_result = {
                        "conversations": [],
                        "completed": False,
                        "api_calls": 0,
                        "error": str(e),
                        "metadata": {"timestamp": datetime.now().isoformat()}
                    }
                    results.append(error_result)
                    await _finish_owned_task(
                        asyncio.create_task(
                            _write_result(error_result),
                            name="mini-swe-runner-jsonl-write",
                        )
                    )

        print(f"\n✅ Batch complete! {len(results)} trajectories saved to {output_file}")
        return results
