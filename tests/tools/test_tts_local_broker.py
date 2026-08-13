"""Persistent local-TTS broker lifecycle and isolation coverage."""

from __future__ import annotations

import asyncio
import gc
import os
import weakref
from pathlib import Path

import aiofiles
import aiofiles.os
import pytest

from hermes_constants import (
    reset_hermes_home_override,
    set_hermes_home_override,
)
from agent.subagent_lifecycle import bind_subagent_parent
from run_agent import AIAgent
from tools import tts_tool


class _Owner:
    pass


async def _under_home(home: Path, call, *args):
    token = set_hermes_home_override(home)
    try:
        return await call(*args)
    finally:
        reset_hermes_home_override(token)


async def _install_fake_sdks(root: Path) -> None:
    async with aiofiles.open(root / "piper.py", "w") as module:
        await module.write(
            "import os\n"
            "class SynthesisConfig:\n"
            "    def __init__(self, **kwargs): self.kwargs = kwargs\n"
            "class Voice:\n"
            "    def synthesize_wav(self, text, output, syn_config=None):\n"
            "        output.setnchannels(1); output.setsampwidth(2); "
            "output.setframerate(22050); output.writeframes(b'\\0\\0' * 32)\n"
            "class PiperVoice:\n"
            "    @classmethod\n"
            "    def load(cls, model_path, use_cuda=False):\n"
            "        with open(os.environ['TTS_FAKE_LOAD_LOG'], 'a') as log:\n"
            "            log.write(f'piper:{model_path}:cuda={use_cuda}:' "
            "+ os.environ.get('HERMES_HOME', '') + '\\n')\n"
            "        return Voice()\n"
        )
    async with aiofiles.open(root / "kittentts.py", "w") as module:
        await module.write(
            "import os, time\n"
            "class KittenTTS:\n"
            "    def __init__(self, model):\n"
            "        self.model = model\n"
            "        with open(os.environ['TTS_FAKE_LOAD_LOG'], 'a') as log:\n"
            "            log.write(f'kitten:{model}:' "
            "+ os.environ.get('HERMES_HOME', '') + '\\n')\n"
            "        secret_log = os.environ.get('TTS_FAKE_SECRET_LOG')\n"
            "        if secret_log:\n"
            "            with open(secret_log, 'a') as log:\n"
            "                log.write(os.environ.get('OPENAI_API_KEY', 'missing') "
            "+ '\\n')\n"
            "    def generate(self, text, **kwargs):\n"
            "        marker = os.environ.get('TTS_FAKE_STARTED')\n"
            "        if text == 'value-error':\n"
            "            raise ValueError('bad local setting')\n"
            "        if text == 'oversized-stdout':\n"
            "            print('x' * (1024 * 1024 + 1024), flush=True)\n"
            "        if text == 'hard-exit':\n"
            "            os._exit(17)\n"
            "        if text == 'hang':\n"
            "            if marker:\n"
            "                open(marker, 'w').close()\n"
            "            time.sleep(60)\n"
            "        return [0.0] * 32\n"
        )
    async with aiofiles.open(root / "soundfile.py", "w") as module:
        await module.write(
            "def write(path, audio, samplerate):\n"
            "    with open(path, 'wb') as output:\n"
            "        output.write(b'RIFF\\0\\0\\0\\0WAVE')\n"
        )


def _capture_broker_children(monkeypatch):
    original_create = asyncio.create_subprocess_exec
    children = []

    async def capture_child(*args, **kwargs):
        process = await original_create(*args, **kwargs)
        if any(str(arg).endswith("local_tts_synth.py") for arg in args):
            children.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", capture_child)
    return children


@pytest.mark.asyncio
async def test_agent_lease_reuses_exact_provider_lrus_and_final_close(
    tmp_path,
    monkeypatch,
):
    sdk = tmp_path / "sdk"
    home = tmp_path / "home"
    await aiofiles.os.makedirs(sdk)
    await aiofiles.os.makedirs(home)
    await _install_fake_sdks(sdk)
    load_log = tmp_path / "loads.log"
    monkeypatch.setenv("PYTHONPATH", str(sdk))
    monkeypatch.setenv("TTS_FAKE_LOAD_LOG", str(load_log))
    children = _capture_broker_children(monkeypatch)
    owner = _Owner()
    await _under_home(home, tts_tool._retain_local_tts_lifecycle, owner)

    models = []
    for index in range(4):
        model = tmp_path / f"voice-{index}.onnx"
        async with aiofiles.open(model, "wb") as model_file:
            await model_file.write(b"model")
        models.append(model)

    for index, model_index in enumerate((0, 1, 2, 0, 3, 1)):
        await _under_home(
            home,
            tts_tool._generate_piper_tts,
            "hello",
            str(tmp_path / f"piper-{index}.wav"),
            {"piper": {"voice": str(models[model_index])}},
        )
    for index, model_index in enumerate((0, 1, 2, 0, 3, 1)):
        await _under_home(
            home,
            tts_tool._generate_kittentts,
            "hello",
            str(tmp_path / f"kitten-{index}.wav"),
            {"kittentts": {"model": f"model-{model_index}"}},
        )

    assert len(children) == 1
    assert children[0].returncode is None
    async with aiofiles.open(load_log) as log_file:
        lines = (await log_file.read()).splitlines()
    assert [line.split(":", 2)[:2] for line in lines] == [
        ["piper", str(models[0])],
        ["piper", str(models[1])],
        ["piper", str(models[2])],
        ["piper", str(models[3])],
        ["piper", str(models[1])],
        ["kitten", "model-0"],
        ["kitten", "model-1"],
        ["kitten", "model-2"],
        ["kitten", "model-3"],
        ["kitten", "model-1"],
    ]

    await _under_home(home, tts_tool._release_local_tts_lifecycle, owner)
    assert children[0].returncode is not None
    assert not tts_tool._local_tts_scope_states


@pytest.mark.asyncio
async def test_profiles_get_distinct_workers_and_symlinked_home_shares_one(
    tmp_path,
    monkeypatch,
):
    sdk = tmp_path / "sdk"
    home_a = tmp_path / "home-a"
    home_b = tmp_path / "home-b"
    alias_a = tmp_path / "alias-a"
    for directory in (sdk, home_a, home_b):
        await aiofiles.os.makedirs(directory)
    await aiofiles.os.symlink(home_a, alias_a, target_is_directory=True)
    await _install_fake_sdks(sdk)
    load_log = tmp_path / "loads.log"
    monkeypatch.setenv("PYTHONPATH", str(sdk))
    monkeypatch.setenv("TTS_FAKE_LOAD_LOG", str(load_log))
    children = _capture_broker_children(monkeypatch)
    owner_a = _Owner()
    owner_alias = _Owner()
    owner_b = _Owner()
    await _under_home(home_a, tts_tool._retain_local_tts_lifecycle, owner_a)
    await _under_home(alias_a, tts_tool._retain_local_tts_lifecycle, owner_alias)
    await _under_home(home_b, tts_tool._retain_local_tts_lifecycle, owner_b)

    await asyncio.gather(
        _under_home(
            home_a,
            tts_tool._generate_kittentts,
            "hello",
            str(tmp_path / "a.wav"),
            {"kittentts": {"model": "same"}},
        ),
        _under_home(
            home_b,
            tts_tool._generate_kittentts,
            "hello",
            str(tmp_path / "b.wav"),
            {"kittentts": {"model": "same"}},
        ),
    )
    await _under_home(
        alias_a,
        tts_tool._generate_kittentts,
        "hello",
        str(tmp_path / "alias.wav"),
        {"kittentts": {"model": "same"}},
    )

    assert len(children) == 2
    async with aiofiles.open(load_log) as log_file:
        lines = (await log_file.read()).splitlines()
    assert len(lines) == 2
    assert {line.rsplit(":", 1)[-1] for line in lines} == {
        str(home_a),
        str(home_b),
    }

    await _under_home(alias_a, tts_tool._release_local_tts_lifecycle, owner_a)
    assert sum(child.returncode is None for child in children) == 2
    await _under_home(home_a, tts_tool._release_local_tts_lifecycle, owner_alias)
    assert sum(child.returncode is None for child in children) == 1
    await _under_home(home_b, tts_tool._release_local_tts_lifecycle, owner_b)
    assert all(child.returncode is not None for child in children)


@pytest.mark.asyncio
async def test_standalone_request_closes_worker_deterministically(
    tmp_path,
    monkeypatch,
):
    sdk = tmp_path / "sdk"
    home = tmp_path / "home"
    await aiofiles.os.makedirs(sdk)
    await aiofiles.os.makedirs(home)
    await _install_fake_sdks(sdk)
    monkeypatch.setenv("PYTHONPATH", str(sdk))
    monkeypatch.setenv("TTS_FAKE_LOAD_LOG", str(tmp_path / "loads.log"))
    children = _capture_broker_children(monkeypatch)

    output = tmp_path / "standalone.wav"
    await _under_home(
        home,
        tts_tool._generate_kittentts,
        "hello",
        str(output),
        {},
    )

    assert len(children) == 1
    assert children[0].returncode is not None
    async with aiofiles.open(output, "rb") as generated:
        assert (await generated.read()).startswith(b"RIFF")
    assert not tts_tool._local_tts_scope_states
    assert not [
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task()
        and not task.done()
        and task.get_name().startswith("local-tts-")
    ]


@pytest.mark.asyncio
async def test_worker_spawn_failure_rolls_back_scope_state(tmp_path, monkeypatch):
    home = tmp_path / "home"
    await aiofiles.os.makedirs(home)

    async def fail_spawn(*_args, **_kwargs):
        raise OSError("spawn failed")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fail_spawn)
    with pytest.raises(OSError, match="spawn failed"):
        await _under_home(
            home,
            tts_tool._run_local_tts_synth,
            "kittentts",
            {
                "text": "hello",
                "output_path": str(tmp_path / "never.wav"),
                "model": "model",
            },
        )

    assert not tts_tool._local_tts_scope_states
    assert not [
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task()
        and not task.done()
        and task.get_name().startswith("local-tts-")
    ]


@pytest.mark.asyncio
async def test_worker_error_preserves_upstream_class_and_safe_public_message(
    tmp_path,
    monkeypatch,
):
    sdk = tmp_path / "sdk"
    home = tmp_path / "home"
    await aiofiles.os.makedirs(sdk)
    await aiofiles.os.makedirs(home)
    await _install_fake_sdks(sdk)
    monkeypatch.setenv("PYTHONPATH", str(sdk))
    monkeypatch.setenv("TTS_FAKE_LOAD_LOG", str(tmp_path / "loads.log"))
    children = _capture_broker_children(monkeypatch)

    async def config():
        return {"provider": "kittentts"}

    async def available():
        return True

    monkeypatch.setattr(tts_tool, "_load_tts_config", config)
    monkeypatch.setattr(tts_tool, "_check_kittentts_available", available)
    token = set_hermes_home_override(home)
    try:
        response = await tts_tool.text_to_speech_tool(
            "value-error",
            output_path=str(tmp_path / "error.wav"),
        )
    finally:
        reset_hermes_home_override(token)

    assert '"success": false' in response
    assert "TTS configuration error (kittentts): bad local setting" in response
    assert "Traceback" not in response
    assert len(children) == 1
    assert children[0].returncode is not None
    assert not tts_tool._local_tts_scope_states


@pytest.mark.asyncio
async def test_sdk_stdout_is_redirected_without_corrupting_protocol(
    tmp_path,
    monkeypatch,
):
    sdk = tmp_path / "sdk"
    home = tmp_path / "home"
    await aiofiles.os.makedirs(sdk)
    await aiofiles.os.makedirs(home)
    await _install_fake_sdks(sdk)
    monkeypatch.setenv("PYTHONPATH", str(sdk))
    monkeypatch.setenv("TTS_FAKE_LOAD_LOG", str(tmp_path / "loads.log"))
    children = _capture_broker_children(monkeypatch)

    output = tmp_path / "protocol.wav"
    async with asyncio.timeout(5):
        assert await _under_home(
            home,
            tts_tool._generate_kittentts,
            "oversized-stdout",
            str(output),
            {},
        ) == str(output)

    assert len(children) == 1
    assert children[0].returncode is not None
    async with aiofiles.open(output, "rb") as generated:
        assert (await generated.read()).startswith(b"RIFF")
    assert not tts_tool._local_tts_scope_states
    assert not [
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task()
        and not task.done()
        and task.get_name().startswith("local-tts-")
    ]


@pytest.mark.asyncio
async def test_worker_environment_scrubs_model_provider_credentials(
    tmp_path,
    monkeypatch,
):
    sdk = tmp_path / "sdk"
    home = tmp_path / "home"
    await aiofiles.os.makedirs(sdk)
    await aiofiles.os.makedirs(home)
    await _install_fake_sdks(sdk)
    secret_log = tmp_path / "secret.log"
    monkeypatch.setenv("PYTHONPATH", str(sdk))
    monkeypatch.setenv("TTS_FAKE_LOAD_LOG", str(tmp_path / "loads.log"))
    monkeypatch.setenv("TTS_FAKE_SECRET_LOG", str(secret_log))
    monkeypatch.setenv("OPENAI_API_KEY", "foreign-profile-secret")

    await _under_home(
        home,
        tts_tool._generate_kittentts,
        "hello",
        str(tmp_path / "safe.wav"),
        {},
    )

    async with aiofiles.open(secret_log) as log:
        assert (await log.read()).strip() == "missing"
    assert not tts_tool._local_tts_scope_states


@pytest.mark.asyncio
async def test_unexpected_worker_exit_is_reaped_and_next_request_recovers(
    tmp_path,
    monkeypatch,
):
    sdk = tmp_path / "sdk"
    home = tmp_path / "home"
    await aiofiles.os.makedirs(sdk)
    await aiofiles.os.makedirs(home)
    await _install_fake_sdks(sdk)
    monkeypatch.setenv("PYTHONPATH", str(sdk))
    monkeypatch.setenv("TTS_FAKE_LOAD_LOG", str(tmp_path / "loads.log"))
    children = _capture_broker_children(monkeypatch)
    owner = _Owner()
    await _under_home(home, tts_tool._retain_local_tts_lifecycle, owner)

    with pytest.raises(RuntimeError, match="worker exited unexpectedly"):
        await _under_home(
            home,
            tts_tool._generate_kittentts,
            "hard-exit",
            str(tmp_path / "failed.wav"),
            {},
        )
    assert len(children) == 1
    assert children[0].returncode == 17

    recovered = tmp_path / "recovered.wav"
    assert await _under_home(
        home,
        tts_tool._generate_kittentts,
        "hello",
        str(recovered),
        {},
    ) == str(recovered)
    assert len(children) == 2
    assert children[1].returncode is None

    await _under_home(home, tts_tool._release_local_tts_lifecycle, owner)
    assert children[1].returncode is not None
    assert not tts_tool._local_tts_scope_states
    assert not [
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task()
        and not task.done()
        and task.get_name().startswith("local-tts-")
    ]


@pytest.mark.asyncio
async def test_cancelled_idle_dead_worker_cleanup_drops_broker_from_scope(
    tmp_path,
    monkeypatch,
):
    sdk = tmp_path / "sdk"
    home = tmp_path / "home"
    await aiofiles.os.makedirs(sdk)
    await aiofiles.os.makedirs(home)
    await _install_fake_sdks(sdk)
    monkeypatch.setenv("PYTHONPATH", str(sdk))
    monkeypatch.setenv("TTS_FAKE_LOAD_LOG", str(tmp_path / "loads.log"))
    children = _capture_broker_children(monkeypatch)
    owner = _Owner()
    await _under_home(home, tts_tool._retain_local_tts_lifecycle, owner)
    await _under_home(
        home,
        tts_tool._generate_kittentts,
        "hello",
        str(tmp_path / "first.wav"),
        {},
    )
    _, state = await _under_home(home, tts_tool._activate_local_tts_scope)
    broker = state.broker
    assert broker is not None
    children[0].kill()
    await children[0].wait()
    await broker._finish_readers()

    original_close = broker.close
    close_started = asyncio.Event()
    allow_close = asyncio.Event()

    async def delayed_close():
        close_started.set()
        await allow_close.wait()
        await original_close()

    monkeypatch.setattr(broker, "close", delayed_close)
    request = asyncio.create_task(
        _under_home(
            home,
            tts_tool._generate_kittentts,
            "again",
            str(tmp_path / "cancelled.wav"),
            {},
        )
    )
    await close_started.wait()
    request.cancel("cancel idle cleanup")
    allow_close.set()
    with pytest.raises(asyncio.CancelledError, match="cancel idle cleanup"):
        await request

    assert state.broker is None
    assert broker._stdout_task.done()
    assert broker._stderr_task.done()
    assert len(children) == 1
    await _under_home(home, tts_tool._release_local_tts_lifecycle, owner)
    assert not tts_tool._local_tts_scope_states


@pytest.mark.asyncio
async def test_repeated_cancellation_of_final_release_still_reaps_worker(
    tmp_path,
    monkeypatch,
):
    sdk = tmp_path / "sdk"
    home = tmp_path / "home"
    await aiofiles.os.makedirs(sdk)
    await aiofiles.os.makedirs(home)
    await _install_fake_sdks(sdk)
    monkeypatch.setenv("PYTHONPATH", str(sdk))
    monkeypatch.setenv("TTS_FAKE_LOAD_LOG", str(tmp_path / "loads.log"))
    children = _capture_broker_children(monkeypatch)
    owner = _Owner()
    await _under_home(home, tts_tool._retain_local_tts_lifecycle, owner)
    await _under_home(
        home,
        tts_tool._generate_kittentts,
        "hello",
        str(tmp_path / "owned.wav"),
        {},
    )
    _, state = await _under_home(home, tts_tool._activate_local_tts_scope)
    broker = state.broker
    assert broker is not None
    original_close = broker.close
    close_started = asyncio.Event()
    allow_close = asyncio.Event()

    async def delayed_close():
        close_started.set()
        await allow_close.wait()
        await original_close()

    monkeypatch.setattr(broker, "close", delayed_close)
    release = asyncio.create_task(
        _under_home(home, tts_tool._release_local_tts_lifecycle, owner)
    )
    await close_started.wait()
    release.cancel("first")
    await asyncio.sleep(0)
    release.cancel("second")
    allow_close.set()
    with pytest.raises(asyncio.CancelledError) as cancelled:
        await release

    assert cancelled.value.args == ("first",)
    assert len(children) == 1
    assert children[0].returncode is not None
    assert broker._stdout_task.done()
    assert broker._stderr_task.done()
    assert not tts_tool._local_tts_scope_states
    assert not tts_tool._local_tts_owner_scopes


@pytest.mark.asyncio
async def test_active_aiagent_lazily_owns_worker_until_agent_close(
    tmp_path,
    monkeypatch,
):
    sdk = tmp_path / "sdk"
    home = tmp_path / "home"
    await aiofiles.os.makedirs(sdk)
    await aiofiles.os.makedirs(home)
    await _install_fake_sdks(sdk)
    monkeypatch.setenv("PYTHONPATH", str(sdk))
    monkeypatch.setenv("TTS_FAKE_LOAD_LOG", str(tmp_path / "loads.log"))
    children = _capture_broker_children(monkeypatch)
    agent = AIAgent.__new__(AIAgent)
    token = set_hermes_home_override(home)
    try:
        with bind_subagent_parent(agent):
            await tts_tool._generate_kittentts(
                "hello",
                str(tmp_path / "first.wav"),
                {},
            )
            await tts_tool._generate_kittentts(
                "again",
                str(tmp_path / "second.wav"),
                {},
            )
        assert agent._local_tts_lifecycle_retained is True
        assert len(children) == 1
        assert children[0].returncode is None

        await agent.close()
    finally:
        reset_hermes_home_override(token)

    assert agent._local_tts_lifecycle_retained is False
    assert children[0].returncode is not None
    assert not tts_tool._local_tts_scope_states


@pytest.mark.asyncio
async def test_repeated_cancellation_kills_reaps_and_preserves_first_cancel(
    tmp_path,
    monkeypatch,
):
    sdk = tmp_path / "sdk"
    home = tmp_path / "home"
    await aiofiles.os.makedirs(sdk)
    await aiofiles.os.makedirs(home)
    await _install_fake_sdks(sdk)
    started = tmp_path / "started"
    monkeypatch.setenv("PYTHONPATH", str(sdk))
    monkeypatch.setenv("TTS_FAKE_LOAD_LOG", str(tmp_path / "loads.log"))
    monkeypatch.setenv("TTS_FAKE_STARTED", str(started))
    children = _capture_broker_children(monkeypatch)
    owner = _Owner()
    await _under_home(home, tts_tool._retain_local_tts_lifecycle, owner)
    original_terminate = tts_tool._terminate_command_tts_process_tree
    cleanup_started = asyncio.Event()
    allow_cleanup = asyncio.Event()

    async def delayed_terminate(process):
        cleanup_started.set()
        await allow_cleanup.wait()
        await original_terminate(process)

    monkeypatch.setattr(
        tts_tool,
        "_terminate_command_tts_process_tree",
        delayed_terminate,
    )
    task = asyncio.create_task(
        _under_home(
            home,
            tts_tool._generate_kittentts,
            "hang",
            str(tmp_path / "cancelled.wav"),
            {},
        )
    )
    while not await aiofiles.os.path.exists(started):
        await asyncio.sleep(0)
    _, state = await _under_home(home, tts_tool._activate_local_tts_scope)
    broker = state.broker
    assert broker is not None

    task.cancel("first")
    await cleanup_started.wait()
    task.cancel("second")
    allow_cleanup.set()
    with pytest.raises(asyncio.CancelledError) as cancelled:
        await task

    assert cancelled.value.args == ("first",)
    assert len(children) == 1
    assert children[0].returncode is not None
    assert broker._stdout_task.done()
    assert broker._stderr_task.done()
    assert state.broker is None
    await _under_home(home, tts_tool._release_local_tts_lifecycle, owner)
    assert not tts_tool._local_tts_scope_states


def test_sequential_event_loops_do_not_retain_worker_or_loop(
    tmp_path,
    monkeypatch,
):
    sdk = tmp_path / "sdk"
    home = tmp_path / "home"
    sdk.mkdir()
    home.mkdir()
    asyncio.run(_install_fake_sdks(sdk))
    monkeypatch.setenv("PYTHONPATH", str(sdk))
    monkeypatch.setenv("TTS_FAKE_LOAD_LOG", str(tmp_path / "loads.log"))
    children = _capture_broker_children(monkeypatch)
    loop_refs = []

    async def one_run(index: int) -> None:
        loop_refs.append(weakref.ref(asyncio.get_running_loop()))
        owner = _Owner()
        await _under_home(home, tts_tool._retain_local_tts_lifecycle, owner)
        await _under_home(
            home,
            tts_tool._generate_kittentts,
            "hello",
            str(tmp_path / f"loop-{index}.wav"),
            {},
        )
        await _under_home(home, tts_tool._release_local_tts_lifecycle, owner)

    asyncio.run(one_run(1))
    asyncio.run(one_run(2))

    assert len(children) == 2
    assert all(child.returncode is not None for child in children)
    assert not tts_tool._local_tts_scope_states
    children.clear()
    gc.collect()
    assert all(loop_ref() is None for loop_ref in loop_refs)
