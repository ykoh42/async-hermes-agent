"""A transient read failure on auth.json must not degrade to an empty store.

``_load_auth_store`` treated every exception as corruption and returned
``{"version": ..., "providers": {}}``. This module does read-modify-write in
roughly fifteen places, so an ``OSError`` (EMFILE under fd exhaustion, EACCES,
EIO, a stalled mount) followed by any ``_save_auth_store`` rewrote auth.json
with an empty provider set and destroyed every stored credential.

Genuine corruption still degrades, still preserves a copy, and now only claims
to have preserved one when the copy actually landed.
"""

import errno
import json
import logging

import pytest

import hermes_cli.auth as auth


@pytest.fixture
def store_file(tmp_path):
    f = tmp_path / "auth.json"
    f.write_text(
        json.dumps({"version": 1, "providers": {"nous": {"api_key": "secret"}}}),
        encoding="utf-8",
    )
    return f


class _FailingAsyncContext:
    def __init__(self, exc):
        self._exc = exc

    async def __aenter__(self):
        raise self._exc

    async def __aexit__(self, *_args):
        return False


@pytest.mark.parametrize(
    "exc",
    [
        OSError(errno.EMFILE, "Too many open files"),
        PermissionError(errno.EACCES, "Permission denied"),
        OSError(errno.EIO, "Input/output error"),
    ],
    ids=["emfile", "eacces", "eio"],
)
@pytest.mark.asyncio
async def test_read_failure_raises_and_leaves_the_store_alone(
    store_file, monkeypatch, exc
):
    before = store_file.read_bytes()
    real_open = auth.aiofiles.open

    def failing_open(path, *args, **kwargs):
        if path == store_file:
            return _FailingAsyncContext(exc)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(auth.aiofiles, "open", failing_open)

    with pytest.raises(OSError):
        await auth._load_auth_store(store_file)

    assert store_file.read_bytes() == before, "the store on disk was modified"
    assert not store_file.with_suffix(".json.corrupt").exists(), (
        "a read failure is not corruption and must not write a .corrupt sidecar"
    )


@pytest.mark.asyncio
async def test_unparseable_json_still_degrades_and_preserves_a_copy(store_file):
    store_file.write_text("{ not json", encoding="utf-8")

    result = await auth._load_auth_store(store_file)

    assert result == {"version": auth.AUTH_STORE_VERSION, "providers": {}}
    corrupt = store_file.with_suffix(".json.corrupt")
    assert corrupt.exists(), "genuine corruption must still be preserved"
    assert corrupt.read_text(encoding="utf-8") == "{ not json"


@pytest.mark.asyncio
async def test_healthy_store_is_returned_unchanged(store_file):
    result = await auth._load_auth_store(store_file)
    assert result["providers"]["nous"]["api_key"] == "secret"


@pytest.mark.asyncio
async def test_log_does_not_claim_a_backup_that_was_not_written(
    store_file, monkeypatch, caplog
):
    """The old message advertised the .corrupt path even when copy2 failed."""
    store_file.write_text("{ not json", encoding="utf-8")

    real_open = auth.aiofiles.open
    corrupt_path = store_file.with_suffix(".json.corrupt")

    def failing_backup_open(path, *args, **kwargs):
        if path == corrupt_path:
            return _FailingAsyncContext(OSError(errno.EMFILE, "Too many open files"))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(auth.aiofiles, "open", failing_backup_open)

    with caplog.at_level(logging.WARNING, logger="hermes_cli.auth"):
        result = await auth._load_auth_store(store_file)

    assert result == {"version": auth.AUTH_STORE_VERSION, "providers": {}}
    assert not store_file.with_suffix(".json.corrupt").exists()
    text = caplog.text
    assert "could NOT be preserved" in text
    assert "Corrupt file preserved at" not in text
