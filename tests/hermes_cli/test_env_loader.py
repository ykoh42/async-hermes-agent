import codecs
import importlib
import os
import sys

import pytest

from hermes_cli.env_loader import load_hermes_dotenv











# ---------------------------------------------------------------------------
# UTF-16 / UTF-32 .env sanitizer coverage
#
# Scope note: intentionally NO UTF-8-BOM assertions here. UTF-8 BOM handling
# for _load_dotenv_with_fallback is #65124's un-merged fix; a test here would
# couple the PRs. This suite covers only the sanitizer rewrite path for
# UTF-16/32 (and UTF-8 / cp1252 regression guards for that path).
# ---------------------------------------------------------------------------


def _assert_clean_utf8_env_on_disk(env_file, *, first_key: str) -> None:
    """On-disk file must be clean UTF-8: no BOM, no U+FFFD, canonical key."""
    after = env_file.read_bytes()
    assert not after.startswith(codecs.BOM_UTF8)
    assert not after.startswith(codecs.BOM_UTF16_LE)
    assert not after.startswith(codecs.BOM_UTF16_BE)
    text = after.decode("utf-8")  # strict — raises if not clean UTF-8
    assert "\ufffd" not in text
    assert text.startswith(f"{first_key}=") or f"\n{first_key}=" in text
    assert first_key.encode("ascii") in after




@pytest.mark.asyncio
async def test_utf16_le_bom_preserves_non_ascii_values(tmp_path, monkeypatch):
    """UTF-16-LE+BOM rewrite must preserve non-ASCII values (not just ASCII keys).

    Uses non-credential var names so _sanitize_loaded_credentials does not
    strip non-ASCII from values (that path only targets *_KEY/*_TOKEN/etc.).
    """
    home = tmp_path / "hermes"
    home.mkdir()
    env_file = home / ".env"
    content = "GREETING=café\nCJK_LABEL=日本語\n"
    env_file.write_bytes(codecs.BOM_UTF16_LE + content.encode("utf-16-le"))

    monkeypatch.delenv("GREETING", raising=False)
    monkeypatch.delenv("CJK_LABEL", raising=False)

    loaded = await load_hermes_dotenv(hermes_home=home)

    assert loaded == [env_file]
    assert os.getenv("GREETING") == "café"
    assert os.getenv("CJK_LABEL") == "日本語"
    after = env_file.read_bytes()
    assert after.decode("utf-8")  # strict
    assert "café".encode() in after
    assert "日本語".encode() in after
    assert b"\xef\xbf\xbd" not in after


@pytest.mark.asyncio
async def test_utf32_le_bom_leaves_file_untouched(tmp_path, caplog):
    """UTF-32-LE BOM: refuse-to-mangle (leave bytes untouched + warning).

    UTF-32-LE's BOM starts with UTF-16-LE's FF FE; sniff order must check
    UTF-32 first so we never misdetect and corrupt.

    Exercises ``_sanitize_env_file_if_needed`` only: the dotenv load path
    is out of scope here (#65124's surface) and still cannot ingest UTF-32.
    """
    import logging

    from hermes_cli.env_loader import _sanitize_env_file_if_needed

    env_file = tmp_path / ".env"
    content = "HERMES_TEST_KEY=hello_utf32\nSECOND_KEY=world\n"
    raw = codecs.BOM_UTF32_LE + content.encode("utf-32-le")
    env_file.write_bytes(raw)

    with caplog.at_level(logging.WARNING, logger="hermes_cli.env_loader"):
        await _sanitize_env_file_if_needed(env_file)

    assert env_file.read_bytes() == raw  # untouched
    assert any("UTF-32" in r.message for r in caplog.records)




@pytest.mark.asyncio
async def test_utf32_warning_fires_once_per_path(tmp_path, caplog, monkeypatch):
    """Three sanitize calls on the same UTF-32 file → exactly one warning.

    Matches house style for warn-once (module-level seen-set, same class as
    ``_WARNED_KEYS``): hot-reload / multi-entry load must not spam logs.
    """
    import logging

    import hermes_cli.env_loader as env_loader
    from hermes_cli.env_loader import _sanitize_env_file_if_needed

    # Isolate process-level seen-set so other tests' paths don't leak in.
    monkeypatch.setattr(env_loader, "_WARNED_UTF32_PATHS", set())

    env_file = tmp_path / ".env"
    content = "HERMES_TEST_KEY=hello_utf32\nSECOND_KEY=world\n"
    raw = codecs.BOM_UTF32_LE + content.encode("utf-32-le")
    env_file.write_bytes(raw)

    with caplog.at_level(logging.WARNING, logger="hermes_cli.env_loader"):
        await _sanitize_env_file_if_needed(env_file)
        await _sanitize_env_file_if_needed(env_file)
        await _sanitize_env_file_if_needed(env_file)

    utf32_warnings = [r for r in caplog.records if "UTF-32" in r.message]
    assert len(utf32_warnings) == 1
    assert env_file.read_bytes() == raw




@pytest.mark.asyncio
async def test_plain_utf8_env_regression(tmp_path, monkeypatch):
    """Plain UTF-8 .env must keep loading after the UTF-16 sanitize changes."""
    home = tmp_path / "hermes"
    home.mkdir()
    env_file = home / ".env"
    before = b"OPENAI_API_KEY=sk-plain\nSECOND_KEY=ok\n"
    env_file.write_bytes(before)

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SECOND_KEY", raising=False)

    loaded = await load_hermes_dotenv(hermes_home=home)

    assert loaded == [env_file]
    assert os.getenv("OPENAI_API_KEY") == "sk-plain"
    assert os.getenv("SECOND_KEY") == "ok"
    # No spurious rewrite of an already-clean file.
    assert env_file.read_bytes() == before


@pytest.mark.asyncio
async def test_cp1252_env_regression_does_not_crash(tmp_path, monkeypatch):
    """cp1252/latin-1 body must not crash sanitize; ASCII keys still usable.

    0xE9 is 'é' in cp1252 and incomplete as UTF-8. First line does not begin
    with U+FFFD, so the FFFD guard must not refuse the whole file.

    Sanitize leaves the file bytes alone when the only "change" is
    errors=replace on values (original already replace-decoded equals
    sanitized), so _load_dotenv_with_fallback's latin-1 path recovers café.
    """
    home = tmp_path / "hermes"
    home.mkdir()
    env_file = home / ".env"
    before = b"ASCII_KEY=ok\nLATIN1_VALUE=caf\xe9\n"
    env_file.write_bytes(before)

    monkeypatch.delenv("ASCII_KEY", raising=False)
    monkeypatch.delenv("LATIN1_VALUE", raising=False)

    loaded = await load_hermes_dotenv(hermes_home=home)

    assert loaded == [env_file]
    assert os.getenv("ASCII_KEY") == "ok"
    assert os.getenv("LATIN1_VALUE") == "café"
    # Sanitize must not have rewritten (would have persisted U+FFFD).
    assert env_file.read_bytes() == before


# ---------------------------------------------------------------------------
# config.yaml terminal.* re-apply after dotenv loads.
# Explicit local terminal settings in config.yaml remain authoritative over
# stale values in .env or the parent shell.
# ---------------------------------------------------------------------------


def _seed_terminal_home(tmp_path, monkeypatch, *, config_yaml=None, env_text=None):
    home = tmp_path / "hermes"
    home.mkdir()
    if config_yaml is not None:
        (home / "config.yaml").write_text(config_yaml, encoding="utf-8")
    if env_text is not None:
        (home / ".env").write_text(env_text, encoding="utf-8")
    # The bridge is scoped to the process HERMES_HOME, so point the process at
    # the seeded home.
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@pytest.mark.asyncio
async def test_config_yaml_terminal_cwd_overrides_stale_env(tmp_path, monkeypatch):
    home = _seed_terminal_home(
        tmp_path, monkeypatch,
        config_yaml="terminal:\n  cwd: /configured/workspace\n",
        env_text="TERMINAL_CWD=/stale/workspace\n",
    )

    monkeypatch.delenv("TERMINAL_CWD", raising=False)

    await load_hermes_dotenv(hermes_home=home)

    assert os.getenv("TERMINAL_CWD") == "/configured/workspace"


@pytest.mark.asyncio
async def test_config_yaml_terminal_timeout_overrides_stale_shell(tmp_path, monkeypatch):
    home = _seed_terminal_home(
        tmp_path, monkeypatch,
        config_yaml="terminal:\n  timeout: 600\n",
    )

    monkeypatch.setenv("TERMINAL_TIMEOUT", "30")

    await load_hermes_dotenv(hermes_home=home)

    assert os.getenv("TERMINAL_TIMEOUT") == "600"


@pytest.mark.asyncio
async def test_no_terminal_section_leaves_env_value_alone(tmp_path, monkeypatch):
    """When config.yaml has no terminal section, the .env value is still the
    user's active setting — the bridge must NOT clobber it with merged
    defaults."""
    home = _seed_terminal_home(
        tmp_path, monkeypatch,
        config_yaml="display:\n  show_commentary: true\n",
        env_text="TERMINAL_TIMEOUT=45\n",
    )

    monkeypatch.delenv("TERMINAL_TIMEOUT", raising=False)

    await load_hermes_dotenv(hermes_home=home)

    assert os.getenv("TERMINAL_TIMEOUT") == "45"


@pytest.mark.asyncio
async def test_config_yaml_terminal_omitted_key_does_not_clear_env(
    tmp_path, monkeypatch
):
    """If config.yaml has a terminal block but omits a key, the .env value
    must survive (only explicit config keys override env)."""
    home = _seed_terminal_home(
        tmp_path, monkeypatch,
        config_yaml="terminal:\n  cwd: /configured/workspace\n",
        env_text="TERMINAL_TIMEOUT=45\n",
    )

    monkeypatch.delenv("TERMINAL_TIMEOUT", raising=False)

    await load_hermes_dotenv(hermes_home=home)

    assert os.getenv("TERMINAL_TIMEOUT") == "45"
    assert os.getenv("TERMINAL_CWD") == "/configured/workspace"


@pytest.mark.asyncio
async def test_other_profile_home_does_not_bridge_process_config(
    tmp_path, monkeypatch
):
    """Loading a DIFFERENT profile's .env must not re-bridge this process's
    config.yaml — the shared bridge reads the process-global config, so
    applying it for another home would stamp the wrong profile's terminal
    settings into the env."""
    process_home = tmp_path / "process-home"
    process_home.mkdir()
    (process_home / "config.yaml").write_text(
        "terminal:\n  timeout: 600\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(process_home))

    other_home = tmp_path / "other-profile"
    other_home.mkdir()
    (other_home / ".env").write_text("TERMINAL_TIMEOUT=45\n", encoding="utf-8")

    monkeypatch.delenv("TERMINAL_TIMEOUT", raising=False)

    await load_hermes_dotenv(hermes_home=other_home)

    # The other profile's .env value stands; the process config was not applied.
    assert os.getenv("TERMINAL_TIMEOUT") == "45"
