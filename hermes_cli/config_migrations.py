"""Table-driven config migration registry.

This module holds the per-version migration steps that used to live as a
768-line ladder of ``if current_ver < N:`` blocks inside
``hermes_cli.config.migrate_config``. Each step is a function
``_migrate_to_N(results, quiet)`` whose body is copied verbatim from the
original block; only the shared skeleton (the version gate and the strict
ascending ordering) lives in the :func:`run_migrations` driver.

Semantics preserved exactly from the original ladder:

* ``current_ver`` is computed ONCE by the caller (``check_config_version``)
  and never advances while the ladder runs — every step compares against the
  same initial value. The driver replicates that: it applies every registry
  entry whose target version is ``> current_ver``, in ascending order.
* Each step re-reads the raw on-disk config itself (``read_raw_config``) and
  persists via ``_persist_migration`` — steps therefore observe the writes of
  earlier steps through the filesystem, which is why strict ascending order
  is mandatory.
* All ``results['config_added']`` / ``results['warnings']`` appends and all
  conditional ``print`` output stay inside the step functions, byte-identical
  to the original blocks.

Import direction / cycle avoidance:

``hermes_cli.config`` imports :func:`run_migrations` lazily (inside
``migrate_config``), and every step function here resolves its helpers
(``read_raw_config``, ``_persist_migration``, ``get_env_value``, …) lazily
through the live ``hermes_cli.config`` module object at call time via
:func:`_cfg`. There is deliberately NO module-level import of
``hermes_cli.config`` here, so no circular import can form — and, just as
importantly, tests that monkeypatch helpers on ``hermes_cli.config`` (e.g.
``patch("hermes_cli.config.read_raw_config", ...)``) keep working, because
the steps always go through the module attribute rather than a bound-early
reference.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Tuple

#: Auto-migration support floor. Configs whose on-disk ``_config_version`` is
#: below this are NOT auto-migrated any more (policy decision, July 2026):
#: v12 predates roughly two years of releases, and carrying the sub-v12
#: migration steps (plus the env bridges they consumed, e.g.
#: HERMES_TOOL_PROGRESS*) forever is not worth it. Below-floor configs are
#: left byte-for-byte untouched — the process continues with the config as-is
#: (defaults deep-merged at read time, matching the non-fatal posture used
#: for unparseable configs) and a clear message tells the user how to
#: proceed. The removed steps were the <12 targets: v4 (tool-progress .env →
#: config.yaml), v5 (timezone seed), v9 (clear ANTHROPIC_TOKEN).
SUPPORT_FLOOR_VERSION = 12


def support_floor_message() -> str:
    """Human-facing explanation shown when a config is below the floor."""
    from hermes_constants import display_hermes_home

    return (
        f"This config predates version {SUPPORT_FLOOR_VERSION} (~2 years old) "
        "and can no longer be auto-migrated. Back up "
        f"{display_hermes_home()}/config.yaml and run `hermes doctor --fix` "
        f"or configure it manually. You may also set _config_version: {SUPPORT_FLOOR_VERSION} "
        "after reviewing the changelog."
    )


def _cfg():
    """Return the live ``hermes_cli.config`` module (lazy, cycle-free)."""
    from hermes_cli import config

    return config


def _migrate_to_12(results: Dict[str, Any], quiet: bool) -> None:
    # ── Version 11 → 12: migrate custom_providers list → providers dict ──
    _c = _cfg()
    read_raw_config = _c.read_raw_config
    _persist_migration = _c._persist_migration
    _custom_provider_entry_to_provider_config = _c._custom_provider_entry_to_provider_config

    config = read_raw_config()
    custom_list = config.get("custom_providers")
    if isinstance(custom_list, list) and custom_list:
        providers_dict = config.get("providers", {})
        if not isinstance(providers_dict, dict):
            providers_dict = {}
        migrated_count = 0
        for entry in custom_list:
            if not isinstance(entry, dict):
                continue
            old_name = entry.get("name", "")
            old_url = entry.get("base_url", "") or entry.get("url", "") or entry.get("api", "") or ""
            if not old_url:
                continue  # skip entries with no URL

            # Generate a kebab-case key from the display name
            key = old_name.strip().lower().replace(" ", "-").replace("(", "").replace(")", "")
            # Remove consecutive hyphens and trailing hyphens
            while "--" in key:
                key = key.replace("--", "-")
            key = key.strip("-")
            if not key:
                # Fallback: derive from URL hostname
                try:
                    from urllib.parse import urlparse
                    parsed = urlparse(old_url)
                    key = (parsed.hostname or "endpoint").replace(".", "-")
                except Exception:
                    key = f"endpoint-{migrated_count}"

            # Don't overwrite existing entries
            base_key = key
            suffix = migrated_count
            while key in providers_dict:
                key = f"{base_key}-{suffix}"
                suffix += 1

            new_entry = _custom_provider_entry_to_provider_config(
                entry,
                provider_key=key,
            )
            if new_entry is None:
                continue
            if not old_name:
                new_entry.pop("name", None)
            if new_entry.get("api_key") in {"no-key", "no-key-required", ""}:
                new_entry.pop("api_key", None)

            providers_dict[key] = new_entry
            migrated_count += 1

        if migrated_count > 0:
            config["providers"] = providers_dict
            # Remove the old list — runtime reads via get_compatible_custom_providers()
            config.pop("custom_providers", None)
            _persist_migration(config)
            if not quiet:
                print(f"  ✓ Migrated {migrated_count} custom provider(s) to providers: section")
                for key in list(providers_dict.keys())[-migrated_count:]:
                    ep = providers_dict[key]
                    print(f"    → {key}: {ep.get('api', '')}")


def _migrate_to_13(results: Dict[str, Any], quiet: bool) -> None:
    # ── Version 12 → 13: clear dead LLM_MODEL / OPENAI_MODEL from .env ──
    # These env vars were written by the old setup wizard but nothing reads
    # them anymore (config.yaml is the sole source of truth since March 2026).
    # Stale entries cause user confusion — see issue report.
    _c = _cfg()
    get_env_value = _c.get_env_value
    save_env_value = _c.save_env_value

    for dead_var in ("LLM_MODEL", "OPENAI_MODEL"):
        try:
            old_val = get_env_value(dead_var)
            if old_val:
                save_env_value(dead_var, "")
                if not quiet:
                    print(f"  ✓ Cleared {dead_var} from .env (no longer used — config.yaml is source of truth)")
        except Exception:
            pass


def _migrate_to_17(results: Dict[str, Any], quiet: bool) -> None:
    # ── Version 16 → 17: remove legacy compression.summary_* keys ──
    _c = _cfg()
    read_raw_config = _c.read_raw_config
    _persist_migration = _c._persist_migration

    config = read_raw_config()
    comp = config.get("compression", {})
    if isinstance(comp, dict):
        s_model = comp.pop("summary_model", None)
        s_provider = comp.pop("summary_provider", None)
        s_base_url = comp.pop("summary_base_url", None)
        migrated_keys = []
        # Migrate non-empty, non-default values to auxiliary.compression
        if s_model and str(s_model).strip():
            aux = config.setdefault("auxiliary", {})
            aux_comp = aux.setdefault("compression", {})
            if not aux_comp.get("model"):
                aux_comp["model"] = str(s_model).strip()
                migrated_keys.append(f"model={s_model}")
        if s_provider and str(s_provider).strip() not in {"", "auto"}:
            aux = config.setdefault("auxiliary", {})
            aux_comp = aux.setdefault("compression", {})
            if not aux_comp.get("provider") or aux_comp.get("provider") == "auto":
                aux_comp["provider"] = str(s_provider).strip()
                migrated_keys.append(f"provider={s_provider}")
        if s_base_url and str(s_base_url).strip():
            aux = config.setdefault("auxiliary", {})
            aux_comp = aux.setdefault("compression", {})
            if not aux_comp.get("base_url"):
                aux_comp["base_url"] = str(s_base_url).strip()
                migrated_keys.append(f"base_url={s_base_url}")
        if migrated_keys or s_model is not None or s_provider is not None or s_base_url is not None:
            config["compression"] = comp
            _persist_migration(config)
            if not quiet:
                if migrated_keys:
                    print(f"  ✓ Migrated compression.summary_* → auxiliary.compression: {', '.join(migrated_keys)}")
                else:
                    print("  ✓ Removed unused compression.summary_* keys")


def _migrate_to_21(results: Dict[str, Any], quiet: bool) -> None:
    # ── Version 20 → 21: plugins are now opt-in; grandfather existing user plugins ──
    # The loader now requires plugins to appear in ``plugins.enabled`` before
    # loading. Existing installs had all discovered plugins loading by default
    # (minus anything in ``plugins.disabled``). To avoid silently breaking
    # those setups on upgrade, populate ``plugins.enabled`` with the set of
    # currently-installed user plugins that aren't already disabled.
    #
    # Bundled plugins (shipped in the repo itself) are NOT grandfathered —
    # they ship off for everyone, including existing users, so any user who
    # wants one has to opt in explicitly.
    _c = _cfg()
    read_raw_config = _c.read_raw_config
    _persist_migration = _c._persist_migration
    get_hermes_home = _c.get_hermes_home
    fast_safe_load = _c.fast_safe_load

    config = read_raw_config()
    plugins_cfg = config.get("plugins")
    if not isinstance(plugins_cfg, dict):
        plugins_cfg = {}
    # Only migrate if the enabled allow-list hasn't been set yet.
    if "enabled" not in plugins_cfg:
        disabled = plugins_cfg.get("disabled", []) or []
        if not isinstance(disabled, list):
            disabled = []
        disabled_set = set(disabled)

        # Scan ``$HERMES_HOME/plugins/`` for currently installed user plugins.
        grandfathered: List[str] = []
        try:
            user_plugins_dir = get_hermes_home() / "plugins"
            if user_plugins_dir.is_dir():
                for child in sorted(user_plugins_dir.iterdir()):
                    if not child.is_dir():
                        continue
                    manifest_file = child / "plugin.yaml"
                    if not manifest_file.exists():
                        manifest_file = child / "plugin.yml"
                    if not manifest_file.exists():
                        continue
                    try:
                        with open(manifest_file, encoding="utf-8") as _mf:
                            manifest = fast_safe_load(_mf) or {}
                    except Exception:
                        manifest = {}
                    name = manifest.get("name") or child.name
                    if name in disabled_set:
                        continue
                    grandfathered.append(name)
        except Exception:
            grandfathered = []

        plugins_cfg["enabled"] = grandfathered
        config["plugins"] = plugins_cfg
        _persist_migration(config)
        results["config_added"].append(
            f"plugins.enabled (opt-in allow-list, {len(grandfathered)} grandfathered)"
        )
        if not quiet:
            if grandfathered:
                print(
                    f"  ✓ Plugins now opt-in: grandfathered "
                    f"{len(grandfathered)} existing plugin(s) into plugins.enabled"
                )
            else:
                print(
                    "  ✓ Plugins now opt-in: no existing plugins to grandfather. "
                    "Use `hermes plugins enable <name>` to activate."
                )


def _migrate_to_25(results: Dict[str, Any], quiet: bool) -> None:
    # ── Version 24 → 25: lower model_catalog TTL 24h → 1h ──
    # The model picker now refreshes its curated list hourly so freshly
    # published model-catalog.json deploys reach users without a day-long
    # stale window. Only rewrite the OLD default (24) — never clobber a
    # value the user deliberately customized.
    _c = _cfg()
    read_raw_config = _c.read_raw_config
    _persist_migration = _c._persist_migration

    config = read_raw_config()
    raw_mc = config.get("model_catalog")
    if isinstance(raw_mc, dict) and raw_mc.get("ttl_hours") == 24:
        raw_mc["ttl_hours"] = 1
        config["model_catalog"] = raw_mc
        _persist_migration(config)
        results["config_added"].append("model_catalog.ttl_hours 24→1")
        if not quiet:
            print("  ✓ Lowered model_catalog.ttl_hours to 1 (hourly picker refresh)")


def _migrate_to_29(results: Dict[str, Any], quiet: bool) -> None:
    # ── Version 28 → 29: retire the removed write-mode gates ──
    # The library build has no curator/CLI approval queue. Remove the historical
    # keys instead of migrating them into configuration that no runtime path
    # consumes.
    _c = _cfg()
    read_raw_config = _c.read_raw_config
    _persist_migration = _c._persist_migration

    config = read_raw_config()
    touched = False
    for subsystem in ("memory", "skills"):
        sub = config.get(subsystem)
        if not isinstance(sub, dict):
            continue
        if "write_mode" not in sub and "write_approval" not in sub:
            continue
        sub.pop("write_mode", None)
        sub.pop("write_approval", None)
        if sub:
            config[subsystem] = sub
        else:
            config.pop(subsystem, None)
        touched = True
        results["config_added"].append(f"removed obsolete {subsystem} write gate")
    if touched:
        _persist_migration(config)
        if not quiet:
            print("  ✓ Removed obsolete memory/skills write gates")


def _migrate_to_33(results: Dict[str, Any], quiet: bool) -> None:
    # ── Version 32 → 33: unify delegation concurrency caps ──
    # delegation.max_async_children is deprecated: max_concurrent_children now
    # caps both a single batch's parallelism and concurrent background
    # delegation units. Fold a raised max_async_children into
    # max_concurrent_children (take the max so nobody loses headroom), then
    # drop the stale key.
    _c = _cfg()
    read_raw_config = _c.read_raw_config
    _persist_migration = _c._persist_migration

    config = read_raw_config()
    raw_deleg = config.get("delegation")
    if isinstance(raw_deleg, dict) and "max_async_children" in raw_deleg:
        old_async = raw_deleg.pop("max_async_children")
        try:
            old_async_i = int(old_async)
        except (TypeError, ValueError):
            old_async_i = None
        if old_async_i is not None and old_async_i > 3:
            try:
                cur_children = int(raw_deleg.get("max_concurrent_children", 3))
            except (TypeError, ValueError):
                cur_children = 3
            if old_async_i > cur_children:
                raw_deleg["max_concurrent_children"] = old_async_i
                results["config_added"].append(
                    f"delegation.max_concurrent_children={old_async_i} "
                    f"(folded from deprecated max_async_children)"
                )
        config["delegation"] = raw_deleg
        _persist_migration(config)
        if not quiet:
            print(
                "  ✓ Removed deprecated delegation.max_async_children — "
                "delegation.max_concurrent_children now caps background "
                "delegations too."
            )


#: Registry of (target_version, migration_fn), strictly ascending. The driver
#: applies every entry whose target version is greater than the on-disk
#: version captured before the ladder started. Order matters: later steps may
#: observe earlier steps' writes via read_raw_config() (filesystem state).
MIGRATIONS: Tuple[Tuple[int, Callable[[Dict[str, Any], bool], None]], ...] = (
    # v12 is the support floor: configs already AT v12 (or newer) still get
    # every remaining step below. Only configs BELOW 12 are refused by the
    # floor gate in run_migrations().
    (12, _migrate_to_12),
    (13, _migrate_to_13),
    (17, _migrate_to_17),
    (21, _migrate_to_21),
    (25, _migrate_to_25),
    (29, _migrate_to_29),
    (33, _migrate_to_33),
)


def run_migrations(current_ver: int, results: Dict[str, Any], quiet: bool) -> None:
    """Apply every registered migration whose target version exceeds *current_ver*.

    Replicates the original ladder's semantics exactly: *current_ver* is the
    on-disk schema version captured ONCE (via ``check_config_version()``)
    before any step runs, and it does not advance between steps — each step
    is gated on the same initial value, exactly like the original sequential
    ``if current_ver < N:`` blocks. Steps run in strict ascending registry
    order and mutate ``results`` in place. The final ``_config_version`` bump
    is NOT performed here; it stays in ``migrate_config`` (persisted once,
    after the informational missing-config scan), matching the original flow.
    """
    for target_ver, migration_fn in MIGRATIONS:
        if current_ver < target_ver:
            migration_fn(results, quiet)
