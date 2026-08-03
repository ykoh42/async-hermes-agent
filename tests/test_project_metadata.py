"""Regression tests for packaging metadata in pyproject.toml."""

from pathlib import Path
import tomllib

def _load_optional_dependencies():
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject_path.open("rb") as handle:
        project = tomllib.load(handle)["project"]
    return project["optional-dependencies"]


def _exact_pins(specs):
    pins = {}
    for spec in specs:
        requirement = spec.split(";", 1)[0].strip()
        if "==" not in requirement:
            continue
        package, version = requirement.split("==", 1)
        package = package.split("[", 1)[0].lower().replace("_", "-")
        pins[package] = version
    return pins





def test_pyproject_pins_match_lazy_deps_pins():
    """Generalize #31817 to the whole pin surface, not just aiohttp.

    Any package that is exact-pinned in BOTH a pyproject extra and a
    `tools/lazy_deps.py` LAZY_DEPS entry must use the SAME version in both
    places. When they drift, `hermes update` resolves the pyproject extra
    pin and downgrades the package to the older version, reopening whatever
    the lazy pin fixed (the aiohttp #31817 case, and the anthropic
    CVE-2026-34450/34452 case found alongside it) — only for the lazy
    refresh to re-upgrade it on next feature use. The lazy pin is the
    security-current source of truth; extras must track it.
    """
    from tools.lazy_deps import LAZY_DEPS

    optional_dependencies = _load_optional_dependencies()

    # package -> version, as pinned across all pyproject extras. If an
    # extra pins a package at a different version than another extra, that
    # is itself a bug (caught below); here we just collect the set.
    pyproject_pins: dict[str, set[str]] = {}
    for specs in optional_dependencies.values():
        for package, version in _exact_pins(specs).items():
            pyproject_pins.setdefault(package, set()).add(version)

    # package -> version, as pinned across all LAZY_DEPS entries.
    lazy_pins: dict[str, set[str]] = {}
    for specs in LAZY_DEPS.values():
        if isinstance(specs, str):
            specs = (specs,)
        for package, version in _exact_pins(specs).items():
            lazy_pins.setdefault(package, set()).add(version)

    shared = sorted(set(pyproject_pins) & set(lazy_pins))
    assert shared, "expected at least one package pinned in both pyproject and LAZY_DEPS"

    drift = {
        package: {
            "pyproject": sorted(pyproject_pins[package]),
            "lazy_deps": sorted(lazy_pins[package]),
        }
        for package in shared
        if pyproject_pins[package] != lazy_pins[package]
    }
    assert not drift, (
        "pyproject extras pins must match tools/lazy_deps.py LAZY_DEPS pins "
        "for every shared package — otherwise `hermes update` downgrades the "
        "package below the security-current lazy pin (see #31817). Drift: "
        f"{drift}"
    )






def _uv_lock_version(package: str) -> str:
    """Resolved version of ``package`` in uv.lock, or fail loudly."""
    versions = _uv_lock_versions(package)
    assert versions, f"{package} not found in uv.lock"
    assert len(versions) == 1, f"{package} resolves to multiple versions in uv.lock: {versions}"
    return next(iter(versions))


def _uv_lock_versions(package: str) -> set[str]:
    """All resolved versions of ``package`` in uv.lock (normally 0 or 1)."""
    import re

    lock_path = Path(__file__).resolve().parents[1] / "uv.lock"
    lock = lock_path.read_text(encoding="utf-8")
    return {
        m.group(1)
        for m in re.finditer(
            rf'\[\[package\]\]\nname = "{re.escape(package)}"\nversion = "([^"]+)"',
            lock,
        )
    }


def test_every_lazy_deps_exact_pin_matches_uv_lock():
    """Class invariant for #60783/#60685: one version per package, everywhere.

    Any package that is BOTH exact-pinned in ``tools/lazy_deps.py`` AND
    resolved in the committed uv.lock is a *shared* package: the core
    install ships the locked version, and the ``hermes update`` lazy-refresh
    pass re-asserts the LAZY_DEPS pin whenever the package is present
    (``active_features()``). If the two disagree, every update churns the
    package — and when the lazy pin is older, it force-DOWNGRADES a version
    another consumer needs (huggingface-hub==1.2.3 vs transformers'
    >=1.5.0 broke Hindsight local embeddings; stale aiohttp pins reopened
    patched CVEs in #31817). Contract: for every such package, pin ==
    locked version. When bumping a pin, regenerate the lock in the same
    commit (`uv lock --upgrade-package <name>`), and vice versa.
    """
    from tools.lazy_deps import LAZY_DEPS

    drift = {}
    seen = set()
    for feature, specs in LAZY_DEPS.items():
        for package, pin in _exact_pins(specs).items():
            if (package, pin) in seen:
                continue
            seen.add((package, pin))
            locked = _uv_lock_versions(package)
            if not locked:
                # Lazy-only package never resolved by the core lock — no
                # shared-version hazard.
                continue
            if pin not in locked:
                drift.setdefault(package, {})[feature] = {
                    "lazy_pin": pin,
                    "uv_lock": sorted(locked),
                }

    assert not drift, (
        "LAZY_DEPS exact pins must match the uv.lock resolved version for "
        "every package the core lock also ships — otherwise `hermes update` "
        "churns/downgrades the shared package out from under its other "
        "consumers (#60783, #31817). Bump the pin AND run "
        "`uv lock --upgrade-package <name>` in the same commit. Drift: "
        f"{drift}"
    )
