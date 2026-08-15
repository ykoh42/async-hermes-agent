import ast
import json
import re
import tomllib
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
KITTENTTS_OFFICIAL_WHEEL_URL = (
    "https://github.com/KittenML/KittenTTS/releases/download/0.8.1/"
    "kittentts-0.8.1-py3-none-any.whl"
)
KITTENTTS_OFFICIAL_WHEEL_SHA256 = (
    "482a436c4f1f3192153710376e459ff3689517ebcda7c2b051e2fd4187b41851"
)
_PRE_PYTHON_311_STDLIB_BACKPORTS = frozenset({"backports.zoneinfo", "tomli"})


def _find_pre_python_311_stdlib_backport_imports(
    source: str,
) -> list[tuple[int, str]]:
    """Return absolute imports of obsolete stdlib backports and descendants."""
    violations = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported = {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported = {f"{node.module}.{alias.name}" for alias in node.names}
        else:
            continue
        for module in imported:
            if any(
                module == backport or module.startswith(f"{backport}.")
                for backport in _PRE_PYTHON_311_STDLIB_BACKPORTS
            ):
                violations.append((node.lineno, module))
    return violations


def test_release_versions_match_upstream_revision_policy():
    """Python and private npm metadata must encode one upstream revision."""
    pyproject = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    project_version = pyproject["project"]["version"]
    version_policy = pyproject["tool"]["async-hermes"]
    package = json.loads(
        (REPO_ROOT / "package.json").read_text(encoding="utf-8")
    )
    package_lock = json.loads(
        (REPO_ROOT / "package-lock.json").read_text(encoding="utf-8")
    )

    cli_module = ast.parse(
        (REPO_ROOT / "hermes_cli" / "__init__.py").read_text(encoding="utf-8")
    )
    cli_metadata = {
        target.id: node.value.value
        for node in cli_module.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, (str, int))
    }

    upstream_version = version_policy["upstream_version"]
    upstream_tag = version_policy["upstream_tag"]
    async_revision = version_policy["async_revision"]
    expected_python_version = f"{upstream_version}.{async_revision}"
    expected_npm_version = f"{upstream_version}-async.{async_revision}"

    assert re.fullmatch(r"\d+\.\d+\.\d+", upstream_version)
    assert re.fullmatch(r"v\d{4}\.\d{1,2}\.\d{1,2}", upstream_tag)
    assert upstream_tag == f'v{cli_metadata["__release_date__"]}'
    assert isinstance(async_revision, int) and async_revision > 0
    assert project_version == expected_python_version
    assert cli_metadata["__version__"] == expected_python_version
    assert package["version"] == expected_npm_version
    assert package_lock["version"] == expected_npm_version
    assert package_lock["packages"][""]["version"] == expected_npm_version


def test_postgres_backend_is_optional_and_packaged_as_a_top_level_module():
    pyproject = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    extra = pyproject["project"]["optional-dependencies"]["postgres"]
    assert extra == ["SQLAlchemy[asyncio]==2.0.51", "asyncpg==0.31.0"]
    assert "hermes_state_postgres" in pyproject["tool"]["setuptools"]["py-modules"]


def test_release_workflow_installs_ripgrep_before_testing_source():
    """Release runners must provide the binary required by search tests."""
    workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )

    install_step = workflow.index("- name: Install ripgrep")
    test_step = workflow.index("- name: Test released source")

    assert install_step < test_step
    assert "sudo apt-get install --yes ripgrep" in workflow[install_step:test_step]


def test_release_workflow_smokes_every_supported_python_version():
    """Release artifacts must install under every declared Python runtime."""
    workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )

    assert "uv python install 3.11 3.12 3.13" in workflow
    assert "for python_version in 3.12 3.13; do" in workflow
    assert 'uv pip check --python "$compat_venv/bin/python"' in workflow
    assert '.release-venv/bin/python -I - <<\'PY\'' in workflow
    assert '"$compat_venv/bin/python" -I - <<\'PY\'' in workflow


def test_python_compatibility_ci_scopes_tests_before_pytest_separator():
    """Compatibility paths must select files, not repeat in every pytest run."""
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    compat_step = workflow.split(
        "- name: Test retained compatibility surface", 1
    )[1].split("\n      - name:", 1)[0]
    separator = compat_step.index("\n          -- ")

    for path in (
        "tests/runtime/test_core.py",
        "tests/test_hermes_state.py",
        "tests/tools/test_terminal_exit_semantics.py",
    ):
        assert compat_step.count(path) == 1
        assert compat_step.index(path) < separator


def test_python_compatibility_ci_installs_ripgrep_before_testing_source():
    """Compatibility runners must provide the binary required by search tests."""
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    compat_job = workflow.split("\n  python-compatibility:", 1)[1]
    install_step = compat_job.index("- name: Install ripgrep")
    test_step = compat_job.index("- name: Test retained compatibility surface")

    assert install_step < test_step
    assert "sudo apt-get install --yes ripgrep" in compat_job[install_step:test_step]


def test_ci_postgres_matrix_covers_supported_python_and_server_pairs():
    """The optional backend is exercised against real PostgreSQL services."""
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    postgres_job = workflow.split("\n  postgres:", 1)[1]
    for pair in (
        "python-version: '3.11'\n            postgres-version: '15'",
        "python-version: '3.12'\n            postgres-version: '16'",
        "python-version: '3.12'\n            postgres-version: '17'",
        "python-version: '3.13'\n            postgres-version: '18'",
    ):
        assert pair in postgres_job
    assert "--extra postgres" in postgres_job
    assert (
        "uv run --locked --python ${{ matrix.python-version }}"
        in postgres_job
    )
    assert "--extra dev --extra postgres pytest" in postgres_job
    assert "HERMES_POSTGRES_TEST_DSN" in postgres_job
    assert "tests/integration/test_postgres_session_db.py" in postgres_job
    assert "tests/integration/test_postgres_compaction_e2e.py" in postgres_job


def test_pages_workflow_validates_pull_requests_without_deploying_them():
    """Trusted PRs build in isolation, while only merged main may deploy."""
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "deploy-site.yml"
    ).read_text(encoding="utf-8")

    assert "pull_request:\n    paths:\n      - 'website/**'" in workflow
    assert "github.event.pull_request.head.repo.full_name == github.repository" in workflow
    assert "group: pages-${{ github.event_name == 'pull_request'" in workflow
    assert "cancel-in-progress: ${{ github.event_name == 'pull_request' }}" in workflow
    assert "github.event_name != 'pull_request' &&" in workflow
    assert "github.ref == 'refs/heads/main'" in workflow
    assert "uses: actions/deploy-pages@" in workflow


def test_release_workflow_publishes_verified_artifacts_with_least_privilege():
    """One verified tag build must feed PyPI and the GitHub release."""
    workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )

    lint_step = workflow.index("- name: Lint released source")
    audit_step = workflow.index("- name: Audit the native async runtime")
    test_step = workflow.index("- name: Test released source")
    build_step = workflow.index("- name: Build distributions")
    stage_step = workflow.index("- name: Stage a private GitHub draft release")
    publish_step = workflow.index("- name: Publish verified distributions to PyPI")
    verify_step = workflow.index("- name: Verify published PyPI digests")
    release_step = workflow.index("- name: Publish the staged GitHub release")
    preflight_step = workflow.index("- name: Refuse an existing GitHub release")
    preflight_job = workflow[
        preflight_step : workflow.index("- name: Verify lockfile")
    ]
    stage_job = workflow[
        workflow.index("  stage-release:") : workflow.index("  publish-pypi:")
    ]
    publish_job = workflow[
        workflow.index("  publish-pypi:") : workflow.index("  verify-pypi:")
    ]
    verify_job = workflow[
        workflow.index("  verify-pypi:") : workflow.index("  publish-release:")
    ]
    release_tag_pattern = r"v[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+"

    assert "push:\n    tags:\n      - 'v*.*.*.*'" in workflow
    assert workflow.count(
        f"re.fullmatch(r'{release_tag_pattern}', tag)"
    ) == 2
    assert re.fullmatch(release_tag_pattern, "v0.20.1.1")
    assert not any(
        re.fullmatch(release_tag_pattern, tag)
        for tag in ("v0.20.1", "v0.20.1.1.1", "v0.20.1.1rc1", "0.20.1.1")
    )
    assert "release:\n    types: [published]" not in workflow
    assert (
        'test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)"'
        in workflow
    )
    assert "git merge-base --is-ancestor HEAD origin/main" not in workflow
    assert (
        preflight_step
        < lint_step
        < audit_step
        < test_step
        < build_step
        < stage_step
        < publish_step
        < verify_step
        < release_step
    )
    assert 'gh release create "$RELEASE_TAG" \\' in workflow
    assert 'gh release view "$RELEASE_TAG" --repo "$GH_REPO"' in workflow
    assert workflow.count('gh release view "$RELEASE_TAG" --repo "$GH_REPO"') == 2
    assert 'if gh release view "$RELEASE_TAG" --repo "$GH_REPO"' in preflight_job
    assert "--json isDraft" not in preflight_job
    assert "Refusing to rebuild an existing draft or published release" in workflow
    assert "Rerun only the failed jobs" in workflow
    assert "stage-release:\n    needs: build" in workflow
    assert "--json isDraft" in stage_job
    assert 'test "$is_draft" = true' in stage_job
    assert "--draft \\" in stage_job
    assert 'if [ "$RELEASE_TAG" = "v0.20.1.1" ]; then' in stage_job
    assert "v2026.8.13" in stage_job
    assert '--notes "$release_notes" \\' in stage_job
    assert 'gh release upload "$RELEASE_TAG" dist/* \\' in stage_job
    assert "--clobber \\" in stage_job
    assert '--verify-tag \\\n' in workflow
    assert '--repo "$GH_REPO"' in workflow
    assert "GH_REPO: ${{ github.repository }}" in workflow
    assert "publish-pypi:\n    needs: stage-release" in workflow
    assert "verify-pypi:\n    needs: publish-pypi" in workflow
    assert "publish-release:\n    needs: verify-pypi" in workflow
    assert (
        'gh release edit "$RELEASE_TAG" --draft=false --repo "$GH_REPO"'
        in workflow
    )
    assert "name: pypi" in workflow
    assert "url: https://pypi.org/p/async-hermes-agent" in workflow
    assert (
        "pypa/gh-action-pypi-publish@"
        "cef221092ed1bacb1cc03d23a2d87d1d172e277b"
    ) in workflow
    assert "packages-dir: dist/" in workflow
    assert "skip-existing: true" in publish_job
    assert "permissions: {}" in verify_job
    assert "hashlib.sha256(path.read_bytes()).hexdigest()" in verify_job
    assert 'if published == expected:' in verify_job
    assert 'wheels = [name for name in expected if name.endswith(".whl")]' in verify_job
    assert (
        'sdists = [name for name in expected if name.endswith(".tar.gz")]'
        in verify_job
    )
    assert "if len(wheels) != 1 or len(sdists) != 1:" in verify_job
    assert "pypi.org/pypi/async-hermes-agent/" in verify_job
    assert "time.sleep(5)" in verify_job
    assert "PYPI_TOKEN" not in workflow
    assert "TWINE_" not in workflow
    assert "password:" not in workflow
    assert "${{ secrets." not in publish_job
    assert re.search(
        r"uses: pypa/gh-action-pypi-publish@[0-9a-f]{40}(?:\s|$)",
        publish_job,
    )
    assert workflow.count("id-token: write") == 1
    assert "permissions:\n      id-token: write" in workflow
    assert "permissions:\n  id-token: write" not in workflow
    assert "sha256sum *.whl *.tar.gz > SHA256SUMS" in workflow
    assert (
        "path: |\n            dist/*.whl\n            dist/*.tar.gz"
        in workflow
    )
    assert workflow.count("contents: write") == 2
    assert "permissions:\n  contents: read" in workflow


def test_release_documentation_uses_current_package_version():
    """Published install snippets must match the Python package version."""
    project_version = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]["version"]
    pinned_install = f'async-hermes-agent=={project_version}'
    release_pages = (
        REPO_ROOT / "website" / "docs" / "index.mdx",
        REPO_ROOT / "website" / "docs" / "getting-started" / "installation.md",
        REPO_ROOT / "website" / "docs" / "getting-started" / "platform-support.md",
    )

    for page in release_pages:
        assert pinned_install in page.read_text(encoding="utf-8"), page

    readmes = (REPO_ROOT / "README.md", REPO_ROOT / "README.es.md")
    for readme_path in readmes:
        readme = readme_path.read_text(encoding="utf-8")
        assert pinned_install in readme, readme_path
        assert "not published to PyPI" not in readme, readme_path


# Minimum non-vulnerable Starlette: CVE-2026-48710 ("BadHost") was fixed in
# 1.0.1. Anything below that lets a malformed Host header desync
# ``request.url.path`` from the dispatched ASGI path, bypassing path-based
# authz in middleware/endpoints that gate on ``request.url``. Starlette is a
# transitive dep of sse-starlette/mcp in the core, so we pin it directly and
# enforce the floor in both pyproject and the committed lockfile.
_STARLETTE_CVE_FLOOR = (1, 0, 1)
_UPDATE_DOWNGRADE_GUARD_FLOORS = {
    # `hermes update` reinstalls exact pins from pyproject/lazy_deps. These
    # reviewed CVE pins must not slide back to stale versions that downgrade
    # already-patched user environments.
    "cryptography": (48, 0, 1),
    "starlette": (1, 3, 1),
}


def _version_tuple(spec: str) -> tuple[int, ...]:
    # "1.0.1" -> (1, 0, 1); tolerant of pre/post suffixes by truncating.
    head = spec.split("+", 1)[0]
    parts = []
    for chunk in head.split("."):
        digits = "".join(ch for ch in chunk if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def test_starlette_pinned_above_cve_2026_48710_floor_in_pyproject():
    """Every extra that declares Starlette must pin a patched (>=1.0.1) version.

    Regression guard for #35067 / CVE-2026-48710. A future edit that drops the
    pin (re-exposing the unbounded transitive ``starlette>=0.27`` from MCP)
    or pins a pre-1.0.1 version fails here instead of shipping a Host-header
    auth-bypass to MCP-HTTP users.
    """
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]

    found = {}
    declarations = {"core": data["project"]["dependencies"], **extras}
    for extra, specs in declarations.items():
        for spec in specs:
            name = spec.split("==", 1)[0].split(">", 1)[0].split("<", 1)[0].split("[", 1)[0].strip()
            if name.lower() == "starlette":
                assert "==" in spec, f"[{extra}] must exact-pin starlette, got {spec!r}"
                ver = spec.split("==", 1)[1].split(";", 1)[0].strip()
                found[extra] = ver

    assert "core" in found, (
        "[core] no longer pins starlette directly — CVE-2026-48710 "
        "regression risk (MCP pulls it transitively with no upper bound)"
    )

    for extra, ver in found.items():
        assert _version_tuple(ver) >= _STARLETTE_CVE_FLOOR, (
            f"[{extra}] pins starlette=={ver}, below the CVE-2026-48710 fix "
            f"floor {'.'.join(map(str, _STARLETTE_CVE_FLOOR))}"
        )


def test_locked_starlette_is_not_vulnerable_to_cve_2026_48710():
    """The committed uv.lock must resolve starlette to a patched version.

    pyproject pins protect the declared extras, but the lockfile is what
    hash-verified installs (``uv sync --locked``) actually pull. Assert the
    resolved version is >= the CVE-2026-48710 fix floor so a stale-lock
    regression can't ship a vulnerable Starlette to users.
    """
    lock = (REPO_ROOT / "uv.lock").read_text(encoding="utf-8")
    versions = []
    in_starlette = False
    for line in lock.splitlines():
        if line.startswith("[[package]]"):
            in_starlette = False
        elif line.strip() == 'name = "starlette"':
            in_starlette = True
        elif in_starlette and line.startswith("version = "):
            versions.append(line.split("=", 1)[1].strip().strip('"'))
            in_starlette = False

    assert versions, "starlette not found in uv.lock"
    for ver in versions:
        assert _version_tuple(ver) >= _STARLETTE_CVE_FLOOR, (
            f"uv.lock resolves starlette=={ver}, below the CVE-2026-48710 fix "
            f"floor {'.'.join(map(str, _STARLETTE_CVE_FLOOR))} — regenerate the "
            f"lockfile after bumping the pin"
        )


def test_provider_extras_only_publish_native_async_dependencies():
    """Only providers with native async transports may publish install extras."""
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]
    declared = {
        _canonical(re.split(r"[<>=!~;\[]", spec, maxsplit=1)[0].strip())
        for specs in (data["project"]["dependencies"], *extras.values())
        for spec in specs
    }

    assert extras["bedrock"] == [
        "aiobotocore==3.8.0",
        "anthropic[bedrock]==0.87.0",
    ]
    assert extras["vertex"] == ["google-auth[aiohttp]==2.56.2"]
    assert extras["azure-identity"] == [
        "azure-identity==1.25.3",
        "aiohttp==3.14.3",
    ]
    assert extras["hindsight"] == [
        "hindsight-client==0.6.1",
        "packaging==26.0",
    ]
    assert extras["honcho"] == ["honcho-ai==2.2.0"]
    assert "google-auth" in declared
    assert "azure-identity" in declared
    assert "aiobotocore" in declared
    assert "hindsight-client" in declared
    assert "honcho-ai" in declared
    assert "boto3" not in declared


def test_retained_runtime_dependencies_are_declared_and_exact_pinned():
    """Direct retained imports must not drift through transitive resolution."""
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    core = data["project"]["dependencies"]
    extras = data["project"]["optional-dependencies"]

    assert "certifi==2026.5.20" in core
    assert "cryptography==48.0.1" in core
    assert "protobuf==6.33.6" in core
    assert _locked_versions("protobuf") == {"6.33.6"}
    assert "packaging==26.0" in extras["hindsight"]


def test_published_runtime_has_no_pre_python_311_stdlib_backports():
    """Published code must use stdlib modules guaranteed by requires-python."""
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    setuptools = data["tool"]["setuptools"]
    runtime_files = {
        REPO_ROOT / f"{module}.py"
        for module in setuptools["py-modules"]
    }
    package_names = {
        pattern.split(".", 1)[0]
        for pattern in setuptools["packages"]["find"]["include"]
    }
    for package_name in package_names:
        runtime_files.update((REPO_ROOT / package_name).rglob("*.py"))

    violations = []
    for source_path in sorted(runtime_files):
        source = source_path.read_text(encoding="utf-8")
        for lineno, module in _find_pre_python_311_stdlib_backport_imports(source):
            violations.append(
                f"{source_path.relative_to(REPO_ROOT)}:{lineno}: {module}"
            )

    assert violations == []


@pytest.mark.parametrize(
    "source",
    [
        "import tomli",
        "import tomli.decoder",
        "from tomli import decoder",
        "from backports import zoneinfo",
        "from backports.zoneinfo import ZoneInfo",
    ],
)
def test_pre_python_311_stdlib_backport_detector_rejects_equivalent_imports(source):
    assert _find_pre_python_311_stdlib_backport_imports(source)


@pytest.mark.parametrize(
    "source",
    [
        "import tomllib",
        "from tomllib import loads",
        "import backports",
        "from backports import unrelated",
    ],
)
def test_pre_python_311_stdlib_backport_detector_allows_unrelated_imports(source):
    assert _find_pre_python_311_stdlib_backport_imports(source) == []


def test_full_suite_workflows_install_the_backends_they_exercise():
    """CI and release must resolve the same optional backend test surface."""
    expected = {
        "dev",
        "bedrock",
        "supermemory",
        "hindsight",
        "honcho",
        "mem0",
    }
    for workflow_name in ("ci.yml", "release.yml"):
        workflow = (REPO_ROOT / ".github" / "workflows" / workflow_name).read_text(
            encoding="utf-8"
        )
        install_step = re.search(
            r"- name: Install project and test dependencies\n"
            r"\s+run: (?P<command>[^\n]+)",
            workflow,
        )
        assert install_step is not None, workflow_name
        installed = set(
            re.findall(r"--extra\s+([a-z0-9-]+)", install_step["command"])
        )
        assert installed == expected


def test_retained_no_op_compatibility_extras_are_published():
    """Upstream extra names survive when native transports moved deps to core."""
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]

    for extra in (
        "computer-use",
        "exa",
        "firecrawl",
        "homeassistant",
        "mcp",
        "vision",
    ):
        assert extras[extra] == []




# ---------------------------------------------------------------------------
# Dependency-pin consistency: pyproject extras <-> tools/lazy_deps.py
#
# The same package is exact-pinned in two hand-maintained places: the
# [project.optional-dependencies] extras in pyproject.toml and the LAZY_DEPS
# allowlist in tools/lazy_deps.py (the lazy-install path deliberately mirrors
# the extras — see the comments on LAZY_DEPS: "match the corresponding extra
# in pyproject.toml ... update both this map AND the corresponding extra").
#
# They have silently drifted more than once: the aiohttp Slack pin (3.13.3 in
# the extras vs 3.13.4 in lazy_deps) and the anthropic pin (0.86.0 vs 0.87.0).
# The version a user ends up with then depends on whether the backend was
# installed eagerly (extra) or lazily (lazy_deps) — and for a CVE bump applied
# to only one side, that divergence is a latent security regression. These two
# tests assert the documented contract: the two sources agree, in lockstep.
# ---------------------------------------------------------------------------

# Matches "name==version" and "name[extra]==version", ignoring any trailing
# environment marker / comment. Only exact pins are collected; ranged specs
# (">=", "<") can't be compared for equality and are skipped.
_PIN_RE = re.compile(
    r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*==\s*([^\s;,#]+)"
)


def _canonical(name: str) -> str:
    # PEP 503 normalization so e.g. discord.py / discord-py compare equal.
    return re.sub(r"[-_.]+", "-", name).lower()


def _pins_from_specs(specs):
    """Map canonical package name -> set of exact-pinned versions seen."""
    pins: dict[str, set[str]] = {}
    for spec in specs:
        m = _PIN_RE.match(spec)
        if not m:
            continue
        pins.setdefault(_canonical(m.group(1)), set()).add(m.group(2))
    return pins


def _locked_versions(package: str) -> set[str]:
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
    return {
        pkg["version"]
        for pkg in lock.get("package", [])
        if _canonical(pkg["name"]) == _canonical(package)
    }


def test_local_tts_install_metadata_is_publishable_and_exact():
    """Local TTS guidance must use the reviewed, compatible artifacts."""
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]
    extra_requirements = [spec for specs in extras.values() for spec in specs]

    assert extras["piper-tts"] == ["piper-tts==1.6.0"]
    assert _locked_versions("piper-tts") == {"1.6.0"}
    assert all("://" not in spec for spec in extra_requirements)
    assert all(
        _canonical(re.split(r"[<>=!~;\[ @]", spec, maxsplit=1)[0].strip())
        != "kittentts"
        for spec in extra_requirements
    )

    docs = (REPO_ROOT / "website/docs/getting-started/installation.md").read_text(
        encoding="utf-8"
    )
    install_url = (
        f"{KITTENTTS_OFFICIAL_WHEEL_URL}"
        f"#sha256={KITTENTTS_OFFICIAL_WHEEL_SHA256}"
    )
    assert f"python -m pip install \\\n  '{install_url}'" in docs
    assert "package named `kittentts` is not the compatible KittenML 0.8.1" in docs
    assert "pip install kittentts" not in docs
    assert f"'{install_url}' \\\n  soundfile" not in docs


def _pyproject_pinned_specs():
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    specs = list(data["project"].get("dependencies", []))
    for extra in data["project"].get("optional-dependencies", {}).values():
        specs.extend(extra)
    return specs


def _lazy_deps_pinned_specs():
    """Extract every string literal inside the LAZY_DEPS dict via AST.

    Parsing rather than importing keeps this test free of
    tools/lazy_deps.py's runtime imports and side effects.
    """
    src = (REPO_ROOT / "tools" / "lazy_deps.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    specs: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if not any(isinstance(t, ast.Name) and t.id == "LAZY_DEPS" for t in targets):
            continue
        for sub in ast.walk(node.value):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                specs.append(sub.value)
    assert specs, "could not extract specs from LAZY_DEPS — the AST parser drifted"
    return specs


def test_pyproject_pins_are_internally_consistent():
    """No package may be exact-pinned to two different versions in pyproject.

    A package legitimately appearing in several extras (e.g. aiohttp in
    messaging/slack/homeassistant/sms) must use the SAME version everywhere.
    """
    pins = _pins_from_specs(_pyproject_pinned_specs())
    conflicts = {name: sorted(v) for name, v in pins.items() if len(v) > 1}
    assert not conflicts, (
        "pyproject.toml exact-pins the same package to different versions "
        "across [project.dependencies] / extras: " + str(conflicts)
    )




def _lazy_deps_by_feature():
    """Parse LAZY_DEPS into {feature_name: [spec, ...]} via AST.

    Same parse-don't-import rationale as _lazy_deps_pinned_specs, but keeps the
    feature -> specs grouping so per-feature coverage can be asserted.
    """
    src = (REPO_ROOT / "tools" / "lazy_deps.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        targets = (
            node.targets if isinstance(node, ast.Assign)
            else [node.target] if isinstance(node, ast.AnnAssign)
            else []
        )
        if not any(isinstance(t, ast.Name) and t.id == "LAZY_DEPS" for t in targets):
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        by_feature: dict[str, list[str]] = {}
        for key, value in zip(node.value.keys, node.value.values):
            if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                continue
            by_feature[key.value] = [
                sub.value
                for sub in ast.walk(value)
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str)
            ]
