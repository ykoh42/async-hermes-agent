"""The async library must produce installable wheel and source artifacts."""

import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("kind", "artifact_glob"),
    [("sdist", "async_hermes_agent-*.tar.gz"), ("wheel", "async_hermes_agent-*.whl")],
)
def test_artifact_build_succeeds(kind: str, artifact_glob: str, tmp_path: Path):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from setuptools.build_meta import build_{kind}; build_{kind}(r'{out}')".format(
                kind=kind,
                out=tmp_path,
            ),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert list(tmp_path.glob(artifact_glob))
