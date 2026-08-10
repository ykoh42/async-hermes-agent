"""First-use import coverage for the process-level async bootstrap."""

import subprocess
import sys


def test_process_bootstrap_preloads_httpcore_in_a_fresh_interpreter():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import agent.process_bootstrap; "
                "assert 'httpcore' in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
