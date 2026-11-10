"""Keep the offline example scripts runnable.

Each example under ``examples/`` that needs no network, model download
or third-party SDK is executed in a subprocess with the repository's
interpreter. The scripts double as living documentation, so a refactor
that silently breaks one would otherwise go unnoticed until a reader
copies it.

``adapt_pyrit_to_guards.py`` is excluded because it loads a
sentence-transformers model when one is installed; it is covered by
the slow cluster test instead.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"

OFFLINE_EXAMPLES = [
    "loop_detection.py",
    "stream_cascade.py",
    "flow_openai_agents.py",
    "flow_plan_dsl.py",
    "stream_yaml_config.py",
    "provider_streaming.py",
    "asgi_middleware.py",
]


@pytest.mark.parametrize("script", OFFLINE_EXAMPLES)
def test_example_runs_cleanly(script: str) -> None:
    path = EXAMPLES / script
    assert path.exists(), f"missing example {script}"
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, str(path)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert proc.returncode == 0, f"{script} failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert "Traceback" not in proc.stderr


def test_every_example_is_either_listed_or_documented_as_online() -> None:
    """New examples must opt in here or be named in the online set."""
    online = {"adapt_pyrit_to_guards.py"}
    present = {p.name for p in EXAMPLES.glob("*.py")}
    unaccounted = present - set(OFFLINE_EXAMPLES) - online
    assert not unaccounted, f"add these examples to OFFLINE_EXAMPLES or online: {sorted(unaccounted)}"
