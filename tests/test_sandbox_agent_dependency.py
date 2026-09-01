import subprocess
import sys

from agents.sandbox import SandboxAgent


def test_agents_sdk_exposes_sandbox_agent() -> None:
    assert SandboxAgent.__name__ == "SandboxAgent"


def test_durability_import_does_not_require_private_dbos_bridge() -> None:
    script = """
import builtins
import importlib
import dbos_openai_agents.durability as durability

real_import = builtins.__import__

def blocked_import(name, *args, **kwargs):
    if name in {"dbos._context", "dbos._core", "dbos._dbos"}:
        raise ImportError(f"blocked private import: {name}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = blocked_import
importlib.reload(durability)
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
