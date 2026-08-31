import json
from pathlib import Path

NOTEBOOK = Path("notebooks/durable_agents_examples.ipynb")


def test_notebook_contains_all_agent_examples() -> None:
    notebook = json.loads(NOTEBOOK.read_text())
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    assert "Regular durable agent" in source
    assert "SandboxAgent" in source
    assert "Agent as a tool" in source
    assert "DBOSRunner.run" in source
