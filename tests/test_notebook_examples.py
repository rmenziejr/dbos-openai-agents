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
    assert "DBOSCapability(Shell())" in source
    assert "await DBOSRunner.run(" in source
    assert "DBOS.fork_workflow_async" in source
    assert "dbos.workflow_status" in source
    assert "dbos.operation_outputs" in source
    assert "dbos.streams" in source
    assert "_native_action_step" in source
    assert "dbos-capability-events" in source
    assert "sandbox_temporary_dir = tempfile.TemporaryDirectory(" in source
    assert "sandbox_temporary_dir.cleanup()" in source
    assert (
        source.index("sandbox_temporary_dir = tempfile.TemporaryDirectory(")
        < source.index("DBOS.fork_workflow_async")
        < source.index("sandbox_temporary_dir.cleanup()")
    )
