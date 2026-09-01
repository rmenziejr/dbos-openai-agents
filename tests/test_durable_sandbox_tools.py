import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from agents import RunConfig, Usage
from agents.run_config import SandboxRunConfig
from agents.items import ModelResponse
from agents.sandbox import Manifest, SandboxAgent
from agents.sandbox.capabilities import (
    Filesystem,
    FilesystemToolSet,
    Shell,
    ShellToolSet,
)
from agents.sandbox.entries import LocalDir
from agents.sandbox.sandboxes.unix_local import UnixLocalSandboxClient
from dbos import DBOS
from openai.types.responses import ResponseCustomToolCall

from dbos_openai_agents import DBOSCapability, DBOSRunner
from dbos_openai_agents.durability import run_durable_action
from utils import FakeModel, make_message_response, make_tool_call_response


class _Turnstile:
    async def wait_for(self, call_id: str) -> None:
        return None

    def allow_next_after(self, call_id: str) -> None:
        return None


def state_with_call(call_id: str) -> SimpleNamespace:
    del call_id
    return SimpleNamespace(turnstile=_Turnstile())


async def _stream_events(workflow_id: str, key: str) -> list[object]:
    return [event async for event in DBOS.read_stream_async(workflow_id, key)]


@pytest.mark.asyncio
async def test_durable_action_replays_without_reinvoking_or_duplicate_audit(
    dbos_env: None,
) -> None:
    """Replaying an action uses its recorded result and recorded audit event."""
    calls = 0

    @DBOS.workflow()
    async def workflow() -> str:
        async def invoke() -> str:
            nonlocal calls
            calls += 1
            return "native result"

        return await run_durable_action(
            state=state_with_call("call-1"),
            source="sandbox_capability",
            owner="shell",
            action="exec_command",
            call_id="call-1",
            audit_stream_key="audit",
            invoke=invoke,
        )

    assert await workflow() == "native result"
    workflow_id = (await DBOS.list_workflows_async())[0].workflow_id
    events = await _stream_events(workflow_id, "audit")
    assert events == [
        {
            "source": "sandbox_capability",
            "owner": "shell",
            "action": "exec_command",
            "call_id": "call-1",
            "status": "completed",
        }
    ]

    steps = await DBOS.list_workflow_steps_async(workflow_id)
    replay = await DBOS.fork_workflow_async(workflow_id, steps[-1]["function_id"] + 1)
    assert await replay.get_result() == "native result"
    assert calls == 1
    assert await _stream_events(workflow_id, "audit") == events


@pytest.mark.asyncio
async def test_durable_action_records_sanitized_failure(dbos_env: None) -> None:
    """A native exception records only the approved failed audit event."""

    @DBOS.workflow()
    async def workflow() -> str:
        async def invoke() -> str:
            raise RuntimeError("result and callback arguments stay private")

        try:
            await run_durable_action(
                state=state_with_call("call-2"),
                source="sandbox_capability",
                owner="shell",
                action="exec_command",
                call_id="call-2",
                audit_stream_key="audit",
                invoke=invoke,
            )
        except RuntimeError:
            return "handled"
        raise AssertionError("expected native failure")

    assert await workflow() == "handled"
    workflow_id = (await DBOS.list_workflows_async())[0].workflow_id
    assert await _stream_events(workflow_id, "audit") == [
        {
            "source": "sandbox_capability",
            "owner": "shell",
            "action": "exec_command",
            "call_id": "call-2",
            "status": "failed",
        }
    ]


class _SyncTurnstile:
    def wait_for(self, call_id: str) -> None:
        return None

    def allow_next_after(self, call_id: str) -> None:
        return None


def sync_state_with_call(call_id: str) -> SimpleNamespace:
    del call_id
    return SimpleNamespace(turnstile=_SyncTurnstile())


def test_sync_durable_action_replays_without_reinvoking(dbos_env: None) -> None:
    """The synchronous helper persists the native callback result for replay."""
    from dbos_openai_agents.durability import run_durable_action_sync

    calls = 0

    @DBOS.workflow()
    def workflow() -> str:
        def invoke() -> str:
            nonlocal calls
            calls += 1
            return "native result"

        return run_durable_action_sync(
            state=sync_state_with_call("call-3"),
            source="computer_capability",
            owner="desktop",
            action="click",
            call_id="call-3",
            audit_stream_key="audit",
            invoke=invoke,
        )

    assert workflow() == "native result"
    workflow_id = DBOS.list_workflows()[0].workflow_id
    assert list(DBOS.read_stream(workflow_id, "audit")) == [
        {
            "source": "computer_capability",
            "owner": "desktop",
            "action": "click",
            "call_id": "call-3",
            "status": "completed",
        }
    ]
    steps = DBOS.list_workflow_steps(workflow_id)
    replay = DBOS.fork_workflow(workflow_id, steps[-1]["function_id"] + 1)
    assert replay.get_result() == "native result"
    assert calls == 1


@pytest.mark.asyncio
async def test_dbos_capability_makes_real_shell_invocation_durable(
    dbos_env: None,
) -> None:
    native_invocations = 0
    with tempfile.TemporaryDirectory(
        prefix=".shell-data-", dir=Path("tests")
    ) as directory:
        data_dir = Path(directory)
        (data_dir / "brief.txt").write_text("durable shell test")

        def configure_tools(toolset: ShellToolSet) -> None:
            nonlocal native_invocations
            original = toolset.exec_command.on_invoke_tool

            async def counted(context: object, raw_input: str) -> object:
                nonlocal native_invocations
                native_invocations += 1
                return await original(context, raw_input)  # type: ignore[arg-type]

            toolset.exec_command.on_invoke_tool = counted

        agent = SandboxAgent(
            name="shell",
            model=FakeModel(
                [
                    make_tool_call_response(
                        "call-1",
                        "exec_command",
                        '{"cmd":"cat /workspace/data/brief.txt"}',
                    ),
                    make_message_response("done"),
                ]
            ),
            default_manifest=Manifest(entries={"data": LocalDir(src=data_dir)}),
            capabilities=[DBOSCapability(Shell(configure_tools=configure_tools))],
        )

        @DBOS.workflow()
        async def workflow() -> str:
            result = await DBOSRunner.run(
                agent,
                "read the brief",
                run_config=RunConfig(
                    sandbox=SandboxRunConfig(client=UnixLocalSandboxClient())
                ),
            )
            return str(result.final_output)

        assert await workflow() == "done"
        workflow_id = (await DBOS.list_workflows_async())[0].workflow_id
        steps = await DBOS.list_workflow_steps_async(workflow_id)
        assert any(step["function_name"] == "_native_action_step" for step in steps)
        events = await _stream_events(workflow_id, "dbos-capability-events")
        assert {
            "source": "sandbox_capability",
            "owner": "shell",
            "action": "exec_command",
            "call_id": "call-1",
            "status": "completed",
        } in events

        replay = await DBOS.fork_workflow_async(
            workflow_id, steps[-1]["function_id"] + 1
        )
        assert await replay.get_result() == "done"
        assert native_invocations == 1


@pytest.mark.asyncio
async def test_dbos_capability_retains_filesystem_apply_patch_callback(
    dbos_env: None,
) -> None:
    native_invocations = 0

    def configure_tools(toolset: FilesystemToolSet) -> None:
        nonlocal native_invocations
        original = toolset.apply_patch.on_invoke_tool

        async def counted(context: object, raw_input: str) -> object:
            nonlocal native_invocations
            native_invocations += 1
            return await original(context, raw_input)  # type: ignore[arg-type]

        toolset.apply_patch.on_invoke_tool = counted

    agent = SandboxAgent(
        name="filesystem",
        model=FakeModel(
            [
                ModelResponse(
                    output=[
                        ResponseCustomToolCall(
                            type="custom_tool_call",
                            call_id="call-2",
                            name="apply_patch",
                            input="*** Begin Patch\n*** Add File: durable.txt\n+durable\n*** End Patch\n",
                        )
                    ],
                    usage=Usage(),
                    response_id="resp-1",
                ),
                make_message_response("done"),
            ]
        ),
        capabilities=[DBOSCapability(Filesystem(configure_tools=configure_tools))],
    )

    @DBOS.workflow()
    async def workflow() -> str:
        result = await DBOSRunner.run(
            agent,
            "make the file",
            run_config=RunConfig(
                sandbox=SandboxRunConfig(client=UnixLocalSandboxClient())
            ),
        )
        return str(result.final_output)

    assert await workflow() == "done"
    assert native_invocations == 1
