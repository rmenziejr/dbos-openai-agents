import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from agents import (
    AsyncComputer,
    Computer,
    ComputerProvider,
    ComputerTool,
    RunConfig,
    RunContextWrapper,
    Usage,
    dispose_resolved_computers,
    resolve_computer,
)
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

from dbos_openai_agents import DBOSCapability, DBOSComputerTool, DBOSRunner
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


class RecordingComputer(Computer):
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def screenshot(self) -> str:
        self.calls.append(("screenshot",))
        return f"image-{len(self.calls)}"

    def click(self, x: int, y: int, button: str) -> None:
        self.calls.append(("click", x, y, button))

    def double_click(self, x: int, y: int) -> None:
        self.calls.append(("double_click", x, y))

    def scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        self.calls.append(("scroll", x, y, scroll_x, scroll_y))

    def type(self, text: str) -> None:
        self.calls.append(("type", text))

    def wait(self) -> None:
        self.calls.append(("wait",))

    def move(self, x: int, y: int) -> None:
        self.calls.append(("move", x, y))

    def keypress(self, keys: list[str]) -> None:
        self.calls.append(("keypress", keys))

    def drag(self, path: list[tuple[int, int]]) -> None:
        self.calls.append(("drag", path))


class AsyncRecordingComputer(AsyncComputer):
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    async def screenshot(self) -> str:
        self.calls.append(("screenshot",))
        return f"image-{len(self.calls)}"

    async def click(self, x: int, y: int, button: str) -> None:
        self.calls.append(("click", x, y, button))

    async def double_click(self, x: int, y: int) -> None:
        self.calls.append(("double_click", x, y))

    async def scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        self.calls.append(("scroll", x, y, scroll_x, scroll_y))

    async def type(self, text: str) -> None:
        self.calls.append(("type", text))

    async def wait(self) -> None:
        self.calls.append(("wait",))

    async def move(self, x: int, y: int) -> None:
        self.calls.append(("move", x, y))

    async def keypress(self, keys: list[str]) -> None:
        self.calls.append(("keypress", keys))

    async def drag(self, path: list[tuple[int, int]]) -> None:
        self.calls.append(("drag", path))


@pytest.mark.asyncio
async def test_dbos_computer_tool_replays_click_and_screenshot_once(
    dbos_env: None,
) -> None:
    computer = RecordingComputer()
    tool = DBOSComputerTool(
        ComputerTool(computer=computer), audit_stream_key="computer-audit"
    )
    computer_proxy = cast(RecordingComputer, tool.computer)

    @DBOS.workflow()
    async def workflow() -> str:
        computer_proxy.click(10, 20, "left")
        return computer_proxy.screenshot()

    assert await workflow() == "image-2"
    workflow_id = (await DBOS.list_workflows_async())[0].workflow_id
    events = await _stream_events(workflow_id, "computer-audit")
    assert events == [
        {
            "source": "computer_use",
            "owner": "computer",
            "action": "click",
            "call_id": None,
            "status": "completed",
        },
        {
            "source": "computer_use",
            "owner": "computer",
            "action": "screenshot",
            "call_id": None,
            "status": "completed",
        },
    ]
    assert "left" not in str(events)
    assert "image-2" not in str(events)
    steps = await DBOS.list_workflow_steps_async(workflow_id)
    replay = await DBOS.fork_workflow_async(workflow_id, steps[-1]["function_id"] + 1)
    assert await replay.get_result() == "image-2"
    assert computer.calls == [("click", 10, 20, "left"), ("screenshot",)]


@pytest.mark.asyncio
async def test_dbos_computer_tool_replays_async_type_and_screenshot_once(
    dbos_env: None,
) -> None:
    computer = AsyncRecordingComputer()
    tool = DBOSComputerTool(ComputerTool(computer=computer), audit_stream_key="audit")
    computer_proxy = cast(AsyncRecordingComputer, tool.computer)

    @DBOS.workflow()
    async def workflow() -> str:
        await computer_proxy.type("private typed text")
        return await computer_proxy.screenshot()

    assert await workflow() == "image-2"
    workflow_id = (await DBOS.list_workflows_async())[0].workflow_id
    events = await _stream_events(workflow_id, "audit")
    assert [event["action"] for event in events] == ["type", "screenshot"]  # type: ignore[index]
    assert all(event["source"] == "computer_use" for event in events)  # type: ignore[index]
    assert "private typed text" not in str(events)
    assert "image-2" not in str(events)
    steps = await DBOS.list_workflow_steps_async(workflow_id)
    replay = await DBOS.fork_workflow_async(workflow_id, steps[-1]["function_id"] + 1)
    assert await replay.get_result() == "image-2"
    assert computer.calls == [("type", "private typed text"), ("screenshot",)]


@pytest.mark.asyncio
async def test_dbos_computer_tool_wraps_factory_and_provider_disposes_original(
    dbos_env: None,
) -> None:
    factory_computer = RecordingComputer()

    def factory(*, run_context: RunContextWrapper[None]) -> RecordingComputer:
        assert run_context.context is None
        return factory_computer

    factory_tool = DBOSComputerTool(ComputerTool(computer=factory))
    factory_context = RunContextWrapper(None)
    factory_proxy = await resolve_computer(
        tool=factory_tool, run_context=factory_context
    )
    assert isinstance(factory_proxy, Computer)
    assert factory_proxy is not factory_computer

    disposed: list[Computer] = []
    provider_computer = RecordingComputer()

    def dispose(*, run_context: RunContextWrapper[None], computer: Computer) -> None:
        assert run_context.context is None
        disposed.append(computer)

    provider_tool = DBOSComputerTool(
        ComputerTool(
            computer=ComputerProvider(
                create=lambda run_context: provider_computer,
                dispose=dispose,
            )
        )
    )
    provider_context = RunContextWrapper(None)
    provider_proxy = await resolve_computer(
        tool=provider_tool, run_context=provider_context
    )
    assert isinstance(provider_proxy, Computer)
    assert provider_proxy is not provider_computer
    await dispose_resolved_computers(run_context=provider_context)
    assert disposed == [provider_computer]
