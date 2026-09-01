# Durable Sandbox Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in durable DBOS wrappers for local SandboxAgent capability tools and Agents SDK Computer Use actions, then demonstrate their persisted records in the notebook.

**Architecture:** `DBOSCapability` delegates to one Agents SDK `Capability`, making its local executable tools durable. `DBOSComputerTool` returns a `ComputerTool` backed by a durable local computer proxy. Both share one internal executor that applies DBOS step replay, runner ordering, and sanitized audit stream records.

**Tech Stack:** Python, DBOS, OpenAI Agents SDK, pytest, UnixLocalSandboxClient, psycopg, Jupyter.

**Spec:** `docs/superpowers/specs/2026-08-31-durable-sandbox-tools-design.md`

## Global Constraints

- Keep `DBOSRunner.run`, `run_sync`, and `run_streamed` compatible.
- Instrument only caller-wrapped local operations; do not alter unwrapped or hosted tools.
- Preserve capability lifecycle methods, SDK callbacks, safety callbacks, and ComputerTool direct/factory/provider initialization.
- A replay must use a saved step output and not repeat the native side effect.
- Audit records contain only source, owner, tool-call ID, action name, and status; they never include commands, patch content, typed text, screenshots, or outputs.
- The Shell integration test uses `UnixLocalSandboxClient` with `TemporaryDirectory(dir=tests/)`.
- Preserve the user-owned Compose port and notebook model/tracing settings.

---

## File Structure

- Create `dbos_openai_agents/durability.py`: generic DBOS action step and audit events.
- Create `dbos_openai_agents/capabilities.py`: `DBOSCapability`.
- Create `dbos_openai_agents/computer.py`: `DBOSComputerTool` and computer proxies.
- Modify `dbos_openai_agents/runner.py`: inject per-run state into cloned DBOS capabilities.
- Modify `dbos_openai_agents/__init__.py`: export public wrappers.
- Create `tests/test_durable_sandbox_tools.py`: real Shell and fake Computer Use integration coverage.
- Modify `tests/test_notebook_examples.py`, `notebooks/durable_agents_examples.ipynb`, and `README.md`: examples and evidence queries.

## Task 1: Durable action executor

**Files:**
- Create: `dbos_openai_agents/durability.py`
- Create: `tests/test_durable_sandbox_tools.py`

**Interfaces:**
- Produces `DEFAULT_AUDIT_STREAM_KEY = "dbos-capability-events"`.
- Produces `@DBOS.step() async def _native_action_step(invoke: Callable[[], Awaitable[T]]) -> T`.
- Produces `async def run_durable_action(*, state, source, owner, action, call_id, audit_stream_key, invoke) -> T`.

- [ ] **Step 1: Write a failing replay and audit test**

```python
@pytest.mark.asyncio
async def test_durable_action_replays_without_reinvoking(dbos_env):
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
    steps = await DBOS.list_workflow_steps_async(workflow_id)
    replay = await DBOS.fork_workflow_async(workflow_id, steps[-1]["function_id"] + 1)
    assert await replay.get_result() == "native result"
    assert calls == 1
```

Assert the `audit` stream contains exactly `source`, `owner`, `action`, `call_id`, and `status="completed"`; assert no callback arguments or result occur in the event.

- [ ] **Step 2: Verify the test is red**

Run: `uv run pytest tests/test_durable_sandbox_tools.py::test_durable_action_replays_without_reinvoking -q`

Expected: collection fails because `dbos_openai_agents.durability` does not exist.

- [ ] **Step 3: Implement the executor**

```python
@DBOS.step()
async def _native_action_step(invoke: Callable[[], Awaitable[T]]) -> T:
    return await invoke()

async def run_durable_action(
    *, state, source, owner, action, call_id, audit_stream_key, invoke
):
    if call_id is not None:
        await state.turnstile.wait_for(call_id)
        state.turnstile.allow_next_after(call_id)
    try:
        result = await _native_action_step(invoke)
    except Exception:
        await DBOS.write_stream_async(audit_stream_key, event(status="failed"))
        raise
    await DBOS.write_stream_async(audit_stream_key, event(status="completed"))
    return result
```

Use a TypedDict for the event, permit `call_id=None`, and write no unapproved payload fields.

- [ ] **Step 4: Verify green**

Run: `uv run pytest tests/test_durable_sandbox_tools.py::test_durable_action_replays_without_reinvoking tests/test_dbos_runner.py::test_replay tests/test_dbos_runner.py::test_streamed_tool_call_replays_durably -q`

Expected: PASS and the native callback count remains one.

- [ ] **Step 5: Commit**

```bash
git add dbos_openai_agents/durability.py tests/test_durable_sandbox_tools.py
git commit -m "feat: add durable native action executor"
```

## Task 2: Sandbox capability wrapper

**Files:**
- Create: `dbos_openai_agents/capabilities.py`
- Modify: `dbos_openai_agents/runner.py`
- Modify: `dbos_openai_agents/__init__.py`
- Modify: `tests/test_durable_sandbox_tools.py`

**Interfaces:**
- Produces `DBOSCapability(capability: Capability, *, audit_stream_key: str = DEFAULT_AUDIT_STREAM_KEY)`.
- Produces private `bind_durability_state(state) -> None`.
- Consumes `run_durable_action` from Task 1.

- [ ] **Step 1: Write the real Shell durability test**

```python
@pytest.mark.asyncio
async def test_dbos_capability_makes_real_shell_invocation_durable(dbos_env):
    native_invocations = 0
    with tempfile.TemporaryDirectory(prefix=".shell-data-", dir=Path("tests")) as directory:
        data_dir = Path(directory)
        (data_dir / "brief.txt").write_text("durable shell test")

        def configure_tools(toolset):
            original = toolset.exec_command.on_invoke_tool
            async def counted(context, raw_input):
                nonlocal native_invocations
                native_invocations += 1
                return await original(context, raw_input)
            toolset.exec_command = dataclasses.replace(
                toolset.exec_command, on_invoke_tool=counted
            )

        agent = SandboxAgent(
            name="shell",
            model=FakeModel([
                make_tool_call_response("call-1", "exec_command",
                                        '{"cmd":"cat /workspace/data/brief.txt"}'),
                make_message_response("done"),
            ]),
            default_manifest=Manifest(entries={"data": LocalDir(src=data_dir)}),
            capabilities=[DBOSCapability(Shell(configure_tools=configure_tools))],
        )
```

Finish the test by running the agent with `DBOSRunner.run` and `UnixLocalSandboxClient`, forking past the last recorded step, and asserting `native_invocations == 1`. Assert a `_native_action_step` and a sanitized `shell/exec_command` stream event exist. Add a Filesystem custom-tool test proving an existing `apply_patch` callback is retained.

- [ ] **Step 2: Verify the Shell test is red**

Run: `uv run pytest tests/test_durable_sandbox_tools.py::test_dbos_capability_makes_real_shell_invocation_durable -q`

Expected: import failure for `DBOSCapability`.

- [ ] **Step 3: Implement DBOSCapability**

```python
class DBOSCapability(Capability):
    capability: Capability
    audit_stream_key: str = DEFAULT_AUDIT_STREAM_KEY

    def clone(self) -> DBOSCapability:
        return DBOSCapability(self.capability.clone(),
                              audit_stream_key=self.audit_stream_key)

    def tools(self) -> list[Tool]:
        return [instrument_tool(tool, self._state, self.type, self.audit_stream_key)
                for tool in self.capability.tools()]
```

Delegate `bind`, `bind_run_as`, `bind_workspace_scope`, `required_capability_types`, `process_manifest`, `instructions`, `sampling_params`, and `process_context`. For both `FunctionTool` and `CustomTool`, use `dataclasses.replace` and replace only `on_invoke_tool`; unsupported tool variants pass through untouched. In `runner._wrap_agent`, make run-local clones of only `DBOSCapability` instances, bind the current state, and supply those clones to `SandboxAgent.clone`.

- [ ] **Step 4: Verify green**

Run: `uv run pytest tests/test_durable_sandbox_tools.py tests/test_dbos_runner.py -q && uv run mypy dbos_openai_agents tests && uv run black --check dbos_openai_agents tests`

Expected: PASS, including the real `cat` command under UnixLocalSandboxClient.

- [ ] **Step 5: Commit**

```bash
git add dbos_openai_agents/capabilities.py dbos_openai_agents/runner.py dbos_openai_agents/__init__.py tests/test_durable_sandbox_tools.py
git commit -m "feat: add durable sandbox capability wrapper"
```

## Task 3: Computer Use wrapper

**Files:**
- Create: `dbos_openai_agents/computer.py`
- Modify: `dbos_openai_agents/__init__.py`
- Modify: `tests/test_durable_sandbox_tools.py`

**Interfaces:**
- Produces `DBOSComputerTool(tool: ComputerTool[T], *, audit_stream_key: str = DEFAULT_AUDIT_STREAM_KEY) -> ComputerTool[T]`.
- Produces sync and async local computer proxies.

- [ ] **Step 1: Write failing direct, async, and factory harness tests**

```python
@pytest.mark.asyncio
async def test_dbos_computer_tool_replays_click_and_screenshot_once(dbos_env):
    computer = RecordingComputer()
    tool = DBOSComputerTool(
        ComputerTool(computer=computer), audit_stream_key="computer-audit"
    )

    @DBOS.workflow()
    async def workflow() -> str:
        tool.computer.click(10, 20, "left")
        return tool.computer.screenshot()

    assert await workflow() == "image-1"
    workflow_id = (await DBOS.list_workflows_async())[0].workflow_id
    steps = await DBOS.list_workflow_steps_async(workflow_id)
    replay = await DBOS.fork_workflow_async(workflow_id, steps[-1]["function_id"] + 1)
    assert await replay.get_result() == "image-1"
    assert computer.calls == [("click", 10, 20, "left"), ("screenshot",)]
```

Add `AsyncRecordingComputer` coverage for awaited `type` and `screenshot`. Add a factory test whose factory returns a recording computer and verify that it resolves to a durable proxy. Assert events use `source="computer_use"` and method names without typed text or screenshot data.

- [ ] **Step 2: Verify the direct computer test is red**

Run: `uv run pytest tests/test_durable_sandbox_tools.py::test_dbos_computer_tool_replays_click_and_screenshot_once -q`

Expected: import failure for `DBOSComputerTool`.

- [ ] **Step 3: Implement DBOSComputerTool and proxies**

```python
def DBOSComputerTool(tool: ComputerTool[T], *,
                     audit_stream_key: str = DEFAULT_AUDIT_STREAM_KEY) -> ComputerTool[T]:
    return ComputerTool(
        computer=wrap_computer_initializer(tool.computer, audit_stream_key),
        on_safety_check=tool.on_safety_check,
        custom_data_extractor=tool.custom_data_extractor,
    )
```

Implement `Computer` and `AsyncComputer` proxies that delegate `screenshot`, `click`, `double_click`, `scroll`, `type`, `wait`, `move`, `keypress`, and `drag` through `run_durable_action`. `wrap_computer_initializer` must handle direct values, factories, and providers; its provider disposal callback unwraps the proxy before calling the original provider. Keep safety/custom-data callbacks identical.

- [ ] **Step 4: Verify green**

Run: `uv run pytest tests/test_durable_sandbox_tools.py -q && uv run mypy dbos_openai_agents tests && uv run black --check dbos_openai_agents tests`

Expected: PASS and replay adds no recording-computer calls.

- [ ] **Step 5: Commit**

```bash
git add dbos_openai_agents/computer.py dbos_openai_agents/__init__.py tests/test_durable_sandbox_tools.py
git commit -m "feat: add durable computer tool wrapper"
```

## Task 4: Notebook, SQL evidence, and documentation

**Files:**
- Modify: `README.md`
- Modify: `notebooks/durable_agents_examples.ipynb`
- Modify: `tests/test_notebook_examples.py`

**Interfaces:**
- Consumes the wrappers and DBOS workflow IDs.
- Produces an executable PostgreSQL evidence cell.

- [ ] **Step 1: Extend the notebook-content test**

```python
assert "DBOSCapability(Shell())" in source
assert "await DBOSRunner.run(" in source
assert "DBOS.fork_workflow_async" in source
assert "dbos.workflow_status" in source
assert "dbos.operation_outputs" in source
assert "dbos.streams" in source
assert "_native_action_step" in source
assert "dbos-capability-events" in source
```

- [ ] **Step 2: Verify the notebook test is red**

Run: `uv run pytest tests/test_notebook_examples.py -q`

Expected: FAIL on missing `DBOSCapability(Shell())`.

- [ ] **Step 3: Update the notebook and README**

Keep the user-selected models, disabled tracing, database URL port, and Compose directions. Replace the upstream SandboxAgent run with `DBOSCapability(Shell())` and a `@DBOS.workflow` that awaits `DBOSRunner.run(sandbox_agent, user_input, run_config=RunConfig(sandbox=SandboxRunConfig(client=UnixLocalSandboxClient())))`. Start sandbox and coordinator workflows with `DBOS.start_workflow_async` and retain all three workflow IDs. Remove the unused `Path("tmp")` cell.

Add a replay cell that forks the sandbox workflow after its last function ID and prints original/replay IDs and outputs. Add a final `psycopg` cell that runs these parameterized SQL statements for the three original IDs and the sandbox replay ID:

```sql
SELECT workflow_uuid, name, status, created_at, updated_at
FROM dbos.workflow_status
WHERE workflow_uuid = ANY(%s::uuid[])
ORDER BY created_at;

SELECT workflow_uuid, function_id, function_name, error IS NOT NULL AS failed
FROM dbos.operation_outputs
WHERE workflow_uuid = ANY(%s::uuid[])
ORDER BY workflow_uuid, function_id;

SELECT workflow_uuid, key, "offset", value
FROM dbos.streams
WHERE workflow_uuid = ANY(%s::uuid[])
  AND key = 'dbos-capability-events'
ORDER BY workflow_uuid, "offset";
```

Print labeled result tables and state that the initial workflow has the `_native_action_step`; the fork receives its saved operation output rather than performing another shell command. Add the same usage/limitation/audit-boundary explanation to README.

- [ ] **Step 4: Verify full integration**

Run: `uv run python -c 'import ast, nbformat; nb=nbformat.read("notebooks/durable_agents_examples.ipynb", as_version=4); [compile(cell.source, f"cell-{i}", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT) for i, cell in enumerate(nb.cells) if cell.cell_type == "code"]' && uv run pytest -q && uv run mypy dbos_openai_agents tests && uv run black --check dbos_openai_agents tests`

Expected: PASS; every code cell compiles and all tests/type/format checks succeed.

- [ ] **Step 5: Commit only feature-owned files**

```bash
git add README.md notebooks/durable_agents_examples.ipynb tests/test_notebook_examples.py
git commit -m "docs: demonstrate durable sandbox tool records"
git status --short
```

Leave `docker-compose.yml` and unrelated notebook artifacts unstaged.
