# DBOS Durable OpenAI Agents

Durable execution for the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) using [DBOS](https://github.com/dbos-inc/dbos-transact-py).

## Installation

```bash
pip install dbos-openai-agents
```

## Usage

Call your agent using `DBOSRunner.run()` from a `@DBOS.workflow()`.
Annotate tool calls and guardrails with `@DBOS.step()`.

```python
import asyncio
from agents import Agent, function_tool
from dbos import DBOS, DBOSConfig
from dbos_openai_agents import DBOSRunner

# Decorate tool calls and guardrails with @DBOS.step() for durable execution
@function_tool
@DBOS.step()
async def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"Sunny in {city}"

agent = Agent(name="weather", tools=[get_weather])

# Use DBOSRunner to call your agent from a workflow
@DBOS.workflow()
async def run_agent(user_input: str) -> str:
    result = await DBOSRunner.run(agent, user_input)
    return str(result.final_output)


async def main():
    output = await run_agent("How is the weather in San Francisco")
    print(output)


if __name__ == "__main__":
    config: DBOSConfig = {
        "name": "my-agent",
    }
    DBOS(config=config)
    DBOS.launch()
    asyncio.run(main())
```

`DBOSRunner.run()` is a drop-in replacement for `Runner.run()` with the same arguments.
It must be called from within a `@DBOS.workflow()`.

## Local PostgreSQL environment

The repository includes a local PostgreSQL 16 environment for DBOS-backed examples.
Start it before configuring or launching DBOS:

```bash
docker compose up -d
```

With the defaults, use `postgresql://dbos:dbos@localhost:7432/dbos` as the DBOS
database URL. `POSTGRES_DB`, `POSTGRES_USER`, and `POSTGRES_PASSWORD` can be
overridden in your shell when starting Compose; the provided values are intended
only for local development. If port 7432 is in use, choose an available host port
when starting Compose and set `DBOS_DATABASE_URL` to the matching PostgreSQL URL
before running the notebook. Stop the environment with `docker compose down`.

See `notebooks/durable_agents_examples.ipynb` for regular, sandboxed, and
agent-as-tool examples.

## Durable nested agent tools

Use `DBOSAgentTool` when a coordinator agent calls another agent as a function
tool. It runs the nested agent through `DBOSRunner`, so its model calls are
recorded as durable DBOS steps too.

```python
from agents import Agent
from dbos import DBOS
from dbos_openai_agents import DBOSAgentTool, DBOSRunner

researcher = Agent(name="researcher")
coordinator = Agent(
    name="coordinator",
    tools=[DBOSAgentTool(researcher, tool_name="ask_researcher", tool_description="Research a question.")],
)

@DBOS.workflow()
async def coordinate(question: str) -> str:
    result = await DBOSRunner.run(coordinator, question)
    return str(result.final_output)
```

Plain `agent.as_tool()` does not make nested-agent execution durable through
this package; use `DBOSAgentTool` for durable nested runs.

Handoffs already receive recursive DBOS wrapping through `DBOSRunner`, so they
do not need a separate handoff helper. If a `DBOSAgentTool` has a fixed
`stream_key`, invoke that configured tool at most once per workflow; each
invocation writes to the same DBOS stream key.

## Durable sandbox shell tools

Wrap a capability that performs native actions with `DBOSCapability` and run the
`SandboxAgent` through `DBOSRunner` from a DBOS workflow:

```python
from agents import RunConfig
from agents.sandbox import SandboxAgent, SandboxRunConfig
from agents.sandbox.capabilities import Shell
from agents.sandbox.sandboxes import UnixLocalSandboxClient
from dbos import DBOS
from dbos_openai_agents import DBOSCapability, DBOSRunner

sandbox_agent = SandboxAgent(
    name="shell_agent",
    capabilities=[DBOSCapability(Shell())],
)

@DBOS.workflow()
async def run_sandbox_agent(user_input: str) -> str:
    result = await DBOSRunner.run(
        sandbox_agent,
        user_input,
        run_config=RunConfig(
            sandbox=SandboxRunConfig(client=UnixLocalSandboxClient()),
        ),
    )
    return str(result.final_output)
```

The wrapper persists each native action as `_native_action_step`, then writes a
payload-free `dbos-capability-events` record. Forking a completed workflow after its
last function ID reuses the saved operation output, so the shell command is not run
again.

If a failure occurs after a shell action completes but before DBOS persists the
step's success result, a retry can run that external action again. Shell actions are
therefore at-least-once and should be idempotent.

`UnixLocalSandboxClient` is intended for local Unix development, not an isolation or
deployment boundary for untrusted work. The audit stream contains action metadata
only (source, owner, action, call ID, and status); command arguments and results are
not written there. Operation outputs can contain tool results, so PostgreSQL audit
queries belong behind a trusted database-access boundary with appropriate retention
and access controls.

## Streaming

`DBOSRunner.run_streamed()` is a drop-in replacement for `Runner.run_streamed()`.
Pass a `stream_key` to persist every typed `RawResponsesStreamEvent` live as the
provider emits it, including `response.completed`. The completed raw-event list is
also stored in the durable model step so the Agents SDK can replay completed model
calls without calling the provider again.

```python
from dbos import DBOS, SetWorkflowID
from dbos_openai_agents import DBOSRunner

AGENT_STREAM_KEY = "agent-events"

@DBOS.workflow()
async def stream_agent(user_input: str) -> str:
    result = DBOSRunner.run_streamed(
        agent,
        user_input,
        stream_key=AGENT_STREAM_KEY,
    )
    # Drive the Agents SDK stream. Render from the durable DBOS stream instead.
    async for _ in result.stream_events():
        pass
    return str(result.final_output)

with SetWorkflowID(request_id):
    handle = await DBOS.start_workflow_async(stream_agent, user_input)
```

### Attaching and resuming

Use `DBOSRunner.attach_stream()` to consume the durable stream. It returns
`(offset, event)` tuples, where `offset` is the next resumable DBOS stream position.
Persist the last successfully applied offset and supply it when reconnecting.

```python
last_offset = 0

async for last_offset, event in DBOSRunner.attach_stream(
    handle.get_workflow_id(),
    AGENT_STREAM_KEY,
    offset=last_offset,
):
    render(event)
```

The default `replay="raw"` mode skips directly to the supplied offset. This is the
normal reconnect path when the caller already rendered everything before that
cursor.

`replay="compact"` rebuilds historical UI state efficiently: it reads events from
stream offset 0 through the supplied offset, merges consecutive compatible SDK
string-delta events into ordered logical blocks, emits those compacted blocks with
the cursor of the final raw event they represent, and then resumes ordinary
one-event streaming from the supplied offset.

```python
async for cursor, event in DBOSRunner.attach_stream(
    handle.get_workflow_id(),
    AGENT_STREAM_KEY,
    offset=last_offset,
    replay="compact",
):
    render(event)
```

Compaction preserves event order and logical boundaries. Reasoning deltas, output
text deltas, and streamed tool-call arguments are compacted only while consecutive
and associated with the same SDK event identity. Lifecycle and non-delta events
remain separate.

### Timeouts and polling

`attach_stream()` passes `polling_interval_sec` and `timeout_seconds` directly to
DBOS stream reads. A timeout is an inter-event timeout: DBOS restarts the timeout
after each value is received.

```python
async for cursor, event in DBOSRunner.attach_stream(
    handle.get_workflow_id(),
    AGENT_STREAM_KEY,
    offset=last_offset,
    polling_interval_sec=0.25,
    timeout_seconds=30,
):
    render(event)
```

This is useful for detecting a producer that has stopped delivering events without
placing a total-duration limit on a long-running agent.

### Recovery and duplicate-prefix protection

DBOS guarantees exactly-once stream writes when `DBOS.write_stream()` is called
from workflow code, but writes made from a step are at-least-once. The provider
stream must execute inside a durable model step so completed model calls can replay
without another provider request while still delivering tokens live.

To avoid duplicating a prefix when DBOS recovers an incomplete model-stream step,
this package tracks the expected global stream offset. At the beginning of a model
stream it probes the expected offset. If values from the interrupted attempt are
already present, the recovered step consumes and skips those already-persisted
positions until it reaches the first missing offset, then resumes ordinary live
writes. On a fresh model stream only the first expected offset is probed, so normal
streaming does not perform a database read for every token.

This protects the common executor-crash/recovery case while preserving live token
streaming. A narrow edge case remains under concurrent or "zombie" execution: two
executors can race before either write becomes visible and DBOS step stream writes
are still at-least-once, so duplicate transient stream events are possible. The
durable Agents SDK session/response state remains the source of truth once a
response completes; the DBOS stream is used for live delivery and reconnect state,
not as the authoritative completed response record.

The stream is closed when SDK stream consumption finishes. After consuming the
stream, call `handle.get_result()` to surface any terminal workflow error.

`process_stream()` remains an optional compatibility helper for forwarding an
Agents SDK result stream. It does not write or close DBOS streams; use
`run_streamed(..., stream_key=...)` and `DBOSRunner.attach_stream()` for durable,
resumable streaming. Typed event payloads can contain text, reasoning, and tool-call
data, so protect the system database and stream readers appropriately.
