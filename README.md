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

`DBOSRunner.run_streamed()` is a drop-in replacement for `Runner.run_streamed()`. Pass an optional stream key to write every raw, typed `RawResponsesStreamEvent` live as the provider emits it (including `response.completed`) and close that DBOS stream when SDK consumption finishes. The completed raw-event list is also stored in the durable model step so the Agents SDK can replay execution.

```python
from agents.stream_events import RawResponsesStreamEvent
from dbos import DBOS, SetWorkflowID
from dbos_openai_agents import DBOSRunner

AGENT_STREAM_KEY = "agent-events"

@DBOS.workflow()
async def stream_agent(user_input: str) -> str:
    result = DBOSRunner.run_streamed(
        agent, user_input, stream_key=AGENT_STREAM_KEY
    )
    # Drive the agent; render only from the durable DBOS stream below.
    async for _ in result.stream_events():
        pass
    return str(result.final_output)

with SetWorkflowID(request_id):
    handle = await DBOS.start_workflow_async(stream_agent, user_input)

async for event in DBOS.read_stream_async(handle.get_workflow_id(), AGENT_STREAM_KEY):
    assert isinstance(event, RawResponsesStreamEvent)
    render(event)

# Surface a terminal workflow failure after stream consumption.
await handle.get_result()
```

Raw provider events are written to the keyed DBOS stream live as they arrive. Separately, each completed model response stores its complete raw-event list in a durable model step for Agents SDK execution replay.

`process_stream()` remains an optional compatibility helper for forwarding an Agents SDK result stream. It does not write or close DBOS streams; use `run_streamed(..., stream_key=...)` and `DBOS.read_stream_async()` for durable streaming. Typed event payloads can contain text, reasoning, and tool-call data, so protect the system database and stream readers appropriately. If the same completed `request_id` is started again, DBOS reuses the recorded workflow result rather than rerunning it.
