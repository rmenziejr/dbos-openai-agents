# Durable Sandbox Capabilities and Computer Use

## Goal

Make selected locally executed OpenAI Agents SDK sandbox capabilities and
computer-use actions durable DBOS steps. A replay must reuse a saved action
result instead of running the side effect again. The integration must preserve
the SDK's native configuration and opt-in selection model.

## Public API

Expose two opt-in wrappers from `dbos_openai_agents`.

```python
from agents import Agent, ComputerTool
from agents.sandbox import SandboxAgent
from agents.sandbox.capabilities import Filesystem, Shell
from dbos_openai_agents import DBOSCapability, DBOSComputerTool

sandbox_agent = SandboxAgent(
    name="sandbox",
    capabilities=[
        DBOSCapability(Shell()),
        DBOSCapability(Filesystem()),
    ],
)

computer_agent = Agent(
    name="browser",
    tools=[DBOSComputerTool(ComputerTool(computer=my_computer))],
)
```

`DBOSCapability` accepts one Agents SDK `Capability` and presents the same
capability interface. `DBOSComputerTool` accepts one Agents SDK
`ComputerTool` and returns a compatible computer tool. Neither changes the
behaviour of capabilities or computer tools that the caller does not wrap.

## Components

### DBOSCapability

The adapter delegates all capability lifecycle methods to its wrapped
capability: cloning, sandbox-session binding, user binding, workspace-scope
binding, manifest processing, instructions, sampling parameters, and context
processing. This preserves capabilities supplied by the SDK now and in future
releases without a capability-specific DBOS API.

When the SDK asks for tools, the adapter instruments local executable tool
types, including `FunctionTool` and `CustomTool`. The instrumented invocation
keeps the existing DBOS runner turnstile ordering and executes the native
callback through the shared durable action step.

Tools with no local callback are returned unchanged. Their execution remains
covered by the durable model turn, but the package cannot turn an external or
hosted operation into a locally controlled DBOS step.

### DBOSComputerTool

Computer Use is an Agents SDK `ComputerTool` in `Agent.tools`, not a sandbox
`Capability`. `DBOSComputerTool` replaces its computer harness with a proxy.
The proxy delegates all supported sync or async computer methods, such as
screenshots, clicks, typing, scrolling, key presses, and drag operations, to
the original harness through the shared durable action step.

The wrapper must preserve `ComputerTool` callbacks and per-run
computer-initializer semantics. It will support a direct computer object and
the SDK's factory/provider forms.

### Shared instrumentation

One internal action-step helper executes a native callback as a DBOS step. It
receives an action descriptor and a callback, so the callback's output becomes
the durable step output. The wrapper uses the existing run state to allocate
tool calls deterministically when the SDK schedules multiple calls.

The helper writes a small DBOS workflow-stream record on completion or
failure. Records contain only:

- source (`sandbox_capability` or `computer_use`)
- wrapped capability type or tool name
- tool call ID, when one exists
- action name
- terminal status

It excludes raw shell commands, patch content, typed text, screenshots, and
native outputs by default. The stream key is configurable for applications
that need to segregate audit records.

## Execution and replay

1. `DBOSRunner` makes a run-local copy of each DBOS wrapper and supplies its
   ordering state.
2. The Agents SDK prepares the sandbox capability or dispatches a computer
   action.
3. The wrapper waits for its deterministic turn, invokes the durable action
   step, and records sanitized metadata.
4. The first execution performs the side effect and saves its result in DBOS.
5. On replay, DBOS supplies the saved result and the native side effect is not
   invoked again.

Errors from the native handler fail the DBOS step and propagate through the
Agents SDK unchanged. The audit stream records the failed terminal status.

## Testing

Tests are integration-focused where the SDK provides a local implementation.

- A real `SandboxAgent` uses `UnixLocalSandboxClient` and
  `DBOSCapability(Shell())`.
- Its data directory comes from `TemporaryDirectory(dir=tests/)`, satisfying
  the sandbox local-directory constraint while keeping artifacts inside the
  repository test area and removing them after the test.
- The shell test performs a real command against that directory, verifies its
  durable DBOS operation/audit record, and verifies a replay does not rerun
  the command.
- A small fake sync and async computer harness verifies
  `DBOSComputerTool` makes click, type, and screenshot actions durable without
  requiring a graphical environment.
- Unit tests cover lifecycle delegation, configured stream keys, preservation
  of native tool callbacks, deterministic ordering, and safe audit payloads.

## Documentation

The README will document both wrappers and their limits. The durable examples
notebook will use `DBOSCapability(Shell())`, run the sandbox agent through
`DBOSRunner`, and query the resulting DBOS workflow, operation-output, and
audit-stream records from PostgreSQL.
