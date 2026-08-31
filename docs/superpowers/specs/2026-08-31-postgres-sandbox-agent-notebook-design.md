# PostgreSQL and SandboxAgent Example Design

## Goal

Provide a reproducible local PostgreSQL environment and one Jupyter notebook that demonstrates three OpenAI Agents SDK patterns alongside DBOS durability: a regular agent, a SandboxAgent, and an agent exposed as a tool.

## Scope

- Raise the minimum OpenAI Agents SDK dependency to `0.14.0` so `SandboxAgent` is available.
- Add notebook tooling to the development dependencies.
- Add a root Compose file that runs PostgreSQL 16 for local DBOS development.
- Add one notebook at `notebooks/durable_agents_examples.ipynb`.

## Local Database Environment

The Compose file defines one `postgres:16` service with a health check and a named volume. It publishes port 5432 and supplies local-only defaults: database `dbos`, user `dbos`, and password `dbos`. The notebook connects as `postgresql://dbos:dbos@localhost:5432/dbos`; DBOS initializes its own system tables.

The README will state the startup command and that users can override credentials through Compose environment variables.

## Notebook Design

The notebook starts with a prerequisites cell that verifies `OPENAI_API_KEY`, imports dependencies, and checks the PostgreSQL connection. It then initializes DBOS using the Compose connection string.

### Regular agent

A standard `Agent` is run through `DBOSRunner.run()` inside a DBOS workflow. Its local function tool is DBOS-step annotated, illustrating durable model and tool execution.

### SandboxAgent

A `SandboxAgent` uses a `Manifest` with an explicitly mounted temporary data directory and `UnixLocalSandboxClient`. The example runs through the upstream `Runner` with `SandboxRunConfig`, matching the official SDK setup. Its sandbox contains only the mounted example data; database and API credentials are not passed into it.

### Agent as a tool

A specialist `Agent` is converted through `.as_tool()` and supplied to a DBOS-backed coordinator agent. The coordinator is run through `DBOSRunner.run()` in a workflow, showing agent delegation without a handoff.

Each section prints its final output and, where DBOS is used, the generated workflow ID and recorded step names.

## Failure Handling

The notebook stops early with actionable messages when the API key is unset, PostgreSQL is unavailable, or the local sandbox client cannot start. The setup does not attempt to run Docker or install system packages from a notebook cell.

## Verification

- `docker compose config` validates the Compose model.
- `nbformat` validates notebook structure and required cells.
- Dependency resolution confirms the SandboxAgent imports are supplied by the declared minimum Agents SDK version.
- The existing test suite remains green.

## Non-goals

This change does not make `SandboxAgent` durable through `DBOSRunner`; the official sandbox example uses the upstream Runner. It also does not provision a remote sandbox provider or persist real API credentials.
