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

With the defaults, use `postgresql://dbos:dbos@localhost:5432/dbos` as the DBOS
database URL. `POSTGRES_DB`, `POSTGRES_USER`, and `POSTGRES_PASSWORD` can be
overridden in your shell when starting Compose; the provided values are intended
only for local development. If port 5432 is in use, start with
`POSTGRES_PORT=5433 docker compose up -d` and set
`DBOS_DATABASE_URL=postgresql://dbos:dbos@localhost:5433/dbos` before running
the notebook. Stop the environment with `docker compose down`.

See `notebooks/durable_agents_examples.ipynb` for regular, sandboxed, and
agent-as-tool examples.

## Streaming

`DBOSRunner.run_streamed()` is a drop-in replacement for `Runner.run_streamed()`. Consume the returned result with `stream_events()` inside the workflow:

```python
@DBOS.workflow()
async def stream_agent(user_input: str) -> str:
    result = DBOSRunner.run_streamed(agent, user_input)
    async for event in result.stream_events():
        # Handle the OpenAI Agents SDK stream event.
        print(event)
    return str(result.final_output)
```

Each model response is persisted as a DBOS step before its events are yielded. This keeps replay durable, but events are emitted after their model-response step completes rather than token-by-token as they arrive from the provider.
