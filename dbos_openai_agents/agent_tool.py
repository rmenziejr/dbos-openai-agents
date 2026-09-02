"""Durable OpenAI Agent SDK agent tools."""

from collections.abc import Collection
from typing import Any

from agents import Agent, RunConfig
from agents.tool import FunctionTool, function_tool

from .runner import DBOSRunner
from .streaming import StreamEventKind, process_stream


def DBOSAgentTool(
    agent: Agent[Any],
    *,
    tool_name: str,
    tool_description: str,
    run_config: RunConfig | None = None,
    stream_key: str | None = None,
    include: Collection[StreamEventKind] = ("text", "reasoning", "tool_calls"),
) -> FunctionTool:
    """Return a FunctionTool that runs a nested agent through DBOSRunner."""

    @function_tool(
        name_override=tool_name,
        description_override=tool_description,
    )
    async def invoke(input: str) -> str:
        run_kwargs: dict[str, Any] = {}
        if run_config is not None:
            run_kwargs["run_config"] = run_config

        if stream_key is None:
            result = await DBOSRunner.run(agent, input, **run_kwargs)
            return str(result.final_output)

        streaming_result = DBOSRunner.run_streamed(agent, input, **run_kwargs)
        async for _ in process_stream(streaming_result, stream_key, include=include):
            pass
        return str(streaming_result.final_output)

    return invoke


__all__ = ["DBOSAgentTool"]
