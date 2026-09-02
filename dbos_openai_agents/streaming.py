from collections.abc import AsyncIterator, Collection
from typing import Literal

from agents.result import RunResultStreaming
from agents.stream_events import RawResponsesStreamEvent, StreamEvent
from dbos import DBOS
from openai.types.responses import ResponseOutputItemAddedEvent

StreamEventKind = Literal["text", "reasoning", "tool_calls"]


def _should_persist(event: StreamEvent, include: Collection[StreamEventKind]) -> bool:
    if not isinstance(event, RawResponsesStreamEvent):
        return False

    event_type = event.data.type
    if event_type == "response.output_text.delta":
        return "text" in include
    if event_type in {
        "response.reasoning_text.delta",
        "response.reasoning_summary_text.delta",
    }:
        return "reasoning" in include
    if isinstance(event.data, ResponseOutputItemAddedEvent):
        return "tool_calls" in include and event.data.item.type in {
            "function_call",
            "custom_tool_call",
            "mcp_call",
        }
    return "tool_calls" in include and event_type.startswith(
        ("response.function_call_", "response.custom_tool_call_", "response.mcp_")
    )


async def process_stream(
    result: RunResultStreaming,
    stream_key: str,
    *,
    include: Collection[StreamEventKind] = ("text", "reasoning", "tool_calls"),
) -> AsyncIterator[StreamEvent]:
    """Yield SDK stream events while persisting selected raw response events."""
    try:
        async for event in result.stream_events():
            if _should_persist(event, include):
                await DBOS.write_stream_async(stream_key, event)
            yield event
    except Exception as error:
        # Agents SDK tool errors can retain non-pickleable resources (such as
        # a sandbox client's lock) through exception chaining. DBOS persists a
        # workflow error, so pass it a plain, serializable exception instead.
        raise RuntimeError(f"Agents SDK stream failed: {error}") from None
    finally:
        await DBOS.close_stream_async(stream_key)


__all__ = ["StreamEventKind", "process_stream"]
