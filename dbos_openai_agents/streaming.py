from collections.abc import AsyncIterator, Collection
from typing import Literal

from agents.result import RunResultStreaming
from agents.stream_events import StreamEvent

StreamEventKind = Literal["text", "reasoning", "tool_calls"]


async def process_stream(
    result: RunResultStreaming,
    stream_key: str,
    *,
    include: Collection[StreamEventKind] = ("text", "reasoning", "tool_calls"),
) -> AsyncIterator[StreamEvent]:
    """Yield SDK stream events without writing or closing a DBOS stream.

    This compatibility helper only forwards the Agents SDK stream. Use
    ``DBOSRunner.run_streamed(..., stream_key=...)`` for durable transport.
    """
    # Keep the legacy parameters for callers while the runner owns transport.
    _ = stream_key, include
    try:
        async for event in result.stream_events():
            yield event
    except Exception as error:
        # Agents SDK tool errors can retain non-pickleable resources (such as
        # a sandbox client's lock) through exception chaining. DBOS persists a
        # workflow error, so pass it a plain, serializable exception instead.
        raise RuntimeError(f"Agents SDK stream failed: {error}") from None


__all__ = ["StreamEventKind", "process_stream"]
