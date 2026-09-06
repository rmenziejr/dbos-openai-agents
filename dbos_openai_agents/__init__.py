from typing import Any, AsyncIterator

from dbos import DBOS

from .capabilities import DBOSCapability
from .agent_tool import DBOSAgentTool
from .computer import DBOSComputerTool
from .runner import DBOSRunner as _BaseDBOSRunner
from .streaming import StreamEventKind, process_stream


class DBOSRunner(_BaseDBOSRunner):
    """DBOS runner with support for attaching to an existing durable stream."""

    @classmethod
    async def attach_stream(
        cls,
        workflow_id: str,
        stream_key: str,
        *,
        offset: int = 0,
    ) -> AsyncIterator[tuple[int, Any]]:
        """Replay and follow a durable DBOS stream from a consumer cursor.

        ``offset`` is the number of values already consumed. Each yielded tuple
        contains the next resumable offset and the corresponding stream value.
        Attaching only observes an existing stream; it never starts or recovers
        the workflow that owns it.
        """
        if offset < 0:
            raise ValueError("offset must be non-negative")

        next_offset = offset
        async for event in DBOS.read_stream_async(
            workflow_id,
            stream_key,
            offset=offset,
        ):
            next_offset += 1
            yield next_offset, event


__all__ = [
    "DBOSCapability",
    "DBOSAgentTool",
    "DBOSComputerTool",
    "DBOSRunner",
    "StreamEventKind",
    "process_stream",
]
