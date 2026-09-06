import asyncio

import pytest
from dbos import DBOS

from dbos_openai_agents import DBOSRunner


@pytest.mark.asyncio
async def test_attach_stream_reads_closed_stream_from_offset(dbos_env: None) -> None:
    """Attach skips consumed values and yields the next cursor with each event."""

    @DBOS.workflow()
    async def producer() -> None:
        await DBOS.write_stream_async("events", "A")
        await DBOS.write_stream_async("events", "B")
        await DBOS.write_stream_async("events", "C")
        await DBOS.close_stream_async("events")

    handle = await DBOS.start_workflow_async(producer)
    await handle.get_result()

    attached = [
        item
        async for item in DBOSRunner.attach_stream(
            handle.get_workflow_id(), "events", offset=1
        )
    ]

    assert attached == [(2, "B"), (3, "C")]


@pytest.mark.asyncio
async def test_attach_stream_follows_running_stream(dbos_env: None) -> None:
    """Attach replays persisted values, then follows new values until close."""
    first_written = asyncio.Event()
    allow_second = asyncio.Event()

    @DBOS.workflow()
    async def producer() -> None:
        await DBOS.write_stream_async("events", "A")
        first_written.set()
        await allow_second.wait()
        await DBOS.write_stream_async("events", "B")
        await DBOS.close_stream_async("events")

    handle = await DBOS.start_workflow_async(producer)
    await first_written.wait()

    async def consume() -> list[tuple[int, str]]:
        return [
            item
            async for item in DBOSRunner.attach_stream(
                handle.get_workflow_id(), "events", offset=0
            )
        ]

    consumer = asyncio.create_task(consume())
    await asyncio.sleep(0)
    assert not consumer.done()

    allow_second.set()
    assert await asyncio.wait_for(consumer, timeout=5) == [(1, "A"), (2, "B")]
    await handle.get_result()
