import pytest
from dbos import DBOS, error as dboserror

from dbos_openai_agents.runner import DBOSRunner


@pytest.mark.asyncio
async def test_compact_tail_probes_forward_and_replays_to_last_confirmed_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probed: list[tuple[int, float | None]] = []
    read_offsets: list[int] = []

    async def read_stream_offset_async(
        workflow_id: str,
        stream_key: str,
        offset: int,
        *,
        timeout_seconds: float | None = None,
    ):
        probed.append((offset, timeout_seconds))
        if offset in (15, 20):
            return f"event-{offset}"
        raise dboserror.DBOSStreamTimeoutError(
            workflow_id, stream_key, timeout_seconds
        )

    async def read_stream_async(
        workflow_id: str,
        stream_key: str,
        *,
        offset: int = 0,
        polling_interval_sec: float | None = None,
        timeout_seconds: float | None = None,
    ):
        read_offsets.append(offset)
        if offset == 0:
            for i in range(21):
                yield f"history-{i}"
            return
        assert offset == 21
        yield "catch-up"

    monkeypatch.setattr(DBOS, "read_stream_offset_async", read_stream_offset_async)
    monkeypatch.setattr(DBOS, "read_stream_async", read_stream_async)

    attached = [
        item
        async for item in DBOSRunner.attach_stream(
            "workflow",
            "events",
            offset=10,
            replay="compact_tail",
            tail_probe_stride=5,
            tail_probe_attempts=3,
            tail_probe_timeout_seconds=0.01,
        )
    ]

    assert probed == [(15, 0.01), (20, 0.01), (25, 0.01)]
    assert read_offsets == [0, 21]
    assert attached[-1] == (22, "catch-up")
    assert attached[-2] == (21, "history-20")


@pytest.mark.asyncio
async def test_compact_tail_stops_after_first_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probed: list[int] = []
    read_offsets: list[int] = []

    async def read_stream_offset_async(
        workflow_id: str,
        stream_key: str,
        offset: int,
        *,
        timeout_seconds: float | None = None,
    ):
        probed.append(offset)
        raise dboserror.DBOSStreamTimeoutError(
            workflow_id, stream_key, timeout_seconds
        )

    async def read_stream_async(
        workflow_id: str,
        stream_key: str,
        *,
        offset: int = 0,
        polling_interval_sec: float | None = None,
        timeout_seconds: float | None = None,
    ):
        read_offsets.append(offset)
        if offset == 0:
            for i in range(10):
                yield f"history-{i}"
            return
        assert offset == 10
        yield "live"

    monkeypatch.setattr(DBOS, "read_stream_offset_async", read_stream_offset_async)
    monkeypatch.setattr(DBOS, "read_stream_async", read_stream_async)

    attached = [
        item
        async for item in DBOSRunner.attach_stream(
            "workflow",
            "events",
            offset=10,
            replay="compact_tail",
            tail_probe_stride=5,
            tail_probe_attempts=4,
            tail_probe_timeout_seconds=0.005,
        )
    ]

    assert probed == [15]
    assert read_offsets == [0, 10]
    assert attached[-1] == (11, "live")


@pytest.mark.asyncio
async def test_compact_tail_validates_probe_options() -> None:
    with pytest.raises(ValueError, match="tail_probe_stride"):
        async for _ in DBOSRunner.attach_stream(
            "workflow",
            "events",
            replay="compact_tail",
            tail_probe_stride=0,
        ):
            pass

    with pytest.raises(ValueError, match="tail_probe_attempts"):
        async for _ in DBOSRunner.attach_stream(
            "workflow",
            "events",
            replay="compact_tail",
            tail_probe_attempts=0,
        ):
            pass

    with pytest.raises(ValueError, match="tail_probe_timeout_seconds"):
        async for _ in DBOSRunner.attach_stream(
            "workflow",
            "events",
            replay="compact_tail",
            tail_probe_timeout_seconds=0,
        ):
            pass
