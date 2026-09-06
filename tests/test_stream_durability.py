import pytest
from dbos import DBOS
from dbos._error import DBOSStreamTimeoutError

from dbos_openai_agents.runner import DBOSRunner, _consume_model_stream


@pytest.mark.asyncio
async def test_consume_model_stream_skips_replayed_prefix_and_resumes_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe_calls: list[int] = []
    write_calls: list[object] = []
    existing = {3: "old-a", 4: "old-b"}

    async def read_stream_offset_async(
        workflow_id: str,
        stream_key: str,
        offset: int,
        *,
        timeout_seconds: float | None = None,
        polling_interval_sec: float | None = None,
    ) -> object:
        assert workflow_id == "workflow"
        assert stream_key == "events"
        assert timeout_seconds == 0.001
        probe_calls.append(offset)
        if offset in existing:
            return existing[offset]
        raise DBOSStreamTimeoutError(workflow_id, stream_key)

    async def write_stream_async(stream_key: str, value: object) -> None:
        assert stream_key == "events"
        write_calls.append(value)

    async def provider():
        for value in ["a", "b", "c", "d"]:
            yield value

    monkeypatch.setattr(DBOS, "read_stream_offset_async", read_stream_offset_async)
    monkeypatch.setattr(DBOS, "write_stream_async", write_stream_async)

    events = await _consume_model_stream(
        provider,
        stream_key="events",
        stream_offset=3,
        workflow_id="workflow",
    )

    assert events == ["a", "b", "c", "d"]
    assert probe_calls == [3, 4, 5]
    assert [event.data for event in write_calls] == ["c", "d"]


@pytest.mark.asyncio
async def test_consume_model_stream_probes_once_on_fresh_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe_calls: list[int] = []
    write_calls: list[object] = []

    async def read_stream_offset_async(
        workflow_id: str,
        stream_key: str,
        offset: int,
        *,
        timeout_seconds: float | None = None,
        polling_interval_sec: float | None = None,
    ) -> object:
        probe_calls.append(offset)
        raise DBOSStreamTimeoutError(workflow_id, stream_key)

    async def write_stream_async(stream_key: str, value: object) -> None:
        write_calls.append(value)

    async def provider():
        for value in ["a", "b", "c"]:
            yield value

    monkeypatch.setattr(DBOS, "read_stream_offset_async", read_stream_offset_async)
    monkeypatch.setattr(DBOS, "write_stream_async", write_stream_async)

    await _consume_model_stream(
        provider,
        stream_key="events",
        stream_offset=0,
        workflow_id="workflow",
    )

    assert probe_calls == [0]
    assert [event.data for event in write_calls] == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_attach_stream_passes_timeout_and_polling_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, float | None, float | None]] = []

    async def read_stream_async(
        workflow_id: str,
        stream_key: str,
        *,
        offset: int = 0,
        polling_interval_sec: float | None = None,
        timeout_seconds: float | None = None,
    ):
        calls.append((offset, polling_interval_sec, timeout_seconds))
        yield "event"

    monkeypatch.setattr(DBOS, "read_stream_async", read_stream_async)

    attached = [
        item
        async for item in DBOSRunner.attach_stream(
            "workflow",
            "events",
            offset=5,
            replay="raw",
            polling_interval_sec=0.25,
            timeout_seconds=30,
        )
    ]

    assert calls == [(5, 0.25, 30)]
    assert attached == [(6, "event")]


@pytest.mark.asyncio
async def test_compact_attach_passes_options_to_history_and_live_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, float | None, float | None]] = []

    async def read_stream_async(
        workflow_id: str,
        stream_key: str,
        *,
        offset: int = 0,
        polling_interval_sec: float | None = None,
        timeout_seconds: float | None = None,
    ):
        calls.append((offset, polling_interval_sec, timeout_seconds))
        if offset == 0:
            yield "historical"
        else:
            yield "live"

    monkeypatch.setattr(DBOS, "read_stream_async", read_stream_async)

    attached = [
        item
        async for item in DBOSRunner.attach_stream(
            "workflow",
            "events",
            offset=1,
            replay="compact",
            polling_interval_sec=0.5,
            timeout_seconds=20,
        )
    ]

    assert calls == [(0, 0.5, 20), (1, 0.5, 20)]
    assert attached == [(1, "historical"), (2, "live")]
