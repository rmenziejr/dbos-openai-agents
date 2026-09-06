import pytest
from agents.stream_events import RawResponsesStreamEvent
from dbos import DBOS
from openai.types.responses import (
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseReasoningSummaryTextDeltaEvent,
)

from dbos_openai_agents.runner import DBOSRunner, _compact_replay_events


def _reasoning(delta: str, *, item_id: str, sequence_number: int) -> RawResponsesStreamEvent:
    return RawResponsesStreamEvent(
        data=ResponseReasoningSummaryTextDeltaEvent(
            type="response.reasoning_summary_text.delta",
            delta=delta,
            item_id=item_id,
            output_index=0,
            sequence_number=sequence_number,
            summary_index=0,
        )
    )


def _tool(delta: str, *, item_id: str, sequence_number: int) -> RawResponsesStreamEvent:
    return RawResponsesStreamEvent(
        data=ResponseFunctionCallArgumentsDeltaEvent(
            type="response.function_call_arguments.delta",
            delta=delta,
            item_id=item_id,
            output_index=0,
            sequence_number=sequence_number,
        )
    )


def test_compact_replay_events_preserves_logical_boundaries_and_cursors() -> None:
    historical = [
        (1, _reasoning("Need ", item_id="reasoning-1", sequence_number=1)),
        (2, _reasoning("tool", item_id="reasoning-1", sequence_number=2)),
        (3, _tool('{"city":', item_id="tool-1", sequence_number=3)),
        (4, _tool('"NYC"}', item_id="tool-1", sequence_number=4)),
        (5, _reasoning("Result ", item_id="reasoning-2", sequence_number=5)),
        (6, _reasoning("received", item_id="reasoning-2", sequence_number=6)),
    ]

    compacted = _compact_replay_events(historical)

    assert [cursor for cursor, _ in compacted] == [2, 4, 6]
    assert [event.data.delta for _, event in compacted] == [
        "Need tool",
        '{"city":"NYC"}',
        "Result received",
    ]
    assert [event.data.type for _, event in compacted] == [
        "response.reasoning_summary_text.delta",
        "response.function_call_arguments.delta",
        "response.reasoning_summary_text.delta",
    ]


@pytest.mark.asyncio
async def test_attach_stream_raw_skips_directly_to_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    async def read_stream_async(
        workflow_id: str,
        stream_key: str,
        *,
        offset: int = 0,
        polling_interval_sec: float | None = None,
        timeout_seconds: float | None = None,
    ):
        calls.append(offset)
        yield "live-1"
        yield "live-2"

    monkeypatch.setattr(DBOS, "read_stream_async", read_stream_async)

    attached = [
        item
        async for item in DBOSRunner.attach_stream(
            "workflow", "events", offset=5, replay="raw"
        )
    ]

    assert calls == [5]
    assert attached == [(6, "live-1"), (7, "live-2")]


@pytest.mark.asyncio
async def test_attach_stream_compacts_zero_to_offset_then_resumes_raw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    history = [
        _reasoning("Need ", item_id="reasoning-1", sequence_number=1),
        _reasoning("tool", item_id="reasoning-1", sequence_number=2),
        _tool('{"city":', item_id="tool-1", sequence_number=3),
        _tool('"NYC"}', item_id="tool-1", sequence_number=4),
        _reasoning("Current ", item_id="reasoning-2", sequence_number=5),
        _reasoning("thought", item_id="reasoning-2", sequence_number=6),
    ]
    live = _reasoning(" continues", item_id="reasoning-2", sequence_number=7)

    async def read_stream_async(
        workflow_id: str,
        stream_key: str,
        *,
        offset: int = 0,
        polling_interval_sec: float | None = None,
        timeout_seconds: float | None = None,
    ):
        calls.append(offset)
        if offset == 0:
            for event in history:
                yield event
            return
        assert offset == 6
        yield live

    monkeypatch.setattr(DBOS, "read_stream_async", read_stream_async)

    attached = [
        item
        async for item in DBOSRunner.attach_stream(
            "workflow", "events", offset=6, replay="compact"
        )
    ]

    assert calls == [0, 6]
    assert [cursor for cursor, _ in attached] == [2, 4, 6, 7]
    assert [event.data.delta for _, event in attached] == [
        "Need tool",
        '{"city":"NYC"}',
        "Current thought",
        " continues",
    ]


@pytest.mark.asyncio
async def test_attach_stream_compact_with_zero_offset_streams_raw_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    async def read_stream_async(
        workflow_id: str,
        stream_key: str,
        *,
        offset: int = 0,
        polling_interval_sec: float | None = None,
        timeout_seconds: float | None = None,
    ):
        calls.append(offset)
        yield "live"

    monkeypatch.setattr(DBOS, "read_stream_async", read_stream_async)

    attached = [
        item
        async for item in DBOSRunner.attach_stream(
            "workflow", "events", offset=0, replay="compact"
        )
    ]

    assert calls == [0]
    assert attached == [(1, "live")]
