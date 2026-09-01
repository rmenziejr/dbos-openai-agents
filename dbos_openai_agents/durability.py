"""Durable execution helpers for native capability actions."""

from typing import Awaitable, Callable, Protocol, TypedDict, TypeVar

from dbos import DBOS

DEFAULT_AUDIT_STREAM_KEY = "dbos-capability-events"

T = TypeVar("T")


class AuditEvent(TypedDict):
    """The approved, payload-free record for a capability action."""

    source: str
    owner: str
    action: str
    call_id: str | None
    status: str


class _AsyncTurnstile(Protocol):
    def wait_for(self, call_id: str) -> Awaitable[None]: ...

    def allow_next_after(self, call_id: str) -> None: ...


class _AsyncActionState(Protocol):
    turnstile: _AsyncTurnstile


class _SyncTurnstile(Protocol):
    def wait_for(self, call_id: str) -> None: ...

    def allow_next_after(self, call_id: str) -> None: ...


class _SyncActionState(Protocol):
    turnstile: _SyncTurnstile


def _audit_event(
    *, source: str, owner: str, action: str, call_id: str | None, status: str
) -> AuditEvent:
    return {
        "source": source,
        "owner": owner,
        "action": action,
        "call_id": call_id,
        "status": status,
    }


@DBOS.step()
async def _native_action_step(
    invoke: Callable[[], Awaitable[T]],
    source: str,
    owner: str,
    action: str,
    call_id: str | None,
    audit_stream_key: str,
) -> T:
    """Execute one async native action and persist its audit outcome atomically."""
    try:
        result = await invoke()
    except Exception:
        await DBOS.write_stream_async(
            audit_stream_key,
            _audit_event(
                source=source,
                owner=owner,
                action=action,
                call_id=call_id,
                status="failed",
            ),
        )
        raise

    await DBOS.write_stream_async(
        audit_stream_key,
        _audit_event(
            source=source,
            owner=owner,
            action=action,
            call_id=call_id,
            status="completed",
        ),
    )
    return result


@DBOS.step()
def _native_action_step_sync(
    invoke: Callable[[], T],
    source: str,
    owner: str,
    action: str,
    call_id: str | None,
    audit_stream_key: str,
) -> T:
    """Execute one sync native action and persist its audit outcome atomically."""
    try:
        result = invoke()
    except Exception:
        DBOS.write_stream(
            audit_stream_key,
            _audit_event(
                source=source,
                owner=owner,
                action=action,
                call_id=call_id,
                status="failed",
            ),
        )
        raise

    DBOS.write_stream(
        audit_stream_key,
        _audit_event(
            source=source,
            owner=owner,
            action=action,
            call_id=call_id,
            status="completed",
        ),
    )
    return result


async def run_durable_action(
    *,
    state: _AsyncActionState,
    source: str,
    owner: str,
    action: str,
    call_id: str | None,
    audit_stream_key: str = DEFAULT_AUDIT_STREAM_KEY,
    invoke: Callable[[], Awaitable[T]],
) -> T:
    """Run an ordered async native action in one durable DBOS step."""
    if call_id is not None:
        await state.turnstile.wait_for(call_id)
        state.turnstile.allow_next_after(call_id)
    return await _native_action_step(
        invoke,
        source,
        owner,
        action,
        call_id,
        audit_stream_key,
    )


def run_durable_action_sync(
    *,
    state: _SyncActionState,
    source: str,
    owner: str,
    action: str,
    call_id: str | None,
    audit_stream_key: str = DEFAULT_AUDIT_STREAM_KEY,
    invoke: Callable[[], T],
) -> T:
    """Run an ordered sync native action in one durable DBOS step."""
    if call_id is not None:
        state.turnstile.wait_for(call_id)
        state.turnstile.allow_next_after(call_id)
    return _native_action_step_sync(
        invoke,
        source,
        owner,
        action,
        call_id,
        audit_stream_key,
    )
