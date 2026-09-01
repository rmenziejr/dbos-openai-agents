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


def _write_stream_sync(key: str, value: AuditEvent) -> None:
    """Persist a sync-step audit event even when its workflow is async.

    DBOS's public sync writer rejects an inherited async workflow context, so
    this narrow bridge is loaded only when a synchronous computer action runs.
    """
    try:
        from dbos._context import snapshot_step_context
        from dbos._core import write_stream as write_stream_in_step
        from dbos._dbos import _get_dbos_instance
    except ImportError as exc:
        raise RuntimeError(
            "Synchronous computer-action auditing requires a compatible dbos>=2.10.0"
        ) from exc
    write_stream_in_step(
        _get_dbos_instance(), snapshot_step_context(reserve_sleep_id=False), key, value
    )


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
        _write_stream_sync(
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

    _write_stream_sync(
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
