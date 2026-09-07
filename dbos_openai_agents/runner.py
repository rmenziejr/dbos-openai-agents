import dataclasses
from asyncio import Event
from typing import Any, AsyncIterator, Awaitable, Callable, List, Literal

from agents import (
    Agent,
    Handoff,
    Model,
    RunConfig,
    Runner,
    RunResult,
    TContext,
)
from agents.items import ModelResponse, TResponseOutputItem, TResponseStreamEvent
from agents.stream_events import RawResponsesStreamEvent
from agents.models.multi_provider import MultiProvider
from agents.result import RunResultStreaming
from agents.sandbox import SandboxAgent
from agents.sandbox.capabilities import Capability
from agents.tool import CustomTool, FunctionTool, Tool
from agents.tool_context import ToolContext
from dbos import DBOS, error as dboserror

from .capabilities import DBOSCapability

# Turnstile: ordered execution of concurrent async operations


class Turnstile:
    """Serializes concurrent async operations in a fixed order by ID.

    When the OpenAI agents SDK launches multiple tool calls via asyncio.gather,
    DBOS needs them to start execution in a deterministic order so that
    function_id assignment is consistent on replay.
    """

    def __init__(self, ids: list[str]):
        self.turns = dict(zip(ids, ids[1:]))
        self.events = {id: Event() for id in ids}
        if ids:
            self.events[ids[0]].set()

    async def wait_for(self, id: str) -> None:
        await self.events[id].wait()

    def allow_next_after(self, id: str) -> None:
        next_id = self.turns.get(id)
        if next_id is not None:
            self.events[next_id].set()


class _State:
    __slots__ = ("turnstile", "durable_custom_tools", "stream_key", "stream_offset")

    def __init__(self, stream_key: str | None = None) -> None:
        self.turnstile = Turnstile([])
        self.durable_custom_tools: dict[int, CustomTool] = {}
        self.stream_key = stream_key
        self.stream_offset = 0


# Model wrapping


@DBOS.step()
async def _model_call_step(
    call_fn: Callable[[], Awaitable[ModelResponse]],
) -> ModelResponse:
    """Execute an LLM call as a durable DBOS step with retries."""
    return await call_fn()


async def _consume_model_stream(
    call_fn: Callable[[], AsyncIterator[TResponseStreamEvent]],
    *,
    stream_key: str | None,
    stream_offset: int,
    workflow_id: str | None,
) -> list[TResponseStreamEvent]:
    """Consume one provider stream while avoiding duplicate recovery-prefix writes.

    DBOS stream writes made from a step are at-least-once. On workflow recovery,
    an incomplete model step may therefore re-run after already writing a prefix
    of its provider events. Probe the expected stream offset until the first gap;
    already-present offsets are replayed without another write, and after the gap
    all subsequent provider events are written live without additional probes.
    """
    events: list[TResponseStreamEvent] = []
    probe_existing = stream_key is not None
    next_offset = stream_offset

    if stream_key is not None and workflow_id is None:
        raise RuntimeError("DBOS workflow ID is unavailable while streaming")

    async for event in call_fn():
        if stream_key is not None:
            should_write = True
            if probe_existing:
                try:
                    await DBOS.read_stream_offset_async(
                        workflow_id,
                        stream_key,
                        next_offset,
                        timeout_seconds=0.001,
                    )
                    should_write = False
                except dboserror.DBOSStreamTimeoutError:
                    probe_existing = False

            if should_write:
                await DBOS.write_stream_async(
                    stream_key, RawResponsesStreamEvent(data=event)
                )

        events.append(event)
        next_offset += 1

    return events


@DBOS.step()
async def _model_stream_step(
    call_fn: Callable[[], AsyncIterator[TResponseStreamEvent]],
    stream_key: str | None,
    stream_offset: int,
) -> list[TResponseStreamEvent]:
    """Write provider events live and retry one empty stream before failing."""
    for _ in range(2):
        try:
            events = await _consume_model_stream(
                call_fn,
                stream_key=stream_key,
                stream_offset=stream_offset,
                workflow_id=DBOS.workflow_id,
            )
        except Exception as error:
            # A DBOS step persists its own error, so discard provider exception
            # chains that can retain non-pickleable resources.
            raise RuntimeError(f"Agents SDK stream failed: {error}") from None
        if events:
            return events

    raise RuntimeError("Model stream ended without events after retry")


def _get_function_call_ids(
    output: List[TResponseOutputItem],
    tools: list[Tool],
    durable_custom_tools: dict[int, CustomTool],
) -> List[str]:
    """Extract tool call IDs whose invocation wrappers release the turnstile."""
    durable_custom_tool_names = {
        tool.name
        for tool in tools
        if isinstance(tool, CustomTool) and durable_custom_tools.get(id(tool)) is tool
    }
    call_ids: List[str] = []
    for item in output:
        if item.type == "function_call":
            call_id = getattr(item, "call_id", None)
        elif (
            item.type == "custom_tool_call"
            and getattr(item, "name", None) in durable_custom_tool_names
        ):
            call_id = getattr(item, "call_id", None)
        else:
            continue
        if isinstance(call_id, str):
            call_ids.append(call_id)
    return call_ids


def _get_model_tools(args: tuple[Any, ...], kwargs: dict[str, Any]) -> list[Tool]:
    """Return the exact tools supplied for this model request."""
    candidate = kwargs.get("tools")
    if candidate is None and len(args) > 3:
        candidate = args[3]
    return candidate if isinstance(candidate, list) else []


class DBOSModelProvider(MultiProvider):
    """Model provider that wraps every model in a DBOSModelWrapper."""

    def __init__(self, state: _State):
        super().__init__()
        self._state = state

    def get_model(self, model_name: str | None) -> Model:
        model = super().get_model(model_name or None)
        return DBOSModelWrapper(model, self._state)


class DBOSModelWrapper(Model):
    """Wraps a Model so each get_response() call is a durable DBOS step."""

    def __init__(self, model: Model, state: _State):
        self.model = model
        self.model_name = "DBOSModelWrapper"
        self._state = state

    async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:
        async def call_llm() -> ModelResponse:
            return await self.model.get_response(*args, **kwargs)

        result: ModelResponse = await _model_call_step(call_llm)

        # Prepare the turnstile for any tool calls in the response
        model_tools = _get_model_tools(args, kwargs)
        ids = _get_function_call_ids(
            result.output,
            model_tools,
            self._state.durable_custom_tools,
        )
        self._state.turnstile = Turnstile(ids)

        return result

    def stream_response(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[TResponseStreamEvent]:
        def call_llm() -> AsyncIterator[TResponseStreamEvent]:
            return self.model.stream_response(*args, **kwargs)

        model_tools = _get_model_tools(args, kwargs)

        async def stream() -> AsyncIterator[TResponseStreamEvent]:
            events = await _model_stream_step(
                call_llm,
                self._state.stream_key,
                self._state.stream_offset,
            )
            self._state.stream_offset += len(events)
            for event in events:
                if event.type == "response.completed":
                    self._state.turnstile = Turnstile(
                        _get_function_call_ids(
                            event.response.output,
                            model_tools,
                            self._state.durable_custom_tools,
                        )
                    )
                yield event

        return stream()


# Tool wrapping


def _create_tool_wrapper(
    state: _State, tool: FunctionTool
) -> Callable[[ToolContext[Any], str], Awaitable[Any]]:
    """Create a turnstile-gated on_invoke_tool wrapper."""

    async def on_invoke_tool_wrapper(
        tool_context: ToolContext[Any], tool_input: str
    ) -> Any:
        turnstile = state.turnstile
        call_id = tool_context.tool_call_id

        await turnstile.wait_for(call_id)
        turnstile.allow_next_after(call_id)
        return await tool.on_invoke_tool(tool_context, tool_input)

    return on_invoke_tool_wrapper


def _wrap_agent(agent: Agent[TContext], state: _State) -> Agent[TContext]:
    """Return a clone of *agent* with model and tools wrapped for DBOS durability."""

    clone_kwargs: dict[str, Any] = {}

    if isinstance(agent, SandboxAgent):
        capabilities: list[Capability] = []
        for capability in agent.capabilities:
            if isinstance(capability, DBOSCapability):
                durable_capability = capability.clone()
                durable_capability.bind_durability_state(state)
                capabilities.append(durable_capability)
            else:
                capabilities.append(capability)
        clone_kwargs["capabilities"] = capabilities

    # Wrap the model if it's a Model instance (the SDK uses it directly,
    # bypassing the model_provider).
    if isinstance(agent.model, Model) and not isinstance(agent.model, DBOSModelWrapper):
        clone_kwargs["model"] = DBOSModelWrapper(agent.model, state)

    wrapped_tools: list[Tool] = []
    for tool in agent.tools:
        if isinstance(tool, FunctionTool):
            wrapper = _create_tool_wrapper(state, tool)
            wrapped_tools.append(dataclasses.replace(tool, on_invoke_tool=wrapper))
        else:
            # Other tools either execute entirely server-side (no local component)
            # or execute serially.
            wrapped_tools.append(tool)
    clone_kwargs["tools"] = wrapped_tools

    wrapped_handoffs: list[Agent[Any] | Handoff[Any]] = []
    for handoff in agent.handoffs:
        if isinstance(handoff, Agent):
            wrapped_handoffs.append(_wrap_agent(handoff, state))
        elif isinstance(handoff, Handoff):
            wrapped_handoffs.append(_wrap_handoff(handoff, state))
        else:
            raise TypeError(f"Unsupported handoff type: {type(handoff)}")
    clone_kwargs["handoffs"] = wrapped_handoffs

    return agent.clone(**clone_kwargs)


def _wrap_handoff(handoff: Handoff[TContext], state: _State) -> Handoff[TContext]:
    """Wrap a Handoff so the agent it produces also has wrapped tools."""
    original = handoff.on_invoke_handoff

    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        agent = await original(*args, **kwargs)
        return _wrap_agent(agent, state)

    return dataclasses.replace(handoff, on_invoke_handoff=wrapped)


ReplayMode = Literal["raw", "compact", "compact_tail"]
_DELTA_IDENTITY_FIELDS = (
    "type",
    "item_id",
    "output_index",
    "content_index",
    "summary_index",
    "call_id",
)


def _delta_group_key(event: Any) -> tuple[Any, ...] | None:
    """Return a stable identity for a compactable raw SDK delta event."""
    if not isinstance(event, RawResponsesStreamEvent):
        return None

    data = event.data
    if not isinstance(getattr(data, "delta", None), str):
        return None

    return tuple(getattr(data, field, None) for field in _DELTA_IDENTITY_FIELDS)


def _merge_delta_events(events: list[Any]) -> Any:
    """Merge one logical run of raw SDK delta events into the final event shape."""
    last = events[-1]
    data = last.data
    merged_delta = "".join(event.data.delta for event in events)
    merged_data = data.model_copy(update={"delta": merged_delta})
    return RawResponsesStreamEvent(data=merged_data)


def _compact_replay_events(
    events: list[tuple[int, Any]],
) -> list[tuple[int, Any]]:
    """Compact consecutive compatible deltas while preserving event order."""
    compacted: list[tuple[int, Any]] = []
    pending: list[tuple[int, Any]] = []
    pending_key: tuple[Any, ...] | None = None

    def flush() -> None:
        nonlocal pending, pending_key
        if pending:
            compacted.append(
                (pending[-1][0], _merge_delta_events([event for _, event in pending]))
            )
            pending = []
            pending_key = None

    for cursor, event in events:
        key = _delta_group_key(event)
        if key is None:
            flush()
            compacted.append((cursor, event))
        elif pending and key != pending_key:
            flush()
            pending = [(cursor, event)]
            pending_key = key
        else:
            if not pending:
                pending_key = key
            pending.append((cursor, event))

    flush()
    return compacted


async def _discover_compact_tail(
    workflow_id: str,
    stream_key: str,
    *,
    offset: int,
    stride: int,
    attempts: int,
    timeout_seconds: float,
) -> int:
    """Return a bounded near-tail cursor using sparse single-offset probes."""
    boundary = offset
    for attempt in range(1, attempts + 1):
        probe_offset = offset + stride * attempt
        try:
            await DBOS.read_stream_offset_async(
                workflow_id,
                stream_key,
                probe_offset,
                timeout_seconds=timeout_seconds,
            )
        except dboserror.DBOSStreamTimeoutError:
            break
        boundary = probe_offset + 1
    return boundary


# DBOSRunner


class DBOSRunner:
    # This is not a workflow because the Agent type is not pickle-able.
    @classmethod
    async def run(
        cls,
        starting_agent: Agent[TContext],
        input: str | list[Any],
        **kwargs: Any,
    ) -> RunResult:
        """Run an OpenAI agent with DBOS durability.

        Must be called from within a ``@DBOS.workflow()`` for durable execution.

        Example::

            @DBOS.workflow()
            async def run_agent(user_input: str):
                result = await DBOSRunner.run(agent, user_input)
                return result.final_output
        """

        state = _State()

        run_config = kwargs.pop("run_config", RunConfig())
        run_config = dataclasses.replace(
            run_config,
            model_provider=DBOSModelProvider(state),
        )

        agent = _wrap_agent(starting_agent, state)

        return await Runner.run(
            starting_agent=agent,
            input=input,
            run_config=run_config,
            **kwargs,
        )

    @classmethod
    def run_streamed(
        cls,
        starting_agent: Agent[TContext],
        input: str | list[Any],
        stream_key: str | None = None,
        **kwargs: Any,
    ) -> RunResultStreaming:
        """Run an OpenAI agent with durable streamed model responses.

        Must be called from within a ``@DBOS.workflow()`` for durable execution.
        Raw provider events are written live as received when ``stream_key`` is set.
        The completed event list is retained in the durable model step for SDK replay;
        SDK events are yielded in their original order.
        """

        state = _State(stream_key)

        run_config = kwargs.pop("run_config", RunConfig())
        run_config = dataclasses.replace(
            run_config,
            model_provider=DBOSModelProvider(state),
        )

        agent = _wrap_agent(starting_agent, state)

        result = Runner.run_streamed(
            starting_agent=agent,
            input=input,
            run_config=run_config,
            **kwargs,
        )

        original_stream_events = result.stream_events
        stream_closed = False

        async def stream_events() -> AsyncIterator[Any]:
            nonlocal stream_closed
            try:
                async for event in original_stream_events():
                    yield event
            except Exception as error:
                # Agents SDK tool errors can retain non-pickleable resources
                # through exception chaining. DBOS persists workflow errors, so
                # provide a plain, serializable exception instead.
                if isinstance(error, RuntimeError) and str(error).startswith(
                    "Agents SDK stream failed: "
                ):
                    raise error from None
                raise RuntimeError(f"Agents SDK stream failed: {error}") from None
            finally:
                if stream_key is not None and not stream_closed:
                    stream_closed = True
                    await DBOS.close_stream_async(stream_key)

        setattr(result, "stream_events", stream_events)
        return result

    @classmethod
    async def attach_stream(
        cls,
        workflow_id: str,
        stream_key: str,
        *,
        offset: int = 0,
        replay: ReplayMode = "raw",
        polling_interval_sec: float | None = None,
        timeout_seconds: float | None = None,
        tail_probe_stride: int = 100,
        tail_probe_attempts: int = 5,
        tail_probe_timeout_seconds: float = 0.01,
    ) -> AsyncIterator[tuple[int, Any]]:
        """Replay and follow a durable DBOS stream from a consumer cursor.

        ``offset`` is the number of values already consumed. In ``raw`` mode,
        those values are skipped and reading starts directly at ``offset``. In
        ``compact`` mode, values before ``offset`` are replayed as compacted
        logical delta blocks before normal one-event streaming resumes there.
        ``compact_tail`` sparsely probes beyond ``offset`` to find a bounded
        near-tail cursor, compacts through the last confirmed persisted value,
        then resumes normal streaming from that discovered cursor.
        """
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if replay not in ("raw", "compact", "compact_tail"):
            raise ValueError("replay must be 'raw', 'compact', or 'compact_tail'")
        if replay == "compact_tail":
            if tail_probe_stride <= 0:
                raise ValueError("tail_probe_stride must be positive")
            if tail_probe_attempts <= 0:
                raise ValueError("tail_probe_attempts must be positive")
            if tail_probe_timeout_seconds <= 0:
                raise ValueError("tail_probe_timeout_seconds must be positive")

        replay_boundary = offset
        if replay == "compact_tail":
            replay_boundary = await _discover_compact_tail(
                workflow_id,
                stream_key,
                offset=offset,
                stride=tail_probe_stride,
                attempts=tail_probe_attempts,
                timeout_seconds=tail_probe_timeout_seconds,
            )

        if replay in ("compact", "compact_tail") and replay_boundary > 0:
            historical: list[tuple[int, Any]] = []
            historical_stream = DBOS.read_stream_async(
                workflow_id,
                stream_key,
                offset=0,
                polling_interval_sec=polling_interval_sec,
                timeout_seconds=timeout_seconds,
            )
            try:
                cursor = 0
                async for event in historical_stream:
                    cursor += 1
                    historical.append((cursor, event))
                    if cursor >= replay_boundary:
                        break
            finally:
                await historical_stream.aclose()

            for item in _compact_replay_events(historical):
                yield item

        next_offset = replay_boundary
        async for event in DBOS.read_stream_async(
            workflow_id,
            stream_key,
            offset=replay_boundary,
            polling_interval_sec=polling_interval_sec,
            timeout_seconds=timeout_seconds,
        ):
            next_offset += 1
            yield next_offset, event

    @classmethod
    def run_sync(
        cls,
        starting_agent: Agent[TContext],
        input: str | list[Any],
        **kwargs: Any,
    ) -> RunResult:
        """Run an OpenAI agent synchronously with DBOS durability.

        Must be called from within a ``@DBOS.workflow()`` for durable execution.

        Example::

            @DBOS.workflow()
            def run_agent(user_input: str):
                result = DBOSRunner.run_sync(agent, user_input)
                return result.final_output
        """

        state = _State()

        run_config = kwargs.pop("run_config", RunConfig())
        run_config = dataclasses.replace(
            run_config,
            model_provider=DBOSModelProvider(state),
        )

        agent = _wrap_agent(starting_agent, state)

        return Runner.run_sync(
            starting_agent=agent,
            input=input,
            run_config=run_config,
            **kwargs,
        )
