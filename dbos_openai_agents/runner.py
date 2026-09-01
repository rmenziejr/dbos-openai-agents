import dataclasses
from asyncio import Event
from typing import Any, AsyncIterator, Awaitable, Callable, List

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
from agents.sandbox import SandboxAgent
from agents.sandbox.capabilities import Capability
from agents.models.multi_provider import MultiProvider
from agents.result import RunResultStreaming
from agents.tool import FunctionTool, Tool
from agents.tool_context import ToolContext
from dbos import DBOS

from .capabilities import DBOSCapability

# ---------------------------------------------------------------------------
# Turnstile: ordered execution of concurrent async operations
# ---------------------------------------------------------------------------


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
    __slots__ = ("turnstile", "durable_custom_tool_names")

    def __init__(self) -> None:
        self.turnstile = Turnstile([])
        self.durable_custom_tool_names: set[str] = set()


# ---------------------------------------------------------------------------
# Model wrapping
# ---------------------------------------------------------------------------


@DBOS.step()
async def _model_call_step(
    call_fn: Callable[[], Awaitable[ModelResponse]],
) -> ModelResponse:
    """Execute an LLM call as a durable DBOS step with retries."""
    return await call_fn()


@DBOS.step()
async def _model_stream_step(
    call_fn: Callable[[], AsyncIterator[TResponseStreamEvent]],
) -> list[TResponseStreamEvent]:
    """Collect a streamed LLM response in a durable DBOS step."""
    return [event async for event in call_fn()]


def _get_function_call_ids(
    output: List[TResponseOutputItem],
    durable_custom_tool_names: set[str],
) -> List[str]:
    """Extract tool call IDs whose invocation wrappers release the turnstile."""
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
        ids = _get_function_call_ids(
            result.output, self._state.durable_custom_tool_names
        )
        self._state.turnstile = Turnstile(ids)

        return result

    def stream_response(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[TResponseStreamEvent]:
        def call_llm() -> AsyncIterator[TResponseStreamEvent]:
            return self.model.stream_response(*args, **kwargs)

        async def stream() -> AsyncIterator[TResponseStreamEvent]:
            events = await _model_stream_step(call_llm)
            for event in events:
                if event.type == "response.completed":
                    self._state.turnstile = Turnstile(
                        _get_function_call_ids(
                            event.response.output,
                            self._state.durable_custom_tool_names,
                        )
                    )
                yield event

        return stream()


# ---------------------------------------------------------------------------
# Tool wrapping
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# DBOSRunner
# ---------------------------------------------------------------------------


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
        **kwargs: Any,
    ) -> RunResultStreaming:
        """Run an OpenAI agent with durable streamed model responses.

        Must be called from within a ``@DBOS.workflow()`` for durable execution.
        Model events are persisted after each completed model response, then yielded
        in their original order.
        """

        state = _State()

        run_config = kwargs.pop("run_config", RunConfig())
        run_config = dataclasses.replace(
            run_config,
            model_provider=DBOSModelProvider(state),
        )

        agent = _wrap_agent(starting_agent, state)

        return Runner.run_streamed(
            starting_agent=agent,
            input=input,
            run_config=run_config,
            **kwargs,
        )

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
