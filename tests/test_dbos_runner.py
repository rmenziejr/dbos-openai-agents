import asyncio
from contextlib import suppress
import pickle
import threading
from typing import Any, AsyncGenerator, AsyncIterator, cast

import pytest
from agents import (
    Agent,
    GuardrailFunctionOutput,
    RunContextWrapper,
    ToolGuardrailFunctionOutput,
    Usage,
    handoff,
    output_guardrail,
    tool_input_guardrail,
    tool_output_guardrail,
)
from agents.items import ModelResponse, TResponseStreamEvent
from agents.stream_events import RawResponsesStreamEvent
from agents.tool import CustomTool, function_tool
from agents.tool_context import ToolContext
from agents.tool_guardrails import ToolInputGuardrailData, ToolOutputGuardrailData
from dbos import DBOS, SetWorkflowID
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseCustomToolCall,
    ResponseFunctionToolCall,
    ResponseTextDeltaEvent,
)
from utils import FakeModel, make_message_response, make_tool_call_response

from dbos_openai_agents import DBOSAgentTool, DBOSRunner, process_stream


@pytest.mark.asyncio
async def test_simple_message(dbos_env: None) -> None:
    """DBOSRunner returns a simple text response."""
    model = FakeModel([make_message_response("Hello!")])
    agent = Agent(name="test", model=model)

    @DBOS.workflow()
    async def wf(user_input: str) -> str:
        result = await DBOSRunner.run(agent, user_input)
        return str(result.final_output)

    output = await wf("Hi")
    assert output == "Hello!"

    # 1 workflow, with 1 model call step
    workflows = await DBOS.list_workflows_async()
    assert len(workflows) == 1
    steps = await DBOS.list_workflow_steps_async(workflows[0].workflow_id)
    assert len(steps) == 1
    assert steps[0]["function_name"] == "_model_call_step"


@pytest.mark.asyncio
async def test_tool_call(dbos_env: None) -> None:
    """DBOSRunner executes a tool call and returns the final message."""
    tool_calls_made: list[str] = []

    @function_tool
    @DBOS.step()
    async def get_weather(city: str) -> str:
        """Get the weather for a city."""
        tool_calls_made.append(city)
        return f"Sunny in {city}"

    model = FakeModel(
        [
            make_tool_call_response("call_1", "get_weather", '{"city": "NYC"}'),
            make_message_response("The weather in NYC is sunny."),
        ]
    )
    agent = Agent(name="test", model=model, tools=[get_weather])

    @DBOS.workflow()
    async def wf(user_input: str) -> str:
        result = await DBOSRunner.run(agent, user_input)
        return str(result.final_output)

    output = await wf("What's the weather in NYC?")
    assert output == "The weather in NYC is sunny."
    assert tool_calls_made == ["NYC"]

    # 1 workflow, with 3 steps: model call, tool call, model call
    workflows = await DBOS.list_workflows_async()
    assert len(workflows) == 1
    steps = await DBOS.list_workflow_steps_async(workflows[0].workflow_id)
    assert len(steps) == 3
    assert steps[0]["function_name"] == "_model_call_step"
    assert "get_weather" in steps[1]["function_name"]
    assert steps[2]["function_name"] == "_model_call_step"


@pytest.mark.asyncio
async def test_multiple_tool_calls(dbos_env: None) -> None:
    """DBOSRunner handles parallel tool calls that start in deterministic order."""
    num_calls = 100
    cities = [f"city_{i}" for i in range(num_calls)]
    concurrent = 0
    max_concurrent = 0

    @function_tool
    @DBOS.step()
    async def get_weather(city: str) -> str:
        """Get the weather for a city."""
        nonlocal concurrent, max_concurrent
        concurrent += 1
        max_concurrent = max(max_concurrent, concurrent)
        await asyncio.sleep(1)
        concurrent -= 1
        return f"Sunny in {city}"

    model = FakeModel(
        [
            ModelResponse(
                output=[
                    ResponseFunctionToolCall(
                        type="function_call",
                        call_id=f"call_{i}",
                        name="get_weather",
                        arguments=f'{{"city": "{city}"}}',
                    )
                    for i, city in enumerate(cities)
                ],
                usage=Usage(),
                response_id="resp_1",
            ),
            make_message_response("Done."),
        ]
    )
    agent = Agent(name="test", model=model, tools=[get_weather])

    @DBOS.workflow()
    async def wf(user_input: str) -> str:
        result = await DBOSRunner.run(agent, user_input)
        return str(result.final_output)

    output = await wf("Weather everywhere?")
    assert output == "Done."
    # Tools actually run concurrently (not sequentially)
    assert (
        max_concurrent > 1
    ), f"Expected concurrent execution, but max_concurrent={max_concurrent}"

    # 1 workflow, with 102 steps: 1 model call + 100 tool calls + 1 model call
    workflows = await DBOS.list_workflows_async()
    assert len(workflows) == 1
    steps = await DBOS.list_workflow_steps_async(workflows[0].workflow_id)
    assert len(steps) == num_calls + 2
    assert steps[0]["function_name"] == "_model_call_step"
    # Steps are ordered by function_id — verify each tool step recorded
    # the correct city output in deterministic order
    for i in range(num_calls):
        assert "get_weather" in steps[i + 1]["function_name"]
        assert steps[i + 1]["output"] == f"Sunny in {cities[i]}"
    assert steps[num_calls + 1]["function_name"] == "_model_call_step"


@pytest.mark.asyncio
async def test_guardrails(dbos_env: None) -> None:
    """DBOSRunner works with DBOS step-annotated guardrails on tools and agent output."""

    @tool_input_guardrail
    @DBOS.step()
    async def validate_tool_input(
        data: ToolInputGuardrailData,
    ) -> ToolGuardrailFunctionOutput:
        """Check tool input is acceptable."""
        return ToolGuardrailFunctionOutput.allow(output_info="input_ok")

    @tool_output_guardrail
    @DBOS.step()
    async def validate_tool_output(
        data: ToolOutputGuardrailData,
    ) -> ToolGuardrailFunctionOutput:
        """Check tool output is acceptable."""
        return ToolGuardrailFunctionOutput.allow(output_info="output_ok")

    @function_tool(
        tool_input_guardrails=[validate_tool_input],
        tool_output_guardrails=[validate_tool_output],
    )
    @DBOS.step()
    async def get_weather(city: str) -> str:
        """Get the weather for a city."""
        return f"Sunny in {city}"

    @output_guardrail
    @DBOS.step()
    async def check_output(
        context: RunContextWrapper,
        agent: Agent,
        output: str,
    ) -> GuardrailFunctionOutput:
        """Verify the output is not empty."""
        return GuardrailFunctionOutput(
            output_info={"length": len(output)},
            tripwire_triggered=len(output) == 0,
        )

    model = FakeModel(
        [
            make_tool_call_response("call_1", "get_weather", '{"city": "NYC"}'),
            make_message_response("The weather in NYC is sunny."),
        ]
    )
    agent = Agent(
        name="test",
        model=model,
        tools=[get_weather],
        output_guardrails=[check_output],
    )

    @DBOS.workflow()
    async def wf(user_input: str) -> str:
        result = await DBOSRunner.run(agent, user_input)
        return str(result.final_output)

    output = await wf("What's the weather in NYC?")
    assert output == "The weather in NYC is sunny."

    # 1 workflow with 6 steps:
    #   model call, tool input guardrail, tool call, tool output guardrail,
    #   model call, output guardrail
    workflows = await DBOS.list_workflows_async()
    assert len(workflows) == 1
    steps = await DBOS.list_workflow_steps_async(workflows[0].workflow_id)
    assert len(steps) == 6
    assert steps[0]["function_name"] == "_model_call_step"
    assert "validate_tool_input" in steps[1]["function_name"]
    assert "get_weather" in steps[2]["function_name"]
    assert "validate_tool_output" in steps[3]["function_name"]
    assert steps[4]["function_name"] == "_model_call_step"
    assert "check_output" in steps[5]["function_name"]


@pytest.mark.asyncio
async def test_handoff(dbos_env: None) -> None:
    """DBOSRunner handles agent handoffs between multiple agents."""

    @function_tool
    @DBOS.step()
    async def get_weather(city: str) -> str:
        """Get the weather for a city."""
        return f"Sunny in {city}"

    # Weather agent: handles weather queries via a tool
    weather_model = FakeModel(
        [
            make_tool_call_response("call_w1", "get_weather", '{"city": "NYC"}'),
            make_message_response("The weather in NYC is sunny."),
        ]
    )
    weather_agent = Agent(
        name="weather_agent", model=weather_model, tools=[get_weather]
    )

    # Router agent: hands off to the weather agent
    router_model = FakeModel(
        [
            make_tool_call_response("call_h1", "transfer_to_weather_agent", "{}"),
        ]
    )
    router_agent = Agent(
        name="router",
        model=router_model,
        handoffs=[weather_agent],
    )

    @DBOS.workflow()
    async def wf(user_input: str) -> str:
        result = await DBOSRunner.run(router_agent, user_input)
        return str(result.final_output)

    output = await wf("What's the weather in NYC?")
    assert output == "The weather in NYC is sunny."

    # 1 workflow with 4 steps:
    #   model call (router → handoff), model call (weather → tool call),
    #   tool call (get_weather), model call (weather → message)
    workflows = await DBOS.list_workflows_async()
    assert len(workflows) == 1
    steps = await DBOS.list_workflow_steps_async(workflows[0].workflow_id)
    assert len(steps) == 4
    assert steps[0]["function_name"] == "_model_call_step"
    assert steps[1]["function_name"] == "_model_call_step"
    assert "get_weather" in steps[2]["function_name"]
    assert steps[3]["function_name"] == "_model_call_step"


@pytest.mark.asyncio
async def test_tool_failure(dbos_env: None) -> None:
    """When a parallel tool call fails, the SDK sends the error back to the model."""

    @function_tool
    @DBOS.step()
    async def good_tool(city: str) -> str:
        """A tool that succeeds."""
        return f"Result for {city}"

    @function_tool
    @DBOS.step()
    async def bad_tool(city: str) -> str:
        """A tool that always fails."""
        raise ValueError("Something went wrong")

    model = FakeModel(
        [
            ModelResponse(
                output=[
                    ResponseFunctionToolCall(
                        type="function_call",
                        call_id="call_1",
                        name="good_tool",
                        arguments='{"city": "NYC"}',
                    ),
                    ResponseFunctionToolCall(
                        type="function_call",
                        call_id="call_2",
                        name="bad_tool",
                        arguments='{"city": "LA"}',
                    ),
                ],
                usage=Usage(),
                response_id="resp_1",
            ),
            make_message_response("Handled the error."),
        ]
    )
    agent = Agent(name="test", model=model, tools=[good_tool, bad_tool])

    @DBOS.workflow()
    async def wf(user_input: str) -> str:
        result = await DBOSRunner.run(agent, user_input)
        return str(result.final_output)

    output = await wf("Do things")
    assert output == "Handled the error."

    # 1 workflow with 4 steps:
    #   model call (returns 2 tool calls), good_tool, bad_tool (error),
    #   model call (returns final message)
    workflows = await DBOS.list_workflows_async()
    assert len(workflows) == 1
    steps = await DBOS.list_workflow_steps_async(workflows[0].workflow_id)
    assert len(steps) == 4
    assert steps[0]["function_name"] == "_model_call_step"
    assert "good_tool" in steps[1]["function_name"]
    assert steps[1]["output"] == "Result for NYC"
    assert steps[1]["error"] is None
    assert "bad_tool" in steps[2]["function_name"]
    assert steps[2]["output"] is None
    assert "Something went wrong" in str(steps[2]["error"])
    assert steps[3]["function_name"] == "_model_call_step"


@pytest.mark.asyncio
async def test_explicit_handoff(dbos_env: None) -> None:
    """DBOSRunner handles explicit Handoff objects (not raw Agent)."""

    @function_tool
    @DBOS.step()
    async def get_weather(city: str) -> str:
        """Get the weather for a city."""
        return f"Sunny in {city}"

    weather_model = FakeModel(
        [
            make_tool_call_response("call_w1", "get_weather", '{"city": "NYC"}'),
            make_message_response("The weather in NYC is sunny."),
        ]
    )
    weather_agent = Agent(
        name="weather_agent", model=weather_model, tools=[get_weather]
    )

    router_model = FakeModel(
        [
            make_tool_call_response("call_h1", "transfer_to_weather_agent", "{}"),
        ]
    )
    router_agent = Agent(
        name="router",
        model=router_model,
        handoffs=[handoff(weather_agent)],
    )

    @DBOS.workflow()
    async def wf(user_input: str) -> str:
        result = await DBOSRunner.run(router_agent, user_input)
        return str(result.final_output)

    output = await wf("What's the weather in NYC?")
    assert output == "The weather in NYC is sunny."

    # 1 workflow with 4 steps:
    #   model call (router → handoff), model call (weather → tool call),
    #   tool call (get_weather), model call (weather → message)
    workflows = await DBOS.list_workflows_async()
    assert len(workflows) == 1
    steps = await DBOS.list_workflow_steps_async(workflows[0].workflow_id)
    assert len(steps) == 4
    assert steps[0]["function_name"] == "_model_call_step"
    assert steps[1]["function_name"] == "_model_call_step"
    assert "get_weather" in steps[2]["function_name"]
    assert steps[3]["function_name"] == "_model_call_step"


@pytest.mark.asyncio
async def test_replay(dbos_env: None) -> None:
    """Forking a completed workflow replays all steps from recorded outputs."""
    call_count = 0

    @function_tool
    @DBOS.step()
    async def get_weather(city: str) -> str:
        """Get the weather for a city."""
        nonlocal call_count
        call_count += 1
        return f"Sunny in {city}"

    model = FakeModel(
        [
            make_tool_call_response("call_1", "get_weather", '{"city": "NYC"}'),
            make_message_response("The weather in NYC is sunny."),
        ]
    )
    agent = Agent(name="test", model=model, tools=[get_weather])

    @DBOS.workflow()
    async def wf(user_input: str) -> str:
        result = await DBOSRunner.run(agent, user_input)
        return str(result.final_output)

    # Run the workflow for the first time
    output = await wf("What's the weather in NYC?")
    assert output == "The weather in NYC is sunny."
    assert call_count == 1

    workflows = await DBOS.list_workflows_async()
    assert len(workflows) == 1
    original_id = workflows[0].workflow_id
    steps = await DBOS.list_workflow_steps_async(original_id)
    assert len(steps) == 3

    # Fork from past the last step so all steps replay from recorded outputs.
    # function_ids are 1-based, so start_step must exceed the max function_id.
    max_function_id = steps[-1]["function_id"]
    handle = await DBOS.fork_workflow_async(original_id, max_function_id + 1)
    replay_output = await handle.get_result()
    assert replay_output == "The weather in NYC is sunny."

    # The tool was NOT re-executed during replay
    assert call_count == 1


@pytest.mark.asyncio
async def test_string_model_name(dbos_env: None) -> None:
    """When agent.model is a string, DBOSModelProvider resolves and wraps it."""
    from unittest.mock import patch

    from agents.models.multi_provider import MultiProvider

    fake = FakeModel([make_message_response("Hello!")])
    agent = Agent(name="test", model="fake-model")

    with patch.object(MultiProvider, "get_model", return_value=fake):

        @DBOS.workflow()
        async def wf(user_input: str) -> str:
            result = await DBOSRunner.run(agent, user_input)
            return str(result.final_output)

        output = await wf("Hi")

    assert output == "Hello!"

    workflows = await DBOS.list_workflows_async()
    assert len(workflows) == 1
    steps = await DBOS.list_workflow_steps_async(workflows[0].workflow_id)
    assert len(steps) == 1
    assert steps[0]["function_name"] == "_model_call_step"


@pytest.mark.asyncio
async def test_streamed_message(dbos_env: None) -> None:
    """DBOSRunner streams a durable model response and returns its final output."""
    model = FakeModel(
        [make_message_response("Hello!")],
        stream_events=[
            [
                ResponseTextDeltaEvent(
                    type="response.output_text.delta",
                    sequence_number=1,
                    content_index=0,
                    delta="Hello",
                    item_id="msg_1",
                    logprobs=[],
                    output_index=0,
                )
            ]
        ],
    )
    agent = Agent(name="test", model=model)

    @DBOS.workflow()
    async def wf(user_input: str) -> tuple[str, list[str]]:
        result = DBOSRunner.run_streamed(agent, user_input)
        events = [event async for event in result.stream_events()]
        raw_events = [
            str(event.data.type)
            for event in events
            if isinstance(event, RawResponsesStreamEvent)
        ]
        return str(result.final_output), raw_events

    output, raw_events = await wf("Hi")

    assert output == "Hello!"
    assert raw_events == ["response.output_text.delta", "response.completed"]

    workflows = await DBOS.list_workflows_async()
    assert len(workflows) == 1
    steps = await DBOS.list_workflow_steps_async(workflows[0].workflow_id)
    assert len(steps) == 1
    assert steps[0]["function_name"] == "_model_stream_step"


@pytest.mark.asyncio
async def test_streamed_message_retries_an_empty_provider_stream(
    dbos_env: None,
) -> None:
    """An empty provider stream is retried before the SDK rejects the turn."""

    class EmptyThenCompletedStreamModel(FakeModel):
        def stream_response(
            self, *args: Any, **kwargs: Any
        ) -> AsyncIterator[TResponseStreamEvent]:
            if self.call_count == 0:
                self.call_count += 1

                async def empty_events() -> AsyncIterator[TResponseStreamEvent]:
                    if False:
                        yield cast(TResponseStreamEvent, None)

                return empty_events()
            return super().stream_response(*args, **kwargs)

    model = EmptyThenCompletedStreamModel(
        [make_message_response("discarded"), make_message_response("Hello!")]
    )
    agent = Agent(name="test", model=model)

    @DBOS.workflow()
    async def wf(user_input: str) -> str:
        result = DBOSRunner.run_streamed(agent, user_input)
        async for _ in result.stream_events():
            pass
        return str(result.final_output)

    assert await wf("Hi") == "Hello!"
    assert model.call_count == 2


@pytest.mark.asyncio
async def test_streamed_message_reports_empty_stream_after_retry(
    dbos_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two empty attempts report a serializable incomplete stream failure."""

    class EmptyStreamModel(FakeModel):
        def stream_response(
            self, *args: Any, **kwargs: Any
        ) -> AsyncIterator[TResponseStreamEvent]:
            self.call_count += 1

            async def empty_events() -> AsyncIterator[TResponseStreamEvent]:
                if False:
                    yield cast(TResponseStreamEvent, None)

            return empty_events()

    close_calls: list[str] = []

    async def close_stream(key: str) -> None:
        close_calls.append(key)

    monkeypatch.setattr(DBOS, "close_stream_async", close_stream)
    model = EmptyStreamModel([make_message_response("discarded")] * 2)

    @DBOS.workflow()
    async def wf() -> None:
        result = DBOSRunner.run_streamed(
            Agent(name="test", model=model), "Hi", stream_key="agent-events"
        )
        async for _ in result.stream_events():
            pass

    with pytest.raises(
        RuntimeError,
        match="Agents SDK stream failed: Model stream ended without events after retry",
    ) as raised:
        await wf()

    assert raised.value.__cause__ is None
    assert model.call_count == 2
    assert close_calls == ["agent-events"]


@pytest.mark.asyncio
async def test_streamed_model_persists_text_before_response_completion(
    dbos_env: None,
) -> None:
    """A raw text delta is durable before the streamed model response completes."""
    allow_completion = asyncio.Event()

    class BlockingStreamModel(FakeModel):
        def stream_response(
            self, *args: Any, **kwargs: Any
        ) -> AsyncIterator[TResponseStreamEvent]:
            response = self.responses[self.call_count]
            self.call_count += 1

            async def events() -> AsyncIterator[TResponseStreamEvent]:
                yield ResponseTextDeltaEvent(
                    type="response.output_text.delta",
                    sequence_number=1,
                    content_index=0,
                    delta="Hello",
                    item_id="msg_1",
                    logprobs=[],
                    output_index=0,
                )
                await allow_completion.wait()
                yield ResponseCompletedEvent(
                    type="response.completed",
                    sequence_number=2,
                    response=Response.model_construct(
                        id=response.response_id,
                        created_at=0.0,
                        model="fake-model",
                        object="response",
                        output=response.output,
                        parallel_tool_calls=False,
                        tool_choice="auto",
                        tools=[],
                        usage=None,
                    ),
                )

            return events()

    agent = Agent(
        name="test",
        model=BlockingStreamModel([make_message_response("Hello!")]),
    )

    @DBOS.workflow()
    async def wf(user_input: str) -> str:
        result = DBOSRunner.run_streamed(agent, user_input, stream_key="agent-events")
        async for _ in result.stream_events():
            pass
        return str(result.final_output)

    handle = await DBOS.start_workflow_async(wf, "Hi")
    first = await anext(
        DBOS.read_stream_async(handle.get_workflow_id(), "agent-events")
    )
    assert isinstance(first, RawResponsesStreamEvent)
    assert isinstance(first.data, ResponseTextDeltaEvent)
    assert first.data.delta == "Hello"
    assert not cast(Any, handle).task.done()

    allow_completion.set()
    assert await handle.get_result() == "Hello!"


@pytest.mark.asyncio
async def test_streamed_workflow_id_replays_typed_dbos_stream(dbos_env: None) -> None:
    """A duplicate workflow ID replays the closed typed stream without a new model call."""
    model = FakeModel(
        [make_message_response("Hello!")],
        stream_events=[
            [
                ResponseTextDeltaEvent(
                    type="response.output_text.delta",
                    sequence_number=1,
                    content_index=0,
                    delta="Hello",
                    item_id="msg_1",
                    logprobs=[],
                    output_index=0,
                )
            ]
        ],
    )
    agent = Agent(name="test", model=model)

    @DBOS.workflow()
    async def wf(user_input: str) -> str:
        result = DBOSRunner.run_streamed(agent, user_input, stream_key="agent-events")
        async for _ in result.stream_events():
            pass
        return str(result.final_output)

    async def read_events(workflow_id: str) -> list[RawResponsesStreamEvent]:
        return [
            event async for event in DBOS.read_stream_async(workflow_id, "agent-events")
        ]

    with SetWorkflowID("request-1"):
        first_handle = await DBOS.start_workflow_async(wf, "Hi")
    first_events = await asyncio.wait_for(
        read_events(first_handle.get_workflow_id()), timeout=5
    )
    assert await first_handle.get_result() == "Hello!"

    with SetWorkflowID("request-1"):
        second_handle = await DBOS.start_workflow_async(wf, "Hi")
    second_events = await asyncio.wait_for(
        read_events(second_handle.get_workflow_id()), timeout=5
    )
    assert await second_handle.get_result() == "Hello!"

    assert first_events == second_events
    assert all(isinstance(event, RawResponsesStreamEvent) for event in second_events)
    assert [event.data.type for event in second_events] == [
        "response.output_text.delta",
        "response.completed",
    ]
    assert model.call_count == 1


@pytest.mark.asyncio
async def test_streamed_tool_call_replays_durably(dbos_env: None) -> None:
    """Streamed tool calls replay without recalling the model or tool."""
    tool_calls: list[str] = []

    @function_tool
    @DBOS.step()
    async def get_weather(city: str) -> str:
        """Return the weather for a city."""
        tool_calls.append(city)
        return f"Sunny in {city}"

    model = FakeModel(
        [
            make_tool_call_response("call_1", "get_weather", '{"city": "NYC"}'),
            make_message_response("The weather in NYC is sunny."),
        ]
    )
    agent = Agent(name="test", model=model, tools=[get_weather])

    @DBOS.workflow()
    async def wf(user_input: str) -> str:
        result = DBOSRunner.run_streamed(agent, user_input)
        async for _ in result.stream_events():
            pass
        return str(result.final_output)

    output = await wf("What is the weather in NYC?")
    assert output == "The weather in NYC is sunny."
    assert tool_calls == ["NYC"]
    assert model.call_count == 2

    workflows = await DBOS.list_workflows_async()
    assert len(workflows) == 1
    workflow_id = workflows[0].workflow_id
    steps = await DBOS.list_workflow_steps_async(workflow_id)
    assert len(steps) == 3
    assert steps[0]["function_name"] == "_model_stream_step"
    assert "get_weather" in steps[1]["function_name"]
    assert steps[2]["function_name"] == "_model_stream_step"

    handle = await DBOS.fork_workflow_async(workflow_id, steps[-1]["function_id"] + 1)
    replay_output = await handle.get_result()
    assert replay_output == "The weather in NYC is sunny."
    assert tool_calls == ["NYC"]
    assert model.call_count == 2


@pytest.mark.asyncio
async def test_unwrapped_custom_tool_does_not_block_function_tool_turnstile(
    dbos_env: None,
) -> None:
    """Ordinary custom tools must not consume a turnstile slot they cannot release."""
    calls: list[str] = []

    async def ordinary_custom(_: object, raw_input: str) -> str:
        calls.append(f"custom:{raw_input}")
        return "custom result"

    @function_tool
    async def ordinary_function() -> str:
        """Return an ordinary local function result."""
        calls.append("function")
        return "function result"

    custom_tool = CustomTool(
        name="ordinary_custom",
        description="An ordinary custom tool.",
        on_invoke_tool=ordinary_custom,
    )
    model = FakeModel(
        [
            ModelResponse(
                output=[
                    ResponseCustomToolCall(
                        type="custom_tool_call",
                        call_id="custom",
                        name="ordinary_custom",
                        input="payload",
                    ),
                    ResponseFunctionToolCall(
                        type="function_call",
                        call_id="function",
                        name="ordinary_function",
                        arguments="{}",
                    ),
                ],
                usage=Usage(),
                response_id="resp_1",
            ),
            make_message_response("done"),
        ]
    )
    agent = Agent(
        name="mixed-tools",
        model=model,
        tools=[custom_tool, ordinary_function],
    )

    @DBOS.workflow()
    async def workflow() -> str:
        result = await DBOSRunner.run(agent, "run both tools")
        return str(result.final_output)

    assert await asyncio.wait_for(workflow(), timeout=1) == "done"
    assert calls == ["custom:payload", "function"]


@pytest.mark.asyncio
async def test_process_stream_forwards_sdk_events_without_writing_or_closing_dbos_stream(
    dbos_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The legacy helper drives an SDK stream but does not own DBOS transport."""
    model = FakeModel(
        [make_message_response("Hello!")],
        stream_events=[
            [
                ResponseTextDeltaEvent(
                    type="response.output_text.delta",
                    sequence_number=1,
                    content_index=0,
                    delta="Hello",
                    item_id="msg_1",
                    logprobs=[],
                    output_index=0,
                )
            ]
        ],
    )
    agent = Agent(name="test", model=model)
    write_calls: list[object] = []
    close_calls: list[object] = []

    async def write_stream(*args: object, **kwargs: object) -> None:
        write_calls.append((args, kwargs))

    async def close_stream(*args: object, **kwargs: object) -> None:
        close_calls.append((args, kwargs))

    monkeypatch.setattr(DBOS, "write_stream_async", write_stream)
    monkeypatch.setattr(DBOS, "close_stream_async", close_stream)

    @DBOS.workflow()
    async def wf(user_input: str) -> list[str]:
        result = DBOSRunner.run_streamed(agent, user_input)
        return [
            event.data.type
            async for event in process_stream(result, "legacy-agent-events")
            if isinstance(event, RawResponsesStreamEvent)
        ]

    assert await wf("Hi") == ["response.output_text.delta", "response.completed"]
    assert write_calls == []
    assert close_calls == []


@pytest.mark.asyncio
async def test_agent_tool_stream_key_persists_typed_runner_events(
    dbos_env: None,
) -> None:
    """A keyed nested agent delegates durable stream ownership to DBOSRunner."""
    model = FakeModel(
        [make_message_response("Hello!")],
        stream_events=[
            [
                ResponseTextDeltaEvent(
                    type="response.output_text.delta",
                    sequence_number=1,
                    content_index=0,
                    delta="Hello",
                    item_id="msg_1",
                    logprobs=[],
                    output_index=0,
                )
            ]
        ],
    )
    tool = DBOSAgentTool(
        Agent(name="nested", model=model),
        tool_name="ask_nested",
        tool_description="Ask the nested agent.",
        stream_key="nested-agent-events",
    )

    @DBOS.workflow()
    async def wf() -> str:
        return str(
            await tool.on_invoke_tool(
                ToolContext(
                    None,
                    tool_name="ask_nested",
                    tool_call_id="call_nested",
                    tool_arguments='{"input":"Hi"}',
                ),
                '{"input":"Hi"}',
            )
        )

    handle = await DBOS.start_workflow_async(wf)
    events = await asyncio.wait_for(
        collect_stream_events(handle.get_workflow_id(), "nested-agent-events"),
        timeout=5,
    )
    assert await handle.get_result() == "Hello!"
    assert all(isinstance(event, RawResponsesStreamEvent) for event in events)
    assert [event.data.type for event in events] == [
        "response.output_text.delta",
        "response.completed",
    ]


@pytest.mark.asyncio
async def test_process_stream_normalizes_nonserializable_errors() -> None:
    """Compatibility forwarding raises a plain, serializable stream failure."""

    class FailingResult:
        async def stream_events(self) -> AsyncIterator[object]:
            error = RuntimeError("boom")
            error.lock = threading.Lock()  # type: ignore[attr-defined]
            raise error
            yield object()

    with pytest.raises(RuntimeError, match="Agents SDK stream failed: boom") as raised:
        async for _ in process_stream(cast(Any, FailingResult()), "legacy-events"):
            pass

    assert raised.value.__cause__ is None
    pickle.dumps(raised.value)


async def collect_stream_events(
    workflow_id: str, stream_key: str
) -> list[RawResponsesStreamEvent]:
    return [event async for event in DBOS.read_stream_async(workflow_id, stream_key)]


@pytest.mark.asyncio
async def test_keyed_runner_forwards_raw_events_and_closes_once(
    dbos_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Runner-owned transport preserves SDK events and has one close owner."""
    model = FakeModel(
        [make_message_response("Hello!")],
        stream_events=[
            [
                ResponseTextDeltaEvent(
                    type="response.output_text.delta",
                    sequence_number=1,
                    content_index=0,
                    delta="Hello",
                    item_id="msg_1",
                    logprobs=[],
                    output_index=0,
                )
            ]
        ],
    )
    close_calls: list[str] = []

    async def close_stream(key: str) -> None:
        close_calls.append(key)

    monkeypatch.setattr(DBOS, "close_stream_async", close_stream)

    @DBOS.workflow()
    async def wf() -> list[RawResponsesStreamEvent]:
        result = DBOSRunner.run_streamed(
            Agent(name="test", model=model), "Hi", stream_key="runner-events"
        )
        return [
            event
            async for event in result.stream_events()
            if isinstance(event, RawResponsesStreamEvent)
        ]

    events = await wf()
    assert [event.data.type for event in events] == [
        "response.output_text.delta",
        "response.completed",
    ]
    assert close_calls == ["runner-events"]


@pytest.mark.asyncio
async def test_keyed_runner_normalizes_unpickleable_stream_error_and_closes_once(
    dbos_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Runner converts an SDK stream error before DBOS persists the workflow error."""

    class FailingStreamModel(FakeModel):
        def stream_response(
            self, *args: Any, **kwargs: Any
        ) -> AsyncIterator[TResponseStreamEvent]:
            async def events() -> AsyncIterator[TResponseStreamEvent]:
                error = RuntimeError("boom")
                error.lock = threading.Lock()  # type: ignore[attr-defined]
                raise error
                yield cast(TResponseStreamEvent, None)

            return events()

    close_calls: list[str] = []

    async def close_stream(key: str) -> None:
        close_calls.append(key)

    monkeypatch.setattr(DBOS, "close_stream_async", close_stream)
    agent = Agent(name="test", model=FailingStreamModel([make_message_response("x")]))

    @DBOS.workflow()
    async def wf() -> str:
        result = DBOSRunner.run_streamed(agent, "Hi", stream_key="runner-events")
        with pytest.raises(
            RuntimeError, match="Agents SDK stream failed: boom"
        ) as raised:
            async for _ in result.stream_events():
                pass
        assert raised.value.__cause__ is None
        pickle.dumps(raised.value)
        return str(raised.value)

    assert await wf() == "Agents SDK stream failed: boom"
    assert close_calls == ["runner-events"]


@pytest.mark.asyncio
async def test_keyed_runner_closes_once_when_stream_consumer_is_cancelled(
    dbos_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancelling the real runner event consumer still releases its DBOS stream."""
    entered_stream = asyncio.Event()
    block_stream = asyncio.Event()

    class BlockingStreamModel(FakeModel):
        def stream_response(
            self, *args: Any, **kwargs: Any
        ) -> AsyncIterator[TResponseStreamEvent]:
            async def events() -> AsyncIterator[TResponseStreamEvent]:
                yield ResponseTextDeltaEvent(
                    type="response.output_text.delta",
                    sequence_number=1,
                    content_index=0,
                    delta="Hello",
                    item_id="msg_1",
                    logprobs=[],
                    output_index=0,
                )
                entered_stream.set()
                await block_stream.wait()

            return events()

    close_calls: list[str] = []

    async def close_stream(key: str) -> None:
        close_calls.append(key)

    monkeypatch.setattr(DBOS, "close_stream_async", close_stream)
    agent = Agent(name="test", model=BlockingStreamModel([make_message_response("x")]))

    @DBOS.workflow()
    async def wf() -> None:
        result = DBOSRunner.run_streamed(agent, "Hi", stream_key="runner-events")
        event_stream = cast(AsyncGenerator[Any, None], result.stream_events())
        consumer: asyncio.Future[Any] = asyncio.ensure_future(anext(event_stream))
        await entered_stream.wait()
        consumer.cancel()
        with suppress(asyncio.CancelledError):
            await consumer
        await event_stream.aclose()

    await asyncio.wait_for(wf(), timeout=2)
    assert close_calls == ["runner-events"]
