from typing import Any, AsyncIterator

from agents import Usage
from agents.items import ModelResponse, TResponseStreamEvent
from agents.models.interface import Model
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)


def make_message_response(text: str, response_id: str = "resp_1") -> ModelResponse:
    return ModelResponse(
        output=[
            ResponseOutputMessage(
                type="message",
                id="msg_1",
                role="assistant",
                status="completed",
                content=[
                    ResponseOutputText(type="output_text", text=text, annotations=[])
                ],
            )
        ],
        usage=Usage(),
        response_id=response_id,
    )


def make_tool_call_response(
    call_id: str, tool_name: str, arguments: str, response_id: str = "resp_1"
) -> ModelResponse:
    return ModelResponse(
        output=[
            ResponseFunctionToolCall(
                type="function_call",
                call_id=call_id,
                name=tool_name,
                arguments=arguments,
            )
        ],
        usage=Usage(),
        response_id=response_id,
    )


class FakeModel(Model):
    """A model that returns a sequence of canned responses."""

    def __init__(
        self,
        responses: list[ModelResponse],
        stream_events: list[list[TResponseStreamEvent]] | None = None,
    ):
        self.responses = list(responses)
        self.stream_events = stream_events
        self.call_count = 0

    async def get_response(
        self,
        system_instructions: Any,
        input: Any,
        model_settings: Any,
        tools: Any,
        output_schema: Any,
        handoffs: Any,
        tracing: Any,
        *,
        previous_response_id: Any = None,
        conversation_id: Any = None,
        prompt: Any = None,
    ) -> ModelResponse:
        resp = self.responses[self.call_count]
        self.call_count += 1
        return resp

    def stream_response(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[TResponseStreamEvent]:
        response = self.responses[self.call_count]
        self.call_count += 1

        async def events() -> AsyncIterator[TResponseStreamEvent]:
            if self.stream_events is not None:
                for event in self.stream_events[self.call_count - 1]:
                    yield event

            yield ResponseCompletedEvent(
                type="response.completed",
                sequence_number=1,
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
