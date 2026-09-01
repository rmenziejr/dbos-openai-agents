"""Durable wrappers for sandbox capabilities."""

import copy
import dataclasses
from typing import Any

from agents.sandbox.capabilities import Capability
from agents.tool import CustomTool, FunctionTool, Tool
from agents.tool_context import ToolContext
from pydantic import PrivateAttr

from .durability import DEFAULT_AUDIT_STREAM_KEY, run_durable_action


class DBOSCapability(Capability):
    """Opt a sandbox capability's native tool actions into DBOS durability."""

    type: str = ""
    capability: Capability
    audit_stream_key: str = DEFAULT_AUDIT_STREAM_KEY
    _state: Any = PrivateAttr(default=None)

    def __init__(
        self,
        capability: Capability,
        *,
        audit_stream_key: str = DEFAULT_AUDIT_STREAM_KEY,
    ) -> None:
        values: dict[str, Any] = {
            "type": capability.type,
            "capability": capability,
            "audit_stream_key": audit_stream_key,
        }
        super().__init__(**values)

    def model_post_init(self, context: Any) -> None:
        _ = context
        self.type = self.capability.type

    def clone(self) -> "DBOSCapability":
        cloned = DBOSCapability(
            capability=self.capability.clone(),
            audit_stream_key=self.audit_stream_key,
        )
        if self._state is not None:
            cloned.bind_durability_state(self._state)
        return cloned

    def bind_durability_state(self, state: Any) -> None:
        """Bind the DBOS run-local state used by instrumented native tools."""
        self._state = state

    def bind(self, session: Any) -> None:
        self.capability.bind(session)

    def bind_run_as(self, user: Any) -> None:
        self.capability.bind_run_as(user)

    def bind_workspace_scope(self, scope: Any) -> None:
        self.capability.bind_workspace_scope(scope)

    def required_capability_types(self) -> set[str]:
        return self.capability.required_capability_types()

    def tools(self) -> list[Tool]:
        if self._state is None:
            raise RuntimeError("DBOSCapability is not bound to a DBOS runner state")
        return [
            instrument_tool(tool, self._state, self.type, self.audit_stream_key)
            for tool in self.capability.tools()
        ]

    def process_manifest(self, manifest: Any) -> Any:
        return self.capability.process_manifest(manifest)

    async def instructions(self, manifest: Any) -> str | None:
        return await self.capability.instructions(manifest)

    def sampling_params(self, sampling_params: dict[str, Any]) -> dict[str, Any]:
        return self.capability.sampling_params(sampling_params)

    def process_context(self, context: list[Any]) -> list[Any]:
        return self.capability.process_context(context)


def instrument_tool(
    tool: Tool,
    state: Any,
    capability_type: str,
    audit_stream_key: str,
) -> Tool:
    """Return a durable copy of local sandbox tool variants."""
    if isinstance(tool, FunctionTool):
        original = tool.on_invoke_tool

        async def invoke(context: ToolContext[Any], raw_input: str) -> Any:
            return await run_durable_action(
                state=state,
                source="sandbox_capability",
                owner=capability_type,
                action=tool.name,
                call_id=context.tool_call_id,
                audit_stream_key=audit_stream_key,
                invoke=lambda: original(context, raw_input),
            )

        return _replace_on_invoke_tool(tool, invoke)

    if isinstance(tool, CustomTool):
        original = tool.on_invoke_tool

        async def invoke(context: ToolContext[Any], raw_input: str) -> Any:
            return await run_durable_action(
                state=state,
                source="sandbox_capability",
                owner=capability_type,
                action=tool.name,
                call_id=context.tool_call_id,
                audit_stream_key=audit_stream_key,
                invoke=lambda: original(context, raw_input),
            )

        wrapped = _replace_on_invoke_tool(tool, invoke)
        durable_custom_tools = getattr(state, "durable_custom_tools", None)
        if isinstance(durable_custom_tools, dict):
            durable_custom_tools[id(wrapped)] = wrapped
        return wrapped

    return tool


def _replace_on_invoke_tool(
    tool: FunctionTool | CustomTool,
    on_invoke_tool: Any,
) -> FunctionTool | CustomTool:
    """Copy a local tool without mutating the capability-provided instance."""
    try:
        return dataclasses.replace(tool, on_invoke_tool=on_invoke_tool)
    except TypeError:
        # Sandbox tool subclasses use custom constructors that cannot be called by
        # dataclasses.replace. Preserve every field and change only the callback.
        cloned = copy.copy(tool)
        cloned.on_invoke_tool = on_invoke_tool
        return cloned
