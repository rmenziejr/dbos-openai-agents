"""Durable wrappers for Agents SDK Computer Use harnesses."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar, cast

from agents import AsyncComputer, Computer, ComputerProvider, ComputerTool
from agents.computer import Button, Environment
from agents.run_context import RunContextWrapper

from .durability import (
    DEFAULT_AUDIT_STREAM_KEY,
    run_durable_action,
    run_durable_action_sync,
)

ComputerT = TypeVar("ComputerT", bound=Computer | AsyncComputer)
ResultT = TypeVar("ResultT")


class _DurableComputer(Computer):
    """Run synchronous local computer actions as durable DBOS steps."""

    def __init__(self, computer: Computer, audit_stream_key: str) -> None:
        self._computer = computer
        self._audit_stream_key = audit_stream_key

    @property
    def environment(self) -> Environment | None:
        return self._computer.environment

    @property
    def dimensions(self) -> tuple[int, int] | None:
        return self._computer.dimensions

    def _invoke(self, action: str, invoke: Callable[[], ResultT]) -> ResultT:
        return run_durable_action_sync(
            state=cast(Any, None),
            source="computer_use",
            owner="computer",
            action=action,
            call_id=None,
            audit_stream_key=self._audit_stream_key,
            invoke=invoke,
        )

    def screenshot(self) -> str:
        return self._invoke("screenshot", self._computer.screenshot)

    def click(self, x: int, y: int, button: Button) -> None:
        self._invoke("click", lambda: self._computer.click(x, y, button))

    def double_click(self, x: int, y: int) -> None:
        self._invoke("double_click", lambda: self._computer.double_click(x, y))

    def scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        self._invoke("scroll", lambda: self._computer.scroll(x, y, scroll_x, scroll_y))

    def type(self, text: str) -> None:
        self._invoke("type", lambda: self._computer.type(text))

    def wait(self) -> None:
        self._invoke("wait", self._computer.wait)

    def move(self, x: int, y: int) -> None:
        self._invoke("move", lambda: self._computer.move(x, y))

    def keypress(self, keys: list[str]) -> None:
        self._invoke("keypress", lambda: self._computer.keypress(keys))

    def drag(self, path: list[tuple[int, int]]) -> None:
        self._invoke("drag", lambda: self._computer.drag(path))


class _DurableAsyncComputer(AsyncComputer):
    """Run asynchronous local computer actions as durable DBOS steps."""

    def __init__(self, computer: AsyncComputer, audit_stream_key: str) -> None:
        self._computer = computer
        self._audit_stream_key = audit_stream_key

    @property
    def environment(self) -> Environment | None:
        return self._computer.environment

    @property
    def dimensions(self) -> tuple[int, int] | None:
        return self._computer.dimensions

    async def _invoke(
        self, action: str, invoke: Callable[[], Awaitable[ResultT]]
    ) -> ResultT:
        return await run_durable_action(
            state=cast(Any, None),
            source="computer_use",
            owner="computer",
            action=action,
            call_id=None,
            audit_stream_key=self._audit_stream_key,
            invoke=invoke,
        )

    async def screenshot(self) -> str:
        return await self._invoke("screenshot", self._computer.screenshot)

    async def click(self, x: int, y: int, button: Button) -> None:
        await self._invoke("click", lambda: self._computer.click(x, y, button))

    async def double_click(self, x: int, y: int) -> None:
        await self._invoke("double_click", lambda: self._computer.double_click(x, y))

    async def scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        await self._invoke(
            "scroll", lambda: self._computer.scroll(x, y, scroll_x, scroll_y)
        )

    async def type(self, text: str) -> None:
        await self._invoke("type", lambda: self._computer.type(text))

    async def wait(self) -> None:
        await self._invoke("wait", self._computer.wait)

    async def move(self, x: int, y: int) -> None:
        await self._invoke("move", lambda: self._computer.move(x, y))

    async def keypress(self, keys: list[str]) -> None:
        await self._invoke("keypress", lambda: self._computer.keypress(keys))

    async def drag(self, path: list[tuple[int, int]]) -> None:
        await self._invoke("drag", lambda: self._computer.drag(path))


def _wrap_computer(computer: ComputerT, audit_stream_key: str) -> ComputerT:
    if isinstance(computer, AsyncComputer):
        return cast(ComputerT, _DurableAsyncComputer(computer, audit_stream_key))
    return cast(ComputerT, _DurableComputer(computer, audit_stream_key))


def _unwrap_computer(computer: Computer | AsyncComputer) -> Computer | AsyncComputer:
    if isinstance(computer, _DurableComputer | _DurableAsyncComputer):
        return computer._computer
    return computer


def _is_provider(candidate: object) -> bool:
    if isinstance(candidate, ComputerProvider):
        return True
    if isinstance(candidate, Computer | AsyncComputer):
        return False
    return hasattr(candidate, "create") and callable(candidate.create)


def wrap_computer_initializer(
    initializer: ComputerT | Callable[..., Any] | ComputerProvider[Any],
    audit_stream_key: str,
) -> ComputerT | Callable[..., Any] | ComputerProvider[Any]:
    """Wrap direct computers and the SDK factory/provider lifecycle forms."""
    if isinstance(initializer, Computer | AsyncComputer):
        return _wrap_computer(initializer, audit_stream_key)

    if _is_provider(initializer):
        provider = cast(Any, initializer)

        async def create(
            *, run_context: RunContextWrapper[Any]
        ) -> Computer | AsyncComputer:
            computer = provider.create(run_context=run_context)
            if inspect.isawaitable(computer):
                computer = await computer
            return _wrap_computer(
                cast(Computer | AsyncComputer, computer), audit_stream_key
            )

        if provider.dispose is None:
            return ComputerProvider(create=create)

        async def dispose(
            *, run_context: RunContextWrapper[Any], computer: Computer | AsyncComputer
        ) -> None:
            result = provider.dispose(
                run_context=run_context, computer=_unwrap_computer(computer)
            )
            if inspect.isawaitable(result):
                await result

        return ComputerProvider(create=create, dispose=dispose)

    factory = cast(Callable[..., Any], initializer)

    async def create_from_factory(
        *, run_context: RunContextWrapper[Any]
    ) -> Computer | AsyncComputer:
        computer = factory(run_context=run_context)
        if inspect.isawaitable(computer):
            computer = await computer
        return _wrap_computer(
            cast(Computer | AsyncComputer, computer), audit_stream_key
        )

    return create_from_factory


def DBOSComputerTool(
    tool: ComputerTool[ComputerT], *, audit_stream_key: str = DEFAULT_AUDIT_STREAM_KEY
) -> ComputerTool[ComputerT]:
    """Return a ComputerTool whose local actions are replay-safe DBOS steps."""
    tool_kwargs: dict[str, Any] = {
        "computer": wrap_computer_initializer(tool.computer, audit_stream_key),
        "on_safety_check": tool.on_safety_check,
    }
    if "custom_data_extractor" in inspect.signature(ComputerTool).parameters:
        tool_kwargs["custom_data_extractor"] = getattr(
            tool, "custom_data_extractor", None
        )
    return cast("ComputerTool[ComputerT]", ComputerTool(**tool_kwargs))
