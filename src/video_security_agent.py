from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class Tool(Protocol):
    name: str

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        ...


@dataclass
class FunctionTool:
    name: str
    func: Any

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        return self.func(context)


class VideoSecurityAgent:
    """A lightweight local agent that executes the video security tool chain."""

    def __init__(self, tools: list[Tool]) -> None:
        self.tools = tools

    def run(self, initial_context: dict[str, Any]) -> dict[str, Any]:
        context = dict(initial_context)
        trace = []
        for tool in self.tools:
            context = tool.run(context)
            trace.append(tool.name)
        context["agent_trace"] = trace
        return context
