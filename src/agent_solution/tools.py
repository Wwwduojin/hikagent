from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field


class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    permission: str = "local"


@dataclass
class RegisteredTool:
    spec: ToolSpec
    handler: Callable[[dict[str, Any]], dict[str, Any]]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, spec: ToolSpec, handler: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        self._tools[spec.name] = RegisteredTool(spec=spec, handler=handler)

    def list_tools(self) -> list[ToolSpec]:
        return [tool.spec for tool in self._tools.values()]

    def invoke(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name].handler(arguments)


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="estimate_complexity",
            description="Estimate implementation complexity for an algorithm solution.",
            input_schema={
                "type": "object",
                "properties": {
                    "core_steps": {"type": "string"},
                    "constraints": {"type": "string"},
                },
                "required": ["core_steps"],
            },
        ),
        estimate_complexity,
    )
    return registry


def estimate_complexity(arguments: dict[str, Any]) -> dict[str, Any]:
    text = f"{arguments.get('core_steps', '')} {arguments.get('constraints', '')}"
    signals = ["时序", "多模型", "实时", "低误报", "边缘设备", "遮挡", "多摄像头"]
    score = 1 + sum(1 for signal in signals if signal in text)
    if score <= 2:
        level = "low"
    elif score <= 4:
        level = "medium"
    else:
        level = "high"
    return {
        "complexity": level,
        "score": score,
        "notes": "Complexity is estimated from workflow and constraint keywords.",
    }

