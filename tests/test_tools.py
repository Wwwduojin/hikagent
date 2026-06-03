from __future__ import annotations

from agent_solution.tools import build_default_registry


def test_default_registry_lists_and_invokes_complexity_tool() -> None:
    registry = build_default_registry()

    tools = registry.list_tools()
    result = registry.invoke(
        "estimate_complexity",
        {"core_steps": "目标检测后进行时序跟踪和多模型融合", "constraints": "需要实时低误报"},
    )

    assert tools[0].name == "estimate_complexity"
    assert result["complexity"] in {"medium", "high"}
    assert result["score"] >= 3

