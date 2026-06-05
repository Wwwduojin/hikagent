from __future__ import annotations

import importlib

import pytest

pytest.importorskip("langgraph")

from agent_solution.graph import append_user_message, initial_state


def test_langgraph_app_exports_invokable_graph(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AGENT_SOLUTION_SIMULATE", "true")
    monkeypatch.setenv("AGENT_SOLUTION_DB", str(tmp_path / "langgraph_app.sqlite"))

    app_module = importlib.import_module("agent_solution.langgraph_app")
    result = app_module.graph.invoke(append_user_message(initial_state(), "我想做一个抽烟识别算法"))

    assert result["status"] in {"collecting", "reviewing"}
    assert result["assistant_reply"]
    assert result["messages"][-1].content == result["assistant_reply"]


def test_chat_graph_exposes_standard_messages_schema(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AGENT_SOLUTION_SIMULATE", "true")
    monkeypatch.setenv("AGENT_SOLUTION_DB", str(tmp_path / "langgraph_chat_schema.sqlite"))

    app_module = importlib.import_module("agent_solution.langgraph_app")
    messages_schema = app_module.chat_graph.get_input_jsonschema()["properties"]["messages"]

    message_types = {item["$ref"] for item in messages_schema["items"]["oneOf"]}
    assert "#/$defs/HumanMessage" in message_types
    assert "#/$defs/AIMessage" in message_types


def test_graph_exposes_agent_subgraphs(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AGENT_SOLUTION_SIMULATE", "true")
    monkeypatch.setenv("AGENT_SOLUTION_DB", str(tmp_path / "langgraph_subgraphs.sqlite"))

    app_module = importlib.import_module("agent_solution.langgraph_app")
    top_level = set(app_module.chat_graph.get_graph().nodes)
    expanded = set(app_module.chat_graph.get_graph(xray=True).nodes)
    intent_nodes = set(app_module.intent_router_graph.get_graph().nodes)

    assert top_level == {
        "__start__",
        "supervisor_agent",
        "knowledge_base_agent",
        "recommendation_agent",
        "tool_orchestration_agent",
        "__end__",
    }
    assert "knowledge_base_agent:retrieve_knowledge" in expanded
    assert "recommendation_agent:algorithm_plan_api_tool" in expanded
    assert "supervisor_agent:generate_summary" in expanded
    assert "tool_orchestration_agent:tool_recall_node" in expanded
    assert "tool_orchestration_agent:judge_agent" in expanded
    assert "intent_recognition_agent" in intent_nodes
