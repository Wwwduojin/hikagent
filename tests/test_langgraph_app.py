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
