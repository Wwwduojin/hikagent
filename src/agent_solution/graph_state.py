from __future__ import annotations

import json
from typing import Any, Callable, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import MessagesState

from agent_solution.models import AgentState, ChatMessage


class GraphState(MessagesState, total=False):
    user_id: str
    thread_id: str | None
    session_type: str
    current_app_id: str | None
    current_app_name: str | None
    intent_type: str | None
    switch_type: str | None
    detected_app_id: str | None
    detected_app_name: str | None
    detected_apps: list[dict[str, str]]
    intent_confidence: float | None
    intent_reason: str | None
    is_kb_satisfaction_feedback: bool
    kb_satisfaction: str | None
    is_solution_confirmation: bool
    user_profile: dict[str, Any]
    user_constraints: dict[str, Any]
    slots: dict[str, Any]
    missing_slots: list[str]
    missing_required_slots: list[str]
    missing_optional_slots: list[str]
    stage: str
    active_agent: str
    kb_context: list[dict[str, Any]]
    kb_cases: list[dict[str, Any]]
    kb_has_match: bool
    user_satisfaction: str
    recommended_solution: dict[str, Any] | None
    recommended_solution_reason: str | None
    algorithm_main_description: str | None
    algorithm_plan_request: dict[str, Any] | None
    algorithm_plan_raw_response: dict[str, Any] | None
    algorithm_plan_normalized: dict[str, Any] | None
    comparison_result: dict[str, Any] | None
    general_recommendations: list[dict[str, Any]]
    draft_summary_markdown: str | None
    draft_solution_json: dict[str, Any] | None
    assistant_reply: str | None
    status: Literal["collecting", "reviewing", "confirmed"]
    tool_calls: list[dict[str, Any]]
    tool_recall_result: dict[str, Any] | None
    selected_tools: list[dict[str, Any]]
    tool_topology: dict[str, Any] | None
    tool_parameters: dict[str, Any] | None
    orchestration_judgement: dict[str, Any] | None
    summary: str
    unresolved_slots: list[str]
    next_agent: Literal["knowledge_base_agent", "recommendation_agent", "tool_orchestration_agent", "end"] | None
    turn_complete: bool


def node_adapter(func: Callable[[AgentState], AgentState]) -> Callable[[GraphState], GraphState]:
    def wrapped(state: GraphState) -> GraphState:
        before = agent_state_from_graph(state)
        previous_message_count = len(before.messages)
        after = func(before)
        return graph_update(previous_message_count, after)

    return wrapped


def agent_state_from_graph(state: GraphState) -> AgentState:
    data = dict(state)
    data["messages"] = [coerce_message(message) for message in data.get("messages", [])]
    return AgentState.model_validate(data)


def coerce_message(message: Any) -> dict[str, str]:
    role_map = {
        "ai": "assistant",
        "assistant": "assistant",
        "human": "user",
        "user": "user",
        "system": "system",
    }
    if isinstance(message, ChatMessage):
        return message.model_dump()
    if isinstance(message, BaseMessage):
        return {"role": role_map.get(message.type, "user"), "content": string_content(message.content)}
    if isinstance(message, dict):
        role = message.get("role") or message.get("type") or "user"
        return {"role": role_map.get(str(role), "user"), "content": string_content(message.get("content", ""))}
    return {"role": "user", "content": str(message)}


def string_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        if text_parts:
            return "\n".join(part for part in text_parts if part)
    return json.dumps(content, ensure_ascii=False)


def graph_update(previous_message_count: int, after: AgentState) -> GraphState:
    update = after.model_dump()
    new_messages = update.pop("messages")[previous_message_count:]
    if new_messages:
        update["messages"] = new_messages
    return update
