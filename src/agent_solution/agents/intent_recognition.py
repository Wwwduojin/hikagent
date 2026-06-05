from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from agent_solution.app_router import IntentRecognitionAgent
from agent_solution.models import AppSession


class IntentRouterState(TypedDict, total=False):
    user_input: str
    current_session: dict[str, Any] | None
    summary: str
    intent: dict[str, Any]


def build_intent_router_graph(llm):
    agent = IntentRecognitionAgent(llm)

    def intent_recognition_agent(state: IntentRouterState) -> IntentRouterState:
        session_data = state.get("current_session")
        session = AppSession.model_validate(session_data) if session_data else None
        intent = agent.predict(
            state.get("user_input", ""),
            current_session=session,
            summary=state.get("summary", ""),
        )
        return {"intent": intent.model_dump()}

    graph = StateGraph(IntentRouterState)
    graph.add_node("intent_recognition_agent", intent_recognition_agent)
    graph.set_entry_point("intent_recognition_agent")
    graph.add_edge("intent_recognition_agent", END)
    return graph.compile(checkpointer=False)
