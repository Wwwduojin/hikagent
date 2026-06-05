from __future__ import annotations

from collections.abc import Callable

from langgraph.graph import END, StateGraph

from agent_solution.graph_state import GraphState


def build_supervisor_subgraph(
    *,
    extract_slots: Callable,
    supervisor_route: Callable,
    handle_kb_feedback: Callable,
    ask_followup: Callable,
    generate_summary: Callable,
    tool_router: Callable,
    confirm_solution: Callable,
    supervisor_decision: Callable,
    feedback_decision: Callable,
    delegate_knowledge_base: Callable,
    delegate_recommendation: Callable,
):
    graph = StateGraph(GraphState)
    graph.add_node("extract_slots", extract_slots)
    graph.add_node("supervisor_route", supervisor_route)
    graph.add_node("handle_kb_feedback", handle_kb_feedback)
    graph.add_node("delegate_knowledge_base", delegate_knowledge_base)
    graph.add_node("delegate_recommendation", delegate_recommendation)
    graph.add_node("ask_followup", ask_followup)
    graph.add_node("generate_summary", generate_summary)
    graph.add_node("tool_router", tool_router)
    graph.add_node("confirm_solution", confirm_solution)

    graph.set_entry_point("extract_slots")
    graph.add_edge("extract_slots", "supervisor_route")
    graph.add_conditional_edges(
        "supervisor_route",
        supervisor_decision,
        {
            "delegate_knowledge_base": "delegate_knowledge_base",
            "handle_kb_feedback": "handle_kb_feedback",
            "delegate_recommendation": "delegate_recommendation",
            "ask_followup": "ask_followup",
            "generate_summary": "generate_summary",
            "confirm_solution": "confirm_solution",
        },
    )
    graph.add_conditional_edges(
        "handle_kb_feedback",
        feedback_decision,
        {
            "delegate_knowledge_base": "delegate_knowledge_base",
            "delegate_recommendation": "delegate_recommendation",
            "ask_followup": "ask_followup",
            "generate_summary": "generate_summary",
        },
    )
    graph.add_edge("delegate_knowledge_base", END)
    graph.add_edge("delegate_recommendation", END)
    graph.add_edge("ask_followup", END)
    graph.add_edge("generate_summary", "tool_router")
    graph.add_edge("tool_router", END)
    graph.add_edge("confirm_solution", END)
    return graph.compile()

