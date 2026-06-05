from __future__ import annotations

from collections.abc import Callable

from langgraph.graph import END, StateGraph

from agent_solution.graph_state import GraphState


def build_knowledge_base_subgraph(
    *,
    retrieve_knowledge: Callable,
    summarize_knowledge_cases: Callable,
    present_kb_cases: Callable,
    knowledge_result_decision: Callable,
    delegate_recommendation: Callable,
):
    graph = StateGraph(GraphState)
    graph.add_node("retrieve_knowledge", retrieve_knowledge)
    graph.add_node("summarize_knowledge_cases", summarize_knowledge_cases)
    graph.add_node("present_kb_cases", present_kb_cases)
    graph.add_node("delegate_recommendation", delegate_recommendation)

    graph.set_entry_point("retrieve_knowledge")
    graph.add_edge("retrieve_knowledge", "summarize_knowledge_cases")
    graph.add_conditional_edges(
        "summarize_knowledge_cases",
        knowledge_result_decision,
        {
            "present_kb_cases": "present_kb_cases",
            "delegate_recommendation": "delegate_recommendation",
        },
    )
    graph.add_edge("present_kb_cases", END)
    graph.add_edge("delegate_recommendation", END)
    return graph.compile()

