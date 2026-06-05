from __future__ import annotations

from collections.abc import Callable

from langgraph.graph import END, StateGraph

from agent_solution.graph_state import GraphState


def build_recommendation_subgraph(
    *,
    check_recommendation_requirements: Callable,
    recommendation_followup: Callable,
    build_algorithm_plan_request: Callable,
    algorithm_plan_api_tool: Callable,
    normalize_algorithm_plan: Callable,
    present_recommendation: Callable,
    recommendation_decision: Callable,
):
    graph = StateGraph(GraphState)
    graph.add_node("check_recommendation_requirements", check_recommendation_requirements)
    graph.add_node("recommendation_followup", recommendation_followup)
    graph.add_node("build_algorithm_plan_request", build_algorithm_plan_request)
    graph.add_node("algorithm_plan_api_tool", algorithm_plan_api_tool)
    graph.add_node("normalize_algorithm_plan", normalize_algorithm_plan)
    graph.add_node("present_recommendation", present_recommendation)

    graph.set_entry_point("check_recommendation_requirements")
    graph.add_conditional_edges(
        "check_recommendation_requirements",
        recommendation_decision,
        {
            "recommendation_followup": "recommendation_followup",
            "build_algorithm_plan_request": "build_algorithm_plan_request",
        },
    )
    graph.add_edge("recommendation_followup", END)
    graph.add_edge("build_algorithm_plan_request", "algorithm_plan_api_tool")
    graph.add_edge("algorithm_plan_api_tool", "normalize_algorithm_plan")
    graph.add_edge("normalize_algorithm_plan", "present_recommendation")
    graph.add_edge("present_recommendation", END)
    return graph.compile()

