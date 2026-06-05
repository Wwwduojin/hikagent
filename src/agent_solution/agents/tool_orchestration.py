from __future__ import annotations

from collections.abc import Callable

from langgraph.graph import END, StateGraph

from agent_solution.graph_state import GraphState


def build_tool_orchestration_subgraph(
    *,
    tool_recall_node: Callable,
    planning_agent: Callable,
    topology_agent: Callable,
    parameter_agent: Callable,
    judge_agent: Callable,
):
    """Build the one-way tool orchestration chain after a solution is confirmed."""

    graph = StateGraph(GraphState)
    graph.add_node("tool_recall_node", tool_recall_node)
    graph.add_node("planning_agent", planning_agent)
    graph.add_node("topology_agent", topology_agent)
    graph.add_node("parameter_agent", parameter_agent)
    graph.add_node("judge_agent", judge_agent)

    graph.set_entry_point("tool_recall_node")
    graph.add_edge("tool_recall_node", "planning_agent")
    graph.add_edge("planning_agent", "topology_agent")
    graph.add_edge("topology_agent", "parameter_agent")
    graph.add_edge("parameter_agent", "judge_agent")
    graph.add_edge("judge_agent", END)
    return graph.compile()
