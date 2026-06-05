from __future__ import annotations

import os

from agent_solution.cli import import_seed_documents
from agent_solution.algorithm_planner import build_algorithm_planner
from agent_solution.config import Settings
from agent_solution.graph import build_chat_graph, build_graph, build_intent_router_graph
from agent_solution.kb import KnowledgeBase
from agent_solution.llm import OpenAICompatibleClient
from agent_solution.simulation import SimulatedLLMClient
from agent_solution.tools import build_default_registry


def _build_runtime():
    settings = Settings.from_env()
    simulate = os.getenv("AGENT_SOLUTION_SIMULATE", "true").lower() not in {"0", "false", "no"}
    llm = SimulatedLLMClient() if simulate else OpenAICompatibleClient(settings)
    kb = KnowledgeBase.from_settings(settings, embedder=llm)
    import_seed_documents(settings, kb)
    tools = build_default_registry()
    planner = build_algorithm_planner(settings.plan_mode, llm)
    return llm, kb, tools, planner


_llm, _kb, _tools, _planner = _build_runtime()

graph = build_graph(_llm, _kb, _tools, planner=_planner)
chat_graph = build_chat_graph(_llm, _kb, _tools, planner=_planner)
intent_router_graph = build_intent_router_graph(_llm)
