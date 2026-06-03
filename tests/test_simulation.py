from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("langgraph")

from agent_solution.cli import import_seed_documents
from agent_solution.config import Settings
from agent_solution.graph import append_user_message, build_graph, initial_state
from agent_solution.kb import KnowledgeBase
from agent_solution.simulation import SMOKING_DEMO_MESSAGES, SimulatedLLMClient
from agent_solution.tools import build_default_registry


def test_simulated_chat_returns_valid_slot_json() -> None:
    llm = SimulatedLLMClient()

    response = llm.chat(
        [
            {"role": "system", "content": "请从对话中抽取算法方案槽位，只返回 JSON。"},
            {"role": "user", "content": "我想做一个抽烟识别算法，输入是摄像头视频流。"},
        ]
    )

    data = json.loads(response)
    assert data["problem_goal"] == "识别视频或图像中的抽烟行为"
    assert data["inputs"] == "园区固定摄像头视频流或图片序列"


def test_simulated_embeddings_are_stable() -> None:
    llm = SimulatedLLMClient()

    first = llm.embed(["抽烟识别 手口动作 烟雾误报"])
    second = llm.embed(["抽烟识别 手口动作 烟雾误报"])

    assert first == second
    assert first[0][0] > 0


def test_simulated_embedding_retrieves_smoking_seed(tmp_path: Path) -> None:
    llm = SimulatedLLMClient()
    settings = Settings.from_env()
    settings = Settings(
        db_path=tmp_path / "kb.sqlite",
        model_base_url=settings.model_base_url,
        chat_model=settings.chat_model,
        embed_model=settings.embed_model,
        api_key=settings.api_key,
    )
    kb = KnowledgeBase.from_settings(settings, embedder=llm)
    import_seed_documents(settings, kb)

    hits = kb.search("抽烟识别 手口动作 烟雾误报", mode="hybrid", top_k=3)

    assert hits
    assert hits[0].source == "seed:smoking_detection.md"


def test_smoking_demo_reaches_confirmed_state(tmp_path: Path) -> None:
    llm = SimulatedLLMClient()
    kb = KnowledgeBase(tmp_path / "kb.sqlite", embedder=llm)
    kb.add_document("seed:smoking_detection.md", "smoking", "抽烟识别需要手口动作、烟雾误报过滤和多帧融合")
    graph = build_graph(llm, kb, build_default_registry())
    state = initial_state()

    for user_text in SMOKING_DEMO_MESSAGES:
        state = append_user_message(state, user_text)
        state = graph.invoke(state)

    assert state["status"] == "confirmed"
    assert "方案已确认" in state["assistant_reply"]
    assert "抽烟" in state["draft_summary_markdown"]
    assert state["recommended_solution"]["name"]


def test_simulated_first_turn_returns_case_summary(tmp_path: Path) -> None:
    llm = SimulatedLLMClient()
    kb = KnowledgeBase(tmp_path / "kb.sqlite", embedder=llm)
    kb.add_document("seed:smoking_detection.md", "smoking", "抽烟识别需要手口动作、烟雾误报过滤和多帧融合")
    graph = build_graph(llm, kb, build_default_registry())

    state = append_user_message(initial_state(), "我想做一个抽烟识别算法")
    result = graph.invoke(state)

    assert result["stage"] == "kb_satisfaction_check"
    assert result["kb_cases"]
    assert "是否符合你的需求" in result["assistant_reply"]


def test_followup_is_single_question_with_acknowledgement() -> None:
    llm = SimulatedLLMClient()

    response = llm.chat(
        [
            {"role": "system", "content": "请先简短确认已记录内容，再只提出 1 个最关键的追问。"},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "known_slots": {"problem_goal": "识别视频或图像中的抽烟行为"},
                        "missing_required_slots": ["inputs", "outputs", "core_steps"],
                        "optional_missing_slots": ["assumptions", "edge_cases"],
                        "knowledge": [],
                        "latest_user_message": "我想做一个抽烟识别算法",
                    },
                    ensure_ascii=False,
                ),
            },
        ]
    )

    assert "我先记为" in response or "我已经记录" in response
    assert response.count("接下来我还需要确认一个关键点：") == 1
