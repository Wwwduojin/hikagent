from __future__ import annotations

import pytest

pytest.importorskip("langgraph")

from agent_solution.graph import _coerce_message, append_user_message, build_graph, initial_state
from agent_solution.kb import KnowledgeBase
from agent_solution.tools import build_default_registry


class FakeLLM:
    def chat(self, messages, temperature=0.2):
        if "知识库案例检索 Agent" in messages[0]["content"]:
            return """
            {
              "cases": [
                {
                  "title": "抽烟识别历史案例",
                  "scenario": "固定摄像头视频监控",
                  "core_flow": "人体和手部检测、手口关系判断、多帧融合",
                  "match_reason": "与当前抽烟识别需求高度相关",
                  "source": "seed:smoking_detection.md"
                }
              ]
            }
            """
        if "知识库案例反馈 Agent" in messages[0]["content"]:
            return "我找到了抽烟识别历史案例。这些历史案例是否符合你的需求？请回复满意或不满意。"
        if "当前的任务不是立即推荐新方案" in messages[0]["content"]:
            return "我已经记录了你对历史案例不满意。\n\n输入数据是什么？请说明数据类型和采集场景。"
        if "算法方案推荐 Agent" in messages[0]["content"]:
            return """
            {
              "name": "多阶段抽烟识别方案",
              "fit_reason": "适合固定摄像头下的实时低误报识别",
              "core_approach": "人体手部检测、手口关系判断、香烟烟雾辅助和多帧融合",
              "preconditions": "需要摄像头视频流",
              "open_questions": ["实时性目标", "验收指标"]
            }
            """
        if "只返回 JSON" in messages[0]["content"]:
            return """
            {
              "problem_goal": "识别视频中的抽烟行为",
              "inputs": "摄像头视频流",
              "outputs": "抽烟事件、人员位置、证据帧和置信度",
              "core_steps": "人体/手部/香烟检测，人员跟踪，手口动作判断，多帧融合",
              "constraints": "实时、低误报、处理遮挡和低光照",
              "assumptions": "摄像头固定安装",
              "metrics": "precision、recall、F1、误报率",
              "edge_cases": "喝水、打电话、摸脸"
            }
            """
        if "只提出 1 个最关键的追问" in messages[0]["content"]:
            return "我已经记录了你刚刚补充的关键信息。\n\n接下来我还需要确认一个关键点：输入数据是什么？"
        return "# 算法方案总结\n\n## 目标\n识别视频中的抽烟行为"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.1] for _ in texts]


def test_first_turn_presents_knowledge_cases(tmp_path):
    llm = FakeLLM()
    kb = KnowledgeBase(tmp_path / "kb.sqlite", embedder=llm)
    kb.add_document("seed:smoking_detection.md", "smoking", "抽烟识别需要手口动作和烟雾误报过滤")
    graph = build_graph(llm, kb, build_default_registry())

    state = append_user_message(initial_state(), "我想做一个抽烟识别算法")
    result = graph.invoke(state)

    assert result["stage"] == "kb_satisfaction_check"
    assert result["active_agent"] == "knowledge_base_agent"
    assert result["kb_has_match"] is True
    assert result["kb_cases"][0]["title"] == "抽烟识别历史案例"
    assert "是否符合你的需求" in result["assistant_reply"]


def test_unsatisfied_feedback_routes_to_recommendation_agent(tmp_path):
    class IncompleteRequirementLLM(FakeLLM):
        def chat(self, messages, temperature=0.2):
            if "只返回 JSON" in messages[0]["content"]:
                return """
                {
                  "problem_goal": "识别视频中的抽烟行为",
                  "inputs": null,
                  "outputs": null,
                  "core_steps": null,
                  "constraints": null,
                  "assumptions": null,
                  "metrics": null,
                  "edge_cases": null
                }
                """
            return super().chat(messages, temperature=temperature)

    llm = IncompleteRequirementLLM()
    kb = KnowledgeBase(tmp_path / "kb.sqlite", embedder=llm)
    kb.add_document("seed:smoking_detection.md", "smoking", "抽烟识别需要手口动作和烟雾误报过滤")
    graph = build_graph(llm, kb, build_default_registry())

    state = append_user_message(initial_state(), "我想做一个抽烟识别算法")
    state = graph.invoke(state)
    state = append_user_message(state, "这些案例不满意，请推荐一个更合适的方案")
    result = graph.invoke(state)

    assert result["stage"] == "recommendation"
    assert result["active_agent"] == "recommendation_agent"
    assert result["recommended_solution"] is None
    assert "输入数据是什么" in result["assistant_reply"]


def test_no_knowledge_match_routes_to_recommendation_agent(tmp_path):
    class IncompleteRequirementLLM(FakeLLM):
        def chat(self, messages, temperature=0.2):
            if "只返回 JSON" in messages[0]["content"]:
                return """
                {
                  "problem_goal": "识别视频中的抽烟行为",
                  "inputs": null,
                  "outputs": null,
                  "core_steps": null,
                  "constraints": null,
                  "assumptions": null,
                  "metrics": null,
                  "edge_cases": null
                }
                """
            return super().chat(messages, temperature=temperature)

    llm = IncompleteRequirementLLM()
    kb = KnowledgeBase(tmp_path / "kb.sqlite", embedder=llm)
    graph = build_graph(llm, kb, build_default_registry())

    state = append_user_message(initial_state(), "我想做一个抽烟识别算法")
    result = graph.invoke(state)

    assert result["kb_has_match"] is False
    assert result["active_agent"] == "recommendation_agent"
    assert result["stage"] == "recommendation"
    assert result["recommended_solution"] is None


def test_satisfied_feedback_generates_review_summary(tmp_path):
    llm = FakeLLM()
    kb = KnowledgeBase(tmp_path / "kb.sqlite", embedder=llm)
    kb.add_document("seed:smoking_detection.md", "smoking", "抽烟识别需要手口动作和烟雾误报过滤")
    graph = build_graph(llm, kb, build_default_registry())

    state = append_user_message(initial_state(), "我想做一个抽烟识别算法")
    state = graph.invoke(state)
    state = append_user_message(state, "满意，可以参考这个案例")
    result = graph.invoke(state)

    assert result["status"] == "reviewing"
    assert result["stage"] == "reviewing"
    assert "算法方案总结" in result["assistant_reply"]
    assert result["draft_solution_json"]["tool_estimates"]["estimate_complexity"]["score"] >= 1


def test_second_turn_updates_inputs_and_outputs(tmp_path):
    class IncrementalLLM(FakeLLM):
        def chat(self, messages, temperature=0.2):
            if "只返回 JSON" in messages[0]["content"]:
                transcript = messages[-1]["content"]
                if "输入是摄像头视频流" in transcript:
                    return """
                    {
                      "problem_goal": null,
                      "inputs": "摄像头视频流",
                      "outputs": "抽烟事件和证据帧",
                      "core_steps": null,
                      "constraints": null,
                      "assumptions": null,
                      "metrics": null,
                      "edge_cases": null
                    }
                    """
                return """
                {
                  "problem_goal": "识别视频中的抽烟行为",
                  "inputs": null,
                  "outputs": null,
                  "core_steps": null,
                  "constraints": null,
                  "assumptions": null,
                  "metrics": null,
                  "edge_cases": null
                }
                """
            if "只提出 1 个最关键的追问" in messages[0]["content"]:
                return "我已经记录了算法目标。\n\n接下来我还需要确认一个关键点：输入数据是什么？"
            return super().chat(messages, temperature=temperature)

    llm = IncrementalLLM()
    kb = KnowledgeBase(tmp_path / "kb.sqlite", embedder=llm)
    kb.add_document("seed:smoking_detection.md", "smoking", "抽烟识别需要手口动作和烟雾误报过滤")
    graph = build_graph(llm, kb, build_default_registry())

    state = append_user_message(initial_state(), "我想做一个抽烟识别算法")
    state = graph.invoke(state)
    state = append_user_message(state, "输入是摄像头视频流，输出要有抽烟事件和证据帧")
    result = graph.invoke(state)

    assert result["slots"]["inputs"] == "摄像头视频流"
    assert "抽烟事件" in result["slots"]["outputs"]


def test_studio_text_blocks_and_explicit_requirements_are_preserved(tmp_path):
    class ExplicitAnswerLLM(FakeLLM):
        def chat(self, messages, temperature=0.2):
            if "只返回 JSON" in messages[0]["content"]:
                return """
                {
                  "problem_goal": null,
                  "inputs": null,
                  "outputs": null,
                  "core_steps": null,
                  "constraints": null,
                  "assumptions": null,
                  "metrics": null,
                  "edge_cases": null
                }
                """
            return super().chat(messages, temperature=temperature)

    llm = ExplicitAnswerLLM()
    kb = KnowledgeBase(tmp_path / "kb.sqlite", embedder=llm)
    graph = build_graph(llm, kb, build_default_registry())
    state = initial_state()

    for user_text in [
        "我想做一个抽烟识别算法",
        "输入数据是单张图片",
        "输出是否有人在抽烟，给出判断的证据",
        "每帧<50ms，部署在GPU服务器上",
        "Accuracy大于80%，漏报率≤5%，误报率≤10%",
    ]:
        state = append_user_message(state, user_text)
        state = graph.invoke(state)

    assert state["slots"]["inputs"] == "单张图片"
    assert "每帧<50ms" in state["slots"]["constraints"]
    assert "GPU服务器" in state["slots"]["constraints"]
    assert "Accuracy大于80%" in state["slots"]["metrics"]
    assert "漏报率≤5%" in state["slots"]["metrics"]

    coerced = _coerce_message({"type": "human", "content": [{"type": "text", "text": "单张图片"}]})
    assert coerced["content"] == "单张图片"
