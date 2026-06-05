from __future__ import annotations

from agent_solution.algorithm_planner import LLMAlgorithmPlanner, SimulatedAlgorithmPlanner
from agent_solution.models import AlgorithmPlan, AlgorithmPlanRequest


def request() -> AlgorithmPlanRequest:
    return AlgorithmPlanRequest(
        algorithm_name="抽烟识别",
        algorithm_description="识别监控画面中的抽烟行为",
        input_data=["视频流"],
        output_target=["抽烟事件", "置信度"],
        constraints=["低误报"],
        metrics=["precision", "recall"],
        user_profile={"deployment_preference": "私有化部署"},
    )


def test_simulated_planner_returns_structured_plan() -> None:
    result = SimulatedAlgorithmPlanner().generate(request())
    plan = AlgorithmPlan.model_validate(result)

    assert "抽烟识别" in plan.solution_name
    assert plan.pipeline
    assert plan.deployment["mode"] == "私有化部署"


def test_llm_planner_falls_back_when_model_response_is_invalid() -> None:
    class BrokenLLM:
        def chat(self, messages, temperature=0.2):
            return "not json"

    result = LLMAlgorithmPlanner(BrokenLLM()).generate(request())

    assert AlgorithmPlan.model_validate(result).pipeline
