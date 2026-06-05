from __future__ import annotations

import json
from typing import Protocol

from agent_solution.llm import OpenAICompatibleClient, parse_json_object
from agent_solution.models import AlgorithmPlan, AlgorithmPlanRequest, AlgorithmPlanStep
from agent_solution.prompts import load_prompt


class AlgorithmPlanner(Protocol):
    def generate(self, request: AlgorithmPlanRequest) -> dict:
        ...


class SimulatedAlgorithmPlanner:
    def generate(self, request: AlgorithmPlanRequest) -> dict:
        name = request.algorithm_name or "算法应用"
        plan = AlgorithmPlan(
            solution_name=f"基于检测、细粒度判别与多帧融合的{name}方案",
            overall_idea="先定位关键目标与候选区域，再结合细粒度视觉线索和业务约束进行判别，最后通过时序或规则融合输出稳定结果。",
            pipeline=[
                AlgorithmPlanStep(
                    step_name="目标定位与候选区域生成",
                    description="检测人员、关键目标或业务区域，生成后续判别所需的候选区域。",
                    input=request.input_data,
                    output=["候选目标与区域"],
                ),
                AlgorithmPlanStep(
                    step_name="细粒度行为或属性判别",
                    description="结合目标关系、局部视觉特征和上下文信息生成候选结果。",
                    input=["候选目标与区域"],
                    output=["候选事件及置信度"],
                ),
                AlgorithmPlanStep(
                    step_name="误报抑制与结果融合",
                    description="结合负样本特征、业务规则和多帧结果降低误报与抖动。",
                    input=["候选事件及置信度"],
                    output=request.output_target,
                ),
            ],
            model_recommendation={
                "detector": "YOLO / RT-DETR",
                "classifier_or_vlm": "细粒度分类模型或视觉语言模型",
                "temporal_module": "滑窗投票或时序置信度融合",
            },
            data_requirement={
                "positive_samples": "覆盖典型目标、困难样本、远距离、小目标和低质量场景。",
                "negative_samples": "覆盖容易混淆的相似行为、背景干扰和业务误报场景。",
                "annotation": "建议标注目标框、事件标签及必要的关键区域。",
            },
            deployment={
                "mode": request.user_profile.get("deployment_preference") or "中心服务器或边缘推理",
                "latency": request.user_profile.get("latency_requirement") or "根据业务实时性要求评估",
            },
            evaluation={
                "main_metrics": request.metrics,
                "focus": "重点关注业务高频误报、漏报和困难场景。",
            },
            risks=[
                "小目标、遮挡和低清画面可能导致漏检。",
                "相似行为或背景干扰可能造成误报。",
                "需要通过真实业务数据持续校准阈值和规则。",
            ],
        )
        return plan.model_dump()


class LLMAlgorithmPlanner:
    def __init__(self, llm: OpenAICompatibleClient, fallback: AlgorithmPlanner | None = None):
        self.llm = llm
        self.fallback = fallback or SimulatedAlgorithmPlanner()

    def generate(self, request: AlgorithmPlanRequest) -> dict:
        try:
            response = self.llm.chat(
                [
                    {"role": "system", "content": load_prompt("algorithm_plan.md")},
                    {"role": "user", "content": json.dumps(request.model_dump(), ensure_ascii=False)},
                ],
                temperature=0.2,
            )
            return parse_json_object(response)
        except Exception:
            return self.fallback.generate(request)


def build_algorithm_planner(mode: str, llm: OpenAICompatibleClient) -> AlgorithmPlanner:
    if mode == "model":
        return LLMAlgorithmPlanner(llm)
    return SimulatedAlgorithmPlanner()
