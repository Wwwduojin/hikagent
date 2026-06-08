from __future__ import annotations

import hashlib
import json
from typing import Any


SMOKING_DEMO_MESSAGES = [
    "我想做一个抽烟识别算法",
    "这些案例还不够贴合，请推荐一个更合适的方案",
    "输入是园区固定摄像头视频流，需要输出抽烟事件、人员框、证据帧和置信度",
    "要求实时、低误报，重点处理喝水、打电话、摸脸、低光照和遮挡",
    "评估指标看 precision、recall、F1、误报率、漏报率和事件级准确率",
    "确认",
    "确认",
]


class SimulatedLLMClient:
    """Deterministic local client used when the real model service is unavailable."""

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.2) -> str:
        system = messages[0]["content"] if messages else ""
        user = messages[-1]["content"] if messages else ""
        if "知识库案例检索 Agent" in system:
            return json.dumps(_knowledge_cases(user), ensure_ascii=False, indent=2)
        if "知识库案例反馈 Agent" in system:
            return _present_cases(user)
        if "当前的任务不是立即推荐新方案" in system:
            return _recommendation_followup(user)
        if "算法方案推荐 Agent" in system:
            return json.dumps(_recommendation(user), ensure_ascii=False, indent=2)
        if "只返回 JSON" in system:
            return json.dumps(_extract_slots(user), ensure_ascii=False, indent=2)
        if "只提出 1 个最关键的追问" in system:
            return _followup(user)
        if "输出一份中文 Markdown 算法方案总结" in system:
            return _summary(user)
        return "我会基于当前信息继续确认算法方案。"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [_keyword_vector(text) for text in texts]

    def describe_images(self, images: list[dict[str, Any]], user_text: str = "") -> str:
        names = "、".join(str(image.get("name") or "未命名图片") for image in images) or "图片"
        return (
            f"已解析图片：{names}。图片看起来是园区固定摄像头场景，"
            "需要关注人员抽烟行为、手口动作、香烟/烟雾线索、证据帧和置信度输出。"
        )


def _extract_slots(transcript: str) -> dict[str, str | None]:
    try:
        payload = json.loads(transcript)
        existing = payload.get("existing_slots", {})
        latest = payload.get("latest_user_message", "")
        history = " ".join(payload.get("user_message_history", []))
    except json.JSONDecodeError:
        existing = {}
        latest = transcript
        history = transcript
    data: dict[str, str | None] = {
        "problem_goal": None,
        "inputs": None,
        "outputs": None,
        "core_steps": None,
        "constraints": None,
        "assumptions": None,
        "metrics": None,
        "edge_cases": None,
    }
    data.update({key: value for key, value in existing.items() if key in data and value})
    transcript = f"{history} {latest}"
    if any(word in transcript for word in ["抽烟", "吸烟"]):
        data["problem_goal"] = "识别视频或图像中的抽烟行为"
    if any(word in latest for word in ["单张图片", "单张图像"]):
        data["inputs"] = "单张图片"
    elif any(word in latest for word in ["视频流", "摄像头", "图片序列", "图像序列"]):
        data["inputs"] = "园区固定摄像头视频流或图片序列"
    if any(word in latest for word in ["输出", "事件", "证据帧", "置信度", "人员框", "判断的证据"]):
        data["outputs"] = "抽烟事件、人员框、证据帧、发生时间段和置信度"
    if any(word in latest for word in ["手口", "手部", "香烟", "烟雾", "多帧", "流程"]):
        data["core_steps"] = "人体/手部/香烟候选检测，人员跟踪，手-口动作关系判断，烟雾视觉特征辅助，多帧置信度融合"
    if any(word in latest for word in ["实时", "低误报", "低光照", "遮挡", "边缘", "延迟", "GPU", "不能漏报"]):
        data["constraints"] = latest
    if any(word in transcript for word in ["固定摄像头", "园区"]):
        data["assumptions"] = "摄像头固定安装，画面覆盖园区重点区域，允许配置检测阈值和告警区域"
    if any(word in latest for word in ["precision", "recall", "F1", "误报率", "漏报率", "准确率", "Accuracy", "accuracy"]):
        data["metrics"] = latest
    if any(word in transcript for word in ["喝水", "打电话", "摸脸", "遮挡", "低光照"]):
        data["edge_cases"] = "喝水、吃东西、打电话、摸脸、低光照、遮挡、多人重叠和背景烟雾"
    return data


def _knowledge_cases(user_payload: str) -> dict[str, list[dict[str, str]]]:
    try:
        payload = json.loads(user_payload)
        hits = payload.get("hits", [])
    except json.JSONDecodeError:
        hits = []
    cases = []
    for hit in hits[:3]:
        source = hit.get("source", "unknown")
        title = source.removeprefix("seed:").removesuffix(".md").replace("_", " ")
        cases.append(
            {
                "title": title,
                "scenario": "视频监控场景中的相关行为识别",
                "core_flow": "目标检测、人员跟踪、关键动作关系判断和多帧置信度融合",
                "match_reason": "该历史案例与当前需求在业务目标和视觉识别流程上高度相关。",
                "source": source,
            }
        )
    return {"cases": cases}


def _present_cases(user_payload: str) -> str:
    try:
        payload = json.loads(user_payload)
        cases = payload.get("cases", [])
    except json.JSONDecodeError:
        cases = []
    sections = ["我从知识库中找到了以下相关历史案例："]
    for index, case in enumerate(cases, 1):
        sections.append(
            f"### {index}. {case.get('title', '历史案例')}\n"
            f"- 适用场景：{case.get('scenario', '待确认')}\n"
            f"- 核心流程：{case.get('core_flow', '待确认')}\n"
            f"- 匹配原因：{case.get('match_reason', '与当前需求相关')}"
        )
    sections.append("这些历史案例是否符合你的需求？你可以回复“满意”“不满意”，或继续补充需求。")
    return "\n\n".join(sections)


def _recommendation(user_payload: str) -> dict[str, Any]:
    return {
        "name": "基于目标检测、手口关系建模与多帧融合的抽烟识别方案",
        "fit_reason": "该方案同时利用人体动作、香烟候选和烟雾辅助信息，适合固定摄像头下兼顾实时性与低误报。",
        "core_approach": "检测人体、手部和香烟候选区域，持续跟踪人员，建模手-口动作关系，并结合烟雾特征和多帧置信度融合输出抽烟事件。",
        "preconditions": "需要固定摄像头视频流、合理的人员成像尺寸，并允许配置检测阈值和告警区域。",
        "open_questions": ["输入视频场景和摄像头位置", "输出事件字段", "实时性和误报约束", "验收指标"],
    }


def _recommendation_followup(user_payload: str) -> str:
    try:
        payload = json.loads(user_payload)
        known_slots = payload.get("known_slots", {})
        missing = payload.get("missing_recommendation_requirements", [])
    except json.JSONDecodeError:
        known_slots = {}
        missing = []
    intro = "我已经记录了你对历史案例不满意，接下来由方案推荐 Agent 进一步确认需求。"
    if known_slots.get("problem_goal"):
        intro = f"目标我先记为：{known_slots['problem_goal']}。接下来由方案推荐 Agent 进一步确认需求。"
    question_map = {
        "problem_goal": "请再说明算法要解决的具体业务问题，以及希望识别或判断什么目标？",
        "inputs": "输入数据是什么？请说明数据类型、采集场景，以及摄像头或传感器条件。",
        "outputs": "你希望算法输出哪些结果？例如事件、目标位置、证据帧、时间段或置信度。",
        "constraints": "有哪些关键约束？例如实时性、部署算力、误报漏报偏好、低光照或遮挡。",
        "metrics": "你希望用哪些指标验收方案？例如 precision、recall、F1、误报率或事件级准确率。",
    }
    question = next((question_map[key] for key in missing if key in question_map), "请继续补充关键业务需求。")
    return f"{intro}\n\n{question}"


def _followup(user_payload: str) -> str:
    try:
        payload = json.loads(user_payload)
        known_slots = payload.get("known_slots", {})
        missing = payload.get("missing_required_slots", [])
    except json.JSONDecodeError:
        known_slots = {}
        missing = []
    acknowledgements = []
    if known_slots.get("problem_goal"):
        acknowledgements.append(f"目标我先记为：{known_slots['problem_goal']}")
    if known_slots.get("inputs"):
        acknowledgements.append(f"输入我已经记录为：{known_slots['inputs']}")
    if known_slots.get("outputs"):
        acknowledgements.append(f"输出我已经记录为：{known_slots['outputs']}")
    intro = "；".join(acknowledgements[:2]) or "我已经记录了你刚刚补充的关键信息。"

    question_map = {
        "inputs": "输入数据是什么？例如园区固定摄像头视频流、图片序列，是否有摄像头区域配置？",
        "outputs": "希望输出哪些结果？例如抽烟事件、人员框、证据帧、时间段和置信度。",
        "core_steps": "核心流程是否需要包含人体/手部检测、手口动作判断、香烟或烟雾辅助、多帧融合？",
        "constraints": "部署约束是什么？例如实时、低误报、低光照、遮挡或边缘设备运行。",
        "metrics": "验收指标关注哪些？例如 precision、recall、F1、误报率、漏报率和事件级准确率。",
        "problem_goal": "算法要解决的具体目标是什么，最终希望识别或判断什么事件？",
    }
    next_question = next((question_map[key] for key in missing if key in question_map), "请确认是否可以基于当前信息生成第一版算法方案总结。")
    return f"{intro}。\n\n接下来我还需要确认一个关键点：{next_question}"


def _summary(user_payload: str) -> str:
    try:
        payload: dict[str, Any] = json.loads(user_payload)
    except json.JSONDecodeError:
        payload = {}
    slots = payload.get("slots", {})
    knowledge = payload.get("knowledge", [])
    recommended = payload.get("recommended_solution") or {}
    references = "\n".join(f"- {item.get('source', 'unknown')}" for item in knowledge[:3]) or "- 暂无"
    return f"""# 抽烟识别算法方案总结

## 目标
{slots.get("problem_goal") or "识别视频或图像中的抽烟行为"}。

## 输入输出
- 输入：{slots.get("inputs") or "园区固定摄像头视频流或图片序列"}。
- 输出：{slots.get("outputs") or "抽烟事件、人员框、证据帧、时间段和置信度"}。

## 核心流程
{slots.get("core_steps") or recommended.get("core_approach") or "人体/手部/香烟检测，人员跟踪，手口动作判断，烟雾辅助，多帧融合"}。

## 关键约束
{slots.get("constraints") or "实时处理、低误报、处理低光照和遮挡"}。

## 假设与边界
- 假设：{slots.get("assumptions") or "默认摄像头固定安装，检测区域和告警阈值可配置，后续可继续确认。"}
- 边界场景：{slots.get("edge_cases") or "建议重点关注遮挡、低光照、多人重叠、喝水、打电话、摸脸和背景烟雾等场景。"}

## 误报漏报风险
重点关注喝水、打电话、摸脸、吃东西、低光照、遮挡、小目标香烟和背景烟雾导致的误报或漏报。

## 评估指标
{slots.get("metrics") or "precision、recall、F1、误报率、漏报率和事件级准确率"}。

## 知识库引用
{references}

## 下一步确认项
确认摄像头安装角度、告警区域、实时性目标、事件触发阈值和证据帧保存策略。
"""


def _keyword_vector(text: str) -> list[float]:
    keywords = [
        ["抽烟", "吸烟", "手口", "香烟", "烟雾"],
        ["安全帽", "佩戴", "头部"],
        ["跌倒", "姿态", "静止"],
        ["入侵", "区域", "越线"],
        ["明火", "火焰", "烟雾"],
        ["实时", "低误报", "遮挡"],
    ]
    vector = [float(sum(1 for word in group if word in text)) for group in keywords]
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    vector.extend(byte / 255.0 for byte in digest[:4])
    return vector
