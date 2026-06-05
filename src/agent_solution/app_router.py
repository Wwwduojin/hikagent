from __future__ import annotations

import json
import re

from agent_solution.llm import OpenAICompatibleClient, parse_json_object
from agent_solution.models import AppSession, IntentResult, UserProfile
from agent_solution.prompts import load_prompt


APP_CATALOG = {
    "smoking_detection": "抽烟识别",
    "helmet_detection": "安全帽识别",
    "fall_detection": "跌倒检测",
    "intrusion_detection": "区域入侵检测",
    "fire_smoke_detection": "明火烟雾检测",
}

APP_ALIASES = {
    "smoking_detection": ["抽烟", "吸烟"],
    "helmet_detection": ["安全帽", "头盔"],
    "fall_detection": ["跌倒", "摔倒"],
    "intrusion_detection": ["入侵", "越线"],
    "fire_smoke_detection": ["明火", "火焰", "烟雾检测", "火灾"],
}


class IntentRecognitionAgent:
    def __init__(self, llm: OpenAICompatibleClient):
        self.llm = llm

    def predict(
        self,
        user_input: str,
        current_session: AppSession | None = None,
        summary: str = "",
    ) -> IntentResult:
        fallback = heuristic_intent(user_input, current_session)
        payload = {
            "current_app_id": current_session.app_id if current_session else None,
            "current_app_name": current_session.app_name if current_session else None,
            "stage": current_session.stage if current_session else None,
            "status": current_session.status if current_session else None,
            "summary": summary,
            "user_input": user_input,
        }
        try:
            data = parse_json_object(
                self.llm.chat(
                    [
                        {"role": "system", "content": load_prompt("intent_recognition.md")},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                    temperature=0.0,
                )
            )
            result = IntentResult.model_validate(data)
        except Exception:
            return fallback
        if fallback.confidence >= 0.9:
            return fallback
        return result


def detect_apps(text: str) -> list[dict[str, str]]:
    matches = []
    for app_id, aliases in APP_ALIASES.items():
        if any(alias in text for alias in aliases):
            matches.append({"app_id": app_id, "app_name": APP_CATALOG[app_id]})
    return matches


def heuristic_intent(user_input: str, current_session: AppSession | None = None) -> IntentResult:
    text = user_input.strip()
    normalized = text.lower()
    apps = detect_apps(text)
    stage = current_session.stage if current_session else None
    status = current_session.status if current_session else None

    if stage == "kb_satisfaction_check":
        if any(term in normalized for term in ["不满意", "不符合", "不合适", "换一个", "不够"]):
            return IntentResult(
                intent_type="kb_feedback",
                switch_type="same_app",
                confidence=0.99,
                is_kb_satisfaction_feedback=True,
                kb_satisfaction="unsatisfied",
                reason="当前处于知识库满意度确认阶段，用户明确表示不满意。",
            )
        if any(term in normalized for term in ["满意", "符合", "合适", "就这个", "可以"]):
            return IntentResult(
                intent_type="kb_feedback",
                switch_type="same_app",
                confidence=0.99,
                is_kb_satisfaction_feedback=True,
                kb_satisfaction="satisfied",
                reason="当前处于知识库满意度确认阶段，用户明确表示满意。",
            )

    if status == "reviewing" and any(term in normalized for term in ["确认", "同意", "就按这个", "可以落方案"]):
        return IntentResult(
            intent_type="solution_confirmation",
            switch_type="same_app",
            confidence=0.99,
            is_solution_confirmation=True,
            reason="当前方案处于 reviewing，用户明确确认方案。",
        )

    compare_terms = ["区别", "对比", "比较", "优劣", "哪个更适合", "哪个好"]
    if len(apps) >= 2 and any(term in text for term in compare_terms):
        return IntentResult(
            intent_type="compare_apps",
            switch_type="compare_apps",
            detected_apps=apps,
            confidence=0.98,
            reason="用户同时提到多个算法应用并询问差异或选择建议。",
        )

    if apps:
        app = apps[0]
        switch = "same_app" if current_session and current_session.app_id == app["app_id"] else "new_app"
        return IntentResult(
            intent_type="single_app_question",
            switch_type=switch,
            detected_app_id=app["app_id"],
            detected_app_name=app["app_name"],
            detected_apps=apps,
            confidence=0.96,
            reason="用户明确提到算法应用名称。",
        )

    if current_session and any(term in text for term in ["这个", "它", "该方案", "刚才", "继续", "还需要"]):
        return IntentResult(
            intent_type="single_app_question",
            switch_type="same_app",
            detected_app_id=current_session.app_id,
            detected_app_name=current_session.app_name,
            confidence=0.9,
            reason="用户使用指代词继续当前算法应用。",
        )

    general_terms = ["园区安防", "有什么算法", "推荐算法", "适合的算法", "业务场景"]
    if any(term in text for term in general_terms):
        return IntentResult(
            intent_type="general_recommendation",
            switch_type="general_recommendation",
            confidence=0.9,
            reason="用户描述业务场景或请求泛推荐，但未指定单个算法应用。",
        )

    if current_session:
        return IntentResult(
            intent_type="single_app_question",
            switch_type="same_app",
            detected_app_id=current_session.app_id,
            detected_app_name=current_session.app_name,
            confidence=0.65,
            reason="未识别到新应用，默认继续当前会话。",
        )
    return IntentResult(
        intent_type="ambiguous",
        switch_type="ambiguous",
        confidence=0.4,
        reason="没有当前应用，也无法从输入中识别明确算法应用或泛推荐场景。",
    )


def extract_user_profile(text: str) -> UserProfile:
    profile = UserProfile()
    if "私有化" in text or "本地部署" in text:
        profile.deployment_preference = "私有化部署"
    elif "云端" in text or "云部署" in text:
        profile.deployment_preference = "云端部署"
    if any(term in text for term in ["实时", "准实时", "延迟", "ms", "毫秒"]):
        match = re.search(r"(?:<|≤|小于|不超过)?\s*\d+\s*(?:ms|毫秒)", text, re.IGNORECASE)
        profile.latency_requirement = match.group(0) if match else "关注实时或准实时处理"
    if "预算" in text:
        if "低" in text:
            profile.budget_level = "低"
        elif "高" in text:
            profile.budget_level = "高"
        else:
            profile.budget_level = "中等"
    if any(term in text for term in ["合规", "隐私", "数据不出域"]):
        profile.compliance_requirement = "关注数据合规与隐私"
    metrics = [term for term in ["precision", "recall", "F1", "准确率", "误报率", "漏报率"] if term in text]
    if metrics:
        profile.acceptance_preference = "、".join(metrics)
    return profile
