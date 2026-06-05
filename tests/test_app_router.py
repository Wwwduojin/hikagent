from __future__ import annotations

from agent_solution.app_router import heuristic_intent
from agent_solution.models import AppSession


def current_session(stage: str = "solution_refinement", status: str = "collecting") -> AppSession:
    return AppSession(
        session_id="s1",
        user_id="u_001",
        thread_id="u_001__smoking_detection__001",
        app_id="smoking_detection",
        app_name="抽烟识别",
        stage=stage,
        status=status,
    )


def test_intent_recognizes_same_and_new_app() -> None:
    same = heuristic_intent("这个需要多少标注数据？", current_session())
    new = heuristic_intent("那安全帽识别呢？", current_session())

    assert same.switch_type == "same_app"
    assert new.switch_type == "new_app"
    assert new.detected_app_id == "helmet_detection"


def test_intent_recognizes_compare_general_feedback_and_confirmation() -> None:
    compare = heuristic_intent("抽烟识别和明火烟雾检测有什么区别？", current_session())
    general = heuristic_intent("园区安防有什么算法？", None)
    feedback = heuristic_intent("这个案例不满意", current_session(stage="kb_satisfaction_check"))
    confirmation = heuristic_intent("确认，就按这个方案", current_session(status="reviewing"))

    assert compare.switch_type == "compare_apps"
    assert len(compare.detected_apps) == 2
    assert general.switch_type == "general_recommendation"
    assert feedback.kb_satisfaction == "unsatisfied"
    assert confirmation.is_solution_confirmation is True
