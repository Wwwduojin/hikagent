from __future__ import annotations

from pathlib import Path

from agent_solution.algorithm_planner import SimulatedAlgorithmPlanner
from agent_solution.kb import KnowledgeBase
from agent_solution.service import AgentSolutionService, create_sqlite_checkpointer
from agent_solution.simulation import SimulatedLLMClient
from agent_solution.storage import BusinessStore
from agent_solution.tools import build_default_registry


def build_service(tmp_path: Path) -> AgentSolutionService:
    llm = SimulatedLLMClient()
    db_path = tmp_path / "agent.sqlite"
    kb = KnowledgeBase(db_path, embedder=llm)
    kb.add_document("seed:smoking_detection.md", "smoking", "抽烟识别需要手口动作、烟雾误报过滤和多帧融合")
    kb.add_document("seed:helmet_detection.md", "helmet", "安全帽识别需要头部检测和佩戴判别")
    store = BusinessStore(db_path)
    return AgentSolutionService(
        llm,
        kb,
        build_default_registry(),
        store,
        SimulatedAlgorithmPlanner(),
        checkpointer=create_sqlite_checkpointer(str(tmp_path / "checkpoints.sqlite")),
    )


def test_service_reuses_same_app_thread_and_isolates_new_app(tmp_path: Path) -> None:
    service = build_service(tmp_path)

    smoking = service.chat("u_001", "我想做一个抽烟识别算法")
    same = service.chat("u_001", "这个需要多少标注数据？", smoking.thread_id)
    helmet = service.chat("u_001", "那安全帽识别呢？", same.thread_id)

    assert same.thread_id == smoking.thread_id
    assert helmet.thread_id != smoking.thread_id
    assert helmet.current_app_id == "helmet_detection"
    assert len(service.list_sessions("u_001")) == 2


def test_service_does_not_reuse_another_users_thread(tmp_path: Path) -> None:
    service = build_service(tmp_path)
    first = service.chat("u_001", "我想做一个抽烟识别算法")

    second = service.chat("u_002", "我想做一个抽烟识别算法", first.thread_id)

    assert second.thread_id != first.thread_id
    assert service.get_session("u_002", first.thread_id) is None


def test_service_clarifies_compare_and_general_requests_without_creating_sessions(tmp_path: Path) -> None:
    service = build_service(tmp_path)

    compare = service.chat("u_001", "抽烟识别和安全帽识别有什么区别？")
    general = service.chat("u_001", "园区安防有什么算法？")

    assert compare.need_clarification is True
    assert general.need_clarification is True
    assert "单个算法应用" in compare.message
    assert service.list_sessions("u_001") == []


def test_user_profile_is_shared_without_copying_app_slots(tmp_path: Path) -> None:
    service = build_service(tmp_path)

    smoking = service.chat("u_001", "我想做一个抽烟识别算法，需要私有化部署和准实时处理")
    helmet = service.chat("u_001", "那安全帽识别呢？", smoking.thread_id)
    profile = service.store.get_user_profile("u_001")
    helmet_state = service.store.load_state(service.get_session("u_001", helmet.thread_id).session_id)

    assert profile.deployment_preference == "私有化部署"
    assert profile.latency_requirement
    assert helmet_state["slots"]["problem_goal"] == "识别人员是否佩戴安全帽"
    assert "抽烟" not in str(helmet_state["slots"])


def test_service_restores_sqlite_thread_after_rebuild(tmp_path: Path) -> None:
    first_service = build_service(tmp_path)
    first = first_service.chat("u_001", "我想做一个抽烟识别算法")

    second_service = build_service(tmp_path)
    resumed = second_service.chat("u_001", "这个需要多少标注数据？", first.thread_id)

    assert resumed.thread_id == first.thread_id
    assert len(second_service.list_sessions("u_001")) == 1


def test_long_conversation_creates_rolling_summary(tmp_path: Path) -> None:
    service = build_service(tmp_path)
    response = service.chat("u_001", "我想做一个抽烟识别算法")

    for index in range(10):
        response = service.chat("u_001", f"这个方案继续补充第 {index} 条信息", response.thread_id)

    session = service.get_session("u_001", response.thread_id)
    state = service.store.load_state(session.session_id)

    assert state["summary"]
    assert len(state["messages"]) <= 16


def test_recommendation_flow_persists_structured_algorithm_plan(tmp_path: Path) -> None:
    service = build_service(tmp_path)
    response = service.chat("u_001", "我想做一个抽烟识别算法")
    response = service.chat("u_001", "这些案例不满意", response.thread_id)
    response = service.chat(
        "u_001",
        "输入是园区固定摄像头视频流，输出抽烟事件、人员框、证据帧和置信度",
        response.thread_id,
    )
    response = service.chat("u_001", "要求实时、低误报和私有化部署", response.thread_id)
    response = service.chat("u_001", "指标看 precision、recall、F1 和误报率", response.thread_id)
    session = service.get_session("u_001", response.thread_id)
    state = service.store.load_state(session.session_id)

    assert state["algorithm_plan_request"]
    assert state["algorithm_plan_raw_response"]
    assert state["algorithm_plan_normalized"]["pipeline"]
    assert state["recommended_solution"]["plan"]["solution_name"]
