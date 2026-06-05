from __future__ import annotations

import json
from typing import Any, Callable, Literal

from langgraph.graph import END, StateGraph

from agent_solution.agents.knowledge_base import build_knowledge_base_subgraph as _build_knowledge_base_subgraph
from agent_solution.agents.intent_recognition import build_intent_router_graph
from agent_solution.agents.recommendation import build_recommendation_subgraph as _build_recommendation_subgraph
from agent_solution.agents.supervisor import build_supervisor_subgraph as _build_supervisor_subgraph
from agent_solution.agents.tool_orchestration import (
    build_tool_orchestration_subgraph as _build_tool_orchestration_subgraph,
)
from agent_solution.kb import KnowledgeBase
from agent_solution.llm import OpenAICompatibleClient, parse_json_object
from agent_solution.algorithm_planner import AlgorithmPlanner, SimulatedAlgorithmPlanner
from agent_solution.graph_state import (
    GraphState,
    agent_state_from_graph,
    coerce_message,
    node_adapter,
)
from agent_solution.models import (
    AgentState,
    AlgorithmPlan,
    AlgorithmPlanRequest,
    AlgorithmSlots,
    ChatMessage,
    KnowledgeCase,
    RecommendedSolution,
)
from agent_solution.prompts import load_prompt
from agent_solution.tools import ToolRegistry


KB_MATCH_THRESHOLD = 0.20


def build_graph(
    llm: OpenAICompatibleClient,
    kb: KnowledgeBase,
    tools: ToolRegistry,
    planner: AlgorithmPlanner | None = None,
    checkpointer: Any | None = None,
):
    planner = planner or SimulatedAlgorithmPlanner()
    supervisor = build_supervisor_subgraph(llm, tools)
    knowledge_base = build_knowledge_base_subgraph(llm, kb)
    recommendation = build_recommendation_subgraph(llm, planner)
    tool_orchestration = build_tool_orchestration_subgraph()
    workflow = StateGraph(GraphState)
    workflow.add_node("supervisor_agent", supervisor)
    workflow.add_node("knowledge_base_agent", knowledge_base)
    workflow.add_node("recommendation_agent", recommendation)
    workflow.add_node("tool_orchestration_agent", tool_orchestration)

    workflow.set_entry_point("supervisor_agent")
    workflow.add_conditional_edges(
        "supervisor_agent",
        parent_agent_decision,
        {
            "knowledge_base_agent": "knowledge_base_agent",
            "recommendation_agent": "recommendation_agent",
            "tool_orchestration_agent": "tool_orchestration_agent",
            "end": END,
        },
    )
    workflow.add_conditional_edges(
        "knowledge_base_agent",
        parent_agent_decision,
        {
            "recommendation_agent": "recommendation_agent",
            "tool_orchestration_agent": "tool_orchestration_agent",
            "end": END,
        },
    )
    workflow.add_conditional_edges(
        "recommendation_agent",
        parent_agent_decision,
        {
            "tool_orchestration_agent": "tool_orchestration_agent",
            "end": END,
        },
    )
    workflow.add_conditional_edges(
        "tool_orchestration_agent",
        parent_agent_decision,
        {
            "end": END,
        },
    )
    return workflow.compile(checkpointer=checkpointer)


def build_chat_graph(
    llm: OpenAICompatibleClient,
    kb: KnowledgeBase,
    tools: ToolRegistry,
    planner: AlgorithmPlanner | None = None,
):
    return build_graph(llm, kb, tools, planner=planner)


def build_supervisor_subgraph(llm: OpenAICompatibleClient, tools: ToolRegistry):
    return _build_supervisor_subgraph(
        extract_slots=_node(extract_slots(llm)),
        supervisor_route=_node(supervisor_route()),
        handle_kb_feedback=_node(handle_kb_feedback()),
        ask_followup=_node(ask_followup(llm)),
        generate_summary=_node(generate_summary(llm)),
        tool_router=_node(tool_router(tools)),
        confirm_solution=_node(confirm_solution()),
        supervisor_decision=supervisor_decision,
        feedback_decision=feedback_decision,
        delegate_knowledge_base=_node(delegate_knowledge_base()),
        delegate_recommendation=_node(delegate_recommendation()),
    )


def build_knowledge_base_subgraph(llm: OpenAICompatibleClient, kb: KnowledgeBase):
    return _build_knowledge_base_subgraph(
        retrieve_knowledge=_node(retrieve_knowledge(kb)),
        summarize_knowledge_cases=_node(summarize_knowledge_cases(llm)),
        present_kb_cases=_node(present_kb_cases(llm)),
        knowledge_result_decision=knowledge_result_decision,
        delegate_recommendation=_node(delegate_recommendation()),
    )


def build_recommendation_subgraph(llm: OpenAICompatibleClient, planner: AlgorithmPlanner):
    return _build_recommendation_subgraph(
        check_recommendation_requirements=_node(check_recommendation_requirements()),
        recommendation_followup=_node(recommendation_followup(llm)),
        build_algorithm_plan_request=_node(build_algorithm_plan_request()),
        algorithm_plan_api_tool=_node(algorithm_plan_api_tool(planner)),
        normalize_algorithm_plan=_node(normalize_algorithm_plan()),
        present_recommendation=_node(present_recommendation()),
        recommendation_decision=recommendation_decision,
    )


def build_tool_orchestration_subgraph():
    return _build_tool_orchestration_subgraph(
        tool_recall_node=_node(tool_recall_node()),
        planning_agent=_node(planning_agent()),
        topology_agent=_node(topology_agent()),
        parameter_agent=_node(parameter_agent()),
        judge_agent=_node(judge_agent()),
    )


def initial_state() -> GraphState:
    return AgentState().model_dump()


def append_user_message(state: GraphState, content: str) -> GraphState:
    agent_state = agent_state_from_graph(state)
    agent_state.messages.append(ChatMessage(role="user", content=content))
    return agent_state.model_dump()


_node = node_adapter
_agent_state_from_graph = agent_state_from_graph
_coerce_message = coerce_message


def parent_agent_decision(state: GraphState) -> str:
    if state.get("turn_complete"):
        return "end"
    return state.get("next_agent") or "end"


def extract_slots(llm: OpenAICompatibleClient) -> Callable[[AgentState], AgentState]:
    def run(state: AgentState) -> AgentState:
        user_messages = [msg for msg in state.messages if msg.role == "user"][-8:]
        payload = {
            "existing_slots": state.slots.model_dump(),
            "latest_user_message": user_messages[-1].content if user_messages else "",
            "user_message_history": [msg.content for msg in user_messages],
            "summary": state.summary,
        }
        prompt = [
            {
                "role": "system",
                "content": load_prompt("extract_slots.md"),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        try:
            data = parse_json_object(llm.chat(prompt, temperature=0.0))
            extracted = AlgorithmSlots(**{key: data.get(key) for key in AlgorithmSlots.model_fields})
        except Exception:
            extracted = heuristic_slots(payload["latest_user_message"])
        state.slots = state.slots.merge(heuristic_slots(payload["latest_user_message"])).merge(extracted)
        state.missing_slots = state.slots.missing()
        state.missing_required_slots = state.slots.missing_required()
        state.missing_optional_slots = state.slots.missing_optional()
        return state

    return run


def supervisor_route() -> Callable[[AgentState], AgentState]:
    def run(state: AgentState) -> AgentState:
        state.active_agent = "supervisor"
        state.next_agent = None
        state.turn_complete = False
        latest = latest_user_message(state).strip().lower()
        if state.stage == "reviewing" and latest in {"yes", "y", "确认", "同意"}:
            state.status = "confirmed"
            state.stage = "confirmed"
        return state

    return run


def supervisor_decision(state: GraphState) -> str:
    agent_state = _agent_state_from_graph(state)
    if agent_state.status == "confirmed":
        return "confirm_solution"
    if agent_state.stage in {"intake", "kb_retrieval"}:
        return "delegate_knowledge_base"
    if agent_state.stage in {"kb_feedback", "kb_satisfaction_check"}:
        return "handle_kb_feedback"
    if agent_state.stage == "recommendation":
        return "delegate_recommendation"
    if agent_state.slots.is_ready():
        return "generate_summary"
    return "ask_followup"


def delegate_knowledge_base() -> Callable[[AgentState], AgentState]:
    def run(state: AgentState) -> AgentState:
        state.next_agent = "knowledge_base_agent"
        state.turn_complete = False
        return state

    return run


def delegate_recommendation() -> Callable[[AgentState], AgentState]:
    def run(state: AgentState) -> AgentState:
        state.next_agent = "recommendation_agent"
        state.turn_complete = False
        return state

    return run


def retrieve_knowledge(kb: KnowledgeBase) -> Callable[[AgentState], AgentState]:
    def run(state: AgentState) -> AgentState:
        state.active_agent = "knowledge_base_agent"
        state.stage = "kb_retrieval"
        state.next_agent = None
        state.turn_complete = False
        query = build_knowledge_query(state)
        hits = kb.search(query, mode="hybrid", top_k=4) if query else []
        state.kb_context = [hit for hit in hits if hit.hybrid_score >= KB_MATCH_THRESHOLD]
        state.kb_cases = []
        if not state.kb_context:
            state.kb_has_match = False
            state.stage = "recommendation"
        return state

    return run


def summarize_knowledge_cases(llm: OpenAICompatibleClient) -> Callable[[AgentState], AgentState]:
    def run(state: AgentState) -> AgentState:
        if not state.kb_context:
            state.kb_has_match = False
            state.stage = "recommendation"
            return state

        payload = {
            "requirements": state.slots.model_dump(),
            "query": build_knowledge_query(state),
            "hits": [
                {"source": hit.source, "content": hit.content, "score": hit.hybrid_score}
                for hit in state.kb_context
            ],
        }
        prompt = [
            {"role": "system", "content": load_prompt("knowledge_base_agent.md")},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        try:
            data = parse_json_object(llm.chat(prompt, temperature=0.1))
            cases = [KnowledgeCase.model_validate(item) for item in data.get("cases", [])[:3]]
        except Exception:
            cases = fallback_knowledge_cases(state.kb_context)
        state.kb_cases = cases
        state.kb_has_match = bool(cases)
        state.stage = "kb_feedback" if state.kb_has_match else "recommendation"
        return state

    return run


def knowledge_result_decision(state: GraphState) -> str:
    agent_state = _agent_state_from_graph(state)
    return "present_kb_cases" if agent_state.kb_has_match and agent_state.kb_cases else "delegate_recommendation"


def present_kb_cases(llm: OpenAICompatibleClient) -> Callable[[AgentState], AgentState]:
    def run(state: AgentState) -> AgentState:
        payload = {"cases": [case.model_dump() for case in state.kb_cases]}
        prompt = [
            {"role": "system", "content": load_prompt("kb_satisfaction_check.md")},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        try:
            reply = llm.chat(prompt, temperature=0.2)
        except Exception:
            reply = fallback_present_kb_cases(state.kb_cases)
        state.assistant_reply = reply
        state.messages.append(ChatMessage(role="assistant", content=reply))
        state.user_satisfaction = "unknown"
        state.stage = "kb_satisfaction_check"
        state.status = "collecting"
        state.next_agent = "end"
        state.turn_complete = True
        return state

    return run


def handle_kb_feedback() -> Callable[[AgentState], AgentState]:
    def run(state: AgentState) -> AgentState:
        feedback = latest_user_message(state)
        state.user_satisfaction = classify_satisfaction(feedback)
        if state.user_satisfaction == "satisfied":
            state.stage = "solution_refinement"
            if state.kb_cases and not state.recommended_solution:
                case = state.kb_cases[0]
                state.recommended_solution = RecommendedSolution(
                    name=case.title,
                    fit_reason=case.match_reason,
                    core_approach=case.core_flow,
                    preconditions=case.scenario,
                    open_questions=[],
                )
                state.recommended_solution_reason = case.match_reason
        elif state.user_satisfaction == "unsatisfied":
            state.stage = "recommendation"
        else:
            state.stage = "kb_retrieval"
        return state

    return run


def feedback_decision(state: GraphState) -> str:
    agent_state = _agent_state_from_graph(state)
    if agent_state.user_satisfaction == "unsatisfied":
        return "delegate_recommendation"
    if agent_state.user_satisfaction == "unknown":
        return "delegate_knowledge_base"
    if agent_state.slots.is_ready():
        return "generate_summary"
    return "ask_followup"


def check_recommendation_requirements() -> Callable[[AgentState], AgentState]:
    def run(state: AgentState) -> AgentState:
        state.active_agent = "recommendation_agent"
        state.stage = "recommendation"
        state.next_agent = None
        state.turn_complete = False
        return state

    return run


def recommendation_followup(llm: OpenAICompatibleClient) -> Callable[[AgentState], AgentState]:
    def run(state: AgentState) -> AgentState:
        missing_requirements = state.slots.missing_recommendation_requirements()
        followup_payload = {
            "known_slots": state.slots.model_dump(),
            "missing_recommendation_requirements": missing_requirements,
            "user_feedback": latest_user_message(state),
        }
        followup_prompt = [
            {"role": "system", "content": load_prompt("recommendation_followup.md")},
            {"role": "user", "content": json.dumps(followup_payload, ensure_ascii=False)},
        ]
        try:
            reply = llm.chat(followup_prompt, temperature=0.2)
        except Exception:
            reply = fallback_recommendation_followup(missing_requirements)
        state.assistant_reply = reply
        state.messages.append(ChatMessage(role="assistant", content=reply))
        state.status = "collecting"
        state.next_agent = "end"
        state.turn_complete = True
        return state

    return run


def recommendation_decision(state: GraphState) -> str:
    agent_state = _agent_state_from_graph(state)
    return "build_algorithm_plan_request" if agent_state.slots.is_ready_for_recommendation() else "recommendation_followup"


def build_algorithm_plan_request() -> Callable[[AgentState], AgentState]:
    def run(state: AgentState) -> AgentState:
        slots = state.slots
        state.algorithm_main_description = "；".join(
            part
            for part in [
                slots.problem_goal,
                f"输入：{slots.inputs}" if slots.inputs else None,
                f"输出：{slots.outputs}" if slots.outputs else None,
                f"约束：{slots.constraints}" if slots.constraints else None,
                f"指标：{slots.metrics}" if slots.metrics else None,
            ]
            if part
        )
        state.algorithm_plan_request = AlgorithmPlanRequest(
            algorithm_name=state.current_app_name or slots.problem_goal or "算法应用",
            algorithm_description=state.algorithm_main_description,
            input_data=split_values(slots.inputs),
            output_target=split_values(slots.outputs),
            constraints=split_values(slots.constraints),
            metrics=split_values(slots.metrics),
            kb_context=[case.core_flow for case in state.kb_cases] or [hit.content for hit in state.kb_context[:3]],
            user_profile=state.user_profile.model_dump(exclude_none=True),
        )
        return state

    return run


def algorithm_plan_api_tool(planner: AlgorithmPlanner) -> Callable[[AgentState], AgentState]:
    def run(state: AgentState) -> AgentState:
        if not state.algorithm_plan_request:
            return state
        state.algorithm_plan_raw_response = planner.generate(state.algorithm_plan_request)
        return state

    return run


def normalize_algorithm_plan() -> Callable[[AgentState], AgentState]:
    def run(state: AgentState) -> AgentState:
        try:
            plan = AlgorithmPlan.model_validate(state.algorithm_plan_raw_response or {})
        except Exception:
            fallback = SimulatedAlgorithmPlanner()
            plan = AlgorithmPlan.model_validate(fallback.generate(state.algorithm_plan_request))
        state.algorithm_plan_normalized = plan
        core_approach = "；".join(f"{step.step_name}：{step.description}" for step in plan.pipeline)
        solution = RecommendedSolution(
            name=plan.solution_name,
            fit_reason=plan.overall_idea,
            core_approach=core_approach,
            preconditions=json.dumps(plan.deployment, ensure_ascii=False),
            open_questions=[],
            plan=plan,
        )
        state.recommended_solution = solution
        state.recommended_solution_reason = solution.fit_reason
        state.slots.core_steps = core_approach
        state.missing_slots = state.slots.missing()
        state.missing_required_slots = state.slots.missing_required()
        state.missing_optional_slots = state.slots.missing_optional()
        return state

    return run


def present_recommendation() -> Callable[[AgentState], AgentState]:
    def run(state: AgentState) -> AgentState:
        solution = state.recommended_solution or fallback_recommendation(state)
        state.assistant_reply = format_recommendation(solution)
        state.messages.append(ChatMessage(role="assistant", content=state.assistant_reply))
        state.stage = "solution_refinement"
        state.status = "collecting"
        state.next_agent = "end"
        state.turn_complete = True
        return state

    return run


def ask_followup(llm: OpenAICompatibleClient) -> Callable[[AgentState], AgentState]:
    def run(state: AgentState) -> AgentState:
        state.active_agent = "supervisor"
        state.stage = "solution_refinement"
        latest_message = latest_user_message(state)
        prompt = [
            {
                "role": "system",
                "content": load_prompt("ask_followup.md"),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "known_slots": state.slots.model_dump(),
                        "missing_required_slots": state.missing_required_slots,
                        "optional_missing_slots": state.missing_optional_slots,
                        "knowledge": [hit.content for hit in state.kb_context],
                        "recommended_solution": (
                            state.recommended_solution.model_dump() if state.recommended_solution else None
                        ),
                        "latest_user_message": latest_message,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        try:
            reply = llm.chat(prompt, temperature=0.2)
        except Exception:
            reply = fallback_followup(state.missing_required_slots)
        state.assistant_reply = reply
        state.messages.append(ChatMessage(role="assistant", content=reply))
        state.status = "collecting"
        state.next_agent = "end"
        state.turn_complete = True
        return state

    return run


def generate_summary(llm: OpenAICompatibleClient) -> Callable[[AgentState], AgentState]:
    def run(state: AgentState) -> AgentState:
        state.active_agent = "supervisor"
        payload = {
            "slots": state.slots.model_dump(),
            "knowledge": [
                {"source": hit.source, "content": hit.content, "score": hit.hybrid_score}
                for hit in state.kb_context
            ],
            "knowledge_cases": [case.model_dump() for case in state.kb_cases],
            "recommended_solution": (
                state.recommended_solution.model_dump() if state.recommended_solution else None
            ),
        }
        prompt = [
            {
                "role": "system",
                "content": load_prompt("generate_summary.md"),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        try:
            markdown = llm.chat(prompt, temperature=0.2)
        except Exception:
            markdown = fallback_summary(state)
        state.draft_summary_markdown = markdown
        state.draft_solution_json = payload
        state.assistant_reply = f"{markdown}\n\n请确认该方案是否可作为第一版方案。输入 yes/确认 进入 confirmed，或继续补充修改。"
        state.messages.append(ChatMessage(role="assistant", content=state.assistant_reply))
        state.status = "reviewing"
        state.stage = "reviewing"
        state.next_agent = "end"
        state.turn_complete = False
        return state

    return run


def confirm_solution() -> Callable[[AgentState], AgentState]:
    def run(state: AgentState) -> AgentState:
        state.active_agent = "supervisor"
        state.status = "confirmed"
        state.stage = "confirmed"
        state.algorithm_main_description = build_algorithm_main_description(state)
        state.assistant_reply = "方案已确认。"
        state.messages.append(ChatMessage(role="assistant", content=state.assistant_reply))
        state.next_agent = "tool_orchestration_agent"
        state.turn_complete = False
        return state

    return run


def tool_recall_node() -> Callable[[AgentState], AgentState]:
    def run(state: AgentState) -> AgentState:
        state.active_agent = "tool_orchestration_agent"
        state.stage = "tool_orchestration"
        state.next_agent = None
        state.turn_complete = False
        state.algorithm_main_description = state.algorithm_main_description or build_algorithm_main_description(state)
        # TODO: Replace this placeholder with the real retrieval service for tool recall.
        state.tool_recall_result = {
            "query": state.algorithm_main_description,
            "candidates": [],
            "note": "placeholder: tool recall service is not implemented yet",
        }
        return state

    return run


def planning_agent() -> Callable[[AgentState], AgentState]:
    def run(state: AgentState) -> AgentState:
        # TODO: Replace this placeholder with a Planning Agent that selects tools.
        state.selected_tools = []
        return state

    return run


def topology_agent() -> Callable[[AgentState], AgentState]:
    def run(state: AgentState) -> AgentState:
        # TODO: Replace this placeholder with a Topology Agent that arranges selected tools.
        state.tool_topology = {
            "description": state.algorithm_main_description,
            "nodes": [],
            "edges": [],
        }
        return state

    return run


def parameter_agent() -> Callable[[AgentState], AgentState]:
    def run(state: AgentState) -> AgentState:
        # TODO: Replace this placeholder with a Parameter Agent that generates tool inputs.
        state.tool_parameters = {"parameters": {}, "note": "placeholder: parameters are not generated yet"}
        return state

    return run


def judge_agent() -> Callable[[AgentState], AgentState]:
    def run(state: AgentState) -> AgentState:
        # TODO: Replace this placeholder with a Judge Agent for post-processing and validation.
        state.orchestration_judgement = {
            "status": "placeholder_ready",
            "message": "tool orchestration chain completed with placeholder results",
        }
        state.next_agent = "end"
        state.turn_complete = True
        return state

    return run


def tool_router(tools: ToolRegistry) -> Callable[[AgentState], AgentState]:
    def run(state: AgentState) -> AgentState:
        if not state.slots.core_steps:
            return state
        try:
            result = tools.invoke(
                "estimate_complexity",
                {
                    "core_steps": state.slots.core_steps,
                    "constraints": state.slots.constraints or "",
                },
            )
            state.draft_solution_json = state.draft_solution_json or {}
            state.draft_solution_json["tool_estimates"] = {"estimate_complexity": result}
        except Exception:
            state.turn_complete = True
            return state
        state.next_agent = "end"
        state.turn_complete = True
        return state

    return run


def latest_user_message(state: AgentState) -> str:
    return next((msg.content for msg in reversed(state.messages) if msg.role == "user"), "")


def build_algorithm_main_description(state: AgentState) -> str:
    """Return the one-sentence confirmed solution description passed into orchestration."""

    if state.recommended_solution:
        solution = state.recommended_solution
        return compact_sentence(
            f"{solution.name}，核心路线是{solution.core_approach}。"
        )

    slots = state.slots
    parts = [
        slots.problem_goal,
        f"输入为{slots.inputs}" if slots.inputs else None,
        f"输出为{slots.outputs}" if slots.outputs else None,
        f"核心流程为{slots.core_steps}" if slots.core_steps else None,
        f"约束为{slots.constraints}" if slots.constraints else None,
    ]
    description = "，".join(part for part in parts if part)
    if description:
        return compact_sentence(f"{description}。")

    if state.draft_summary_markdown:
        lines = [
            line.strip(" #-\t")
            for line in state.draft_summary_markdown.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if lines:
            return compact_sentence(f"{lines[0]}。")

    return "基于用户确认方案进行工具召回、工具选择、拓扑编排、参数生成和后处理。"


def compact_sentence(text: str, max_length: int = 180) -> str:
    sentence = " ".join(text.replace("\n", " ").split())
    sentence = sentence.rstrip("。；;,.，")
    if len(sentence) > max_length:
        sentence = sentence[: max_length - 1].rstrip("，；;,. ")
    return f"{sentence}。"


def build_knowledge_query(state: AgentState) -> str:
    return " ".join(
        part
        for part in [
            state.slots.problem_goal,
            state.slots.inputs,
            state.slots.outputs,
            state.slots.core_steps,
            latest_user_message(state),
        ]
        if part
    )


def split_values(value: str | None) -> list[str]:
    if not value:
        return []
    parts = [item.strip() for item in value.replace("；", ",").replace("、", ",").split(",")]
    return [item for item in parts if item]


def classify_satisfaction(text: str) -> Literal["unknown", "satisfied", "unsatisfied"]:
    normalized = text.strip().lower()
    unsatisfied_terms = ["不满意", "不合适", "没有合适", "没命中", "不符合", "不够", "换一个", "重新推荐"]
    satisfied_terms = ["满意", "合适", "符合", "可以", "有帮助", "就这个", "采用", "参考这个"]
    if any(term in normalized for term in unsatisfied_terms):
        return "unsatisfied"
    if any(term in normalized for term in satisfied_terms):
        return "satisfied"
    return "unknown"


def fallback_knowledge_cases(hits: list[Any]) -> list[KnowledgeCase]:
    cases = []
    for hit in hits[:3]:
        title = hit.source.removeprefix("seed:").removesuffix(".md").replace("_", " ")
        cases.append(
            KnowledgeCase(
                title=title,
                scenario="与当前需求语义相近的历史算法案例",
                core_flow=hit.content[:500],
                match_reason=f"该案例在知识库检索中的匹配分数为 {hit.hybrid_score:.3f}。",
                source=hit.source,
            )
        )
    return cases


def fallback_present_kb_cases(cases: list[KnowledgeCase]) -> str:
    sections = ["我从知识库中找到了以下相关历史案例："]
    for index, case in enumerate(cases, 1):
        sections.append(
            f"### {index}. {case.title}\n"
            f"- 适用场景：{case.scenario}\n"
            f"- 核心流程：{case.core_flow}\n"
            f"- 匹配原因：{case.match_reason}"
        )
    sections.append("这些历史案例是否符合你的需求？你可以回复“满意”“不满意”，或继续补充需求。")
    return "\n\n".join(sections)


def fallback_recommendation(state: AgentState) -> RecommendedSolution:
    goal = state.slots.problem_goal or "当前算法业务需求"
    return RecommendedSolution(
        name=f"{goal}的多阶段视觉识别方案",
        fit_reason="该方案能够把目标检测、时序关系判断和多帧融合结合起来，适合从业务事件角度降低误报。",
        core_approach="先检测关键目标并进行跟踪，再建模关键动作或目标关系，最后通过多帧置信度融合输出事件。",
        preconditions=state.slots.inputs or "需要确认输入数据类型、摄像头场景和部署算力。",
        open_questions=state.missing_required_slots[:3],
    )


def fallback_recommendation_followup(missing_requirements: list[str]) -> str:
    labels = {
        "problem_goal": "请再说明算法要解决的具体业务问题，以及希望识别或判断什么目标？",
        "inputs": "输入数据是什么？请说明数据类型、采集场景，以及摄像头或传感器条件。",
        "outputs": "你希望算法输出哪些结果？例如事件、目标位置、证据帧、时间段或置信度。",
        "constraints": "有哪些关键约束？例如实时性、部署算力、误报漏报偏好、低光照或遮挡。",
        "metrics": "你希望用哪些指标验收方案？例如 precision、recall、F1、误报率或事件级准确率。",
    }
    question = next((labels[key] for key in missing_requirements if key in labels), "请继续补充关键业务需求。")
    return f"我已经记录了你对历史案例不满意，接下来由方案推荐 Agent 进一步确认需求。\n\n{question}"


def format_recommendation(solution: RecommendedSolution) -> str:
    questions = "\n".join(f"- {item}" for item in solution.open_questions) or "- 暂无"
    plan_sections = ""
    if solution.plan:
        pipeline = "\n".join(
            f"{index}. **{step.step_name}**：{step.description}"
            for index, step in enumerate(solution.plan.pipeline, 1)
        )
        risks = "\n".join(f"- {item}" for item in solution.plan.risks) or "- 暂无"
        plan_sections = (
            f"\n\n## 实施流程\n{pipeline}"
            f"\n\n## 数据要求\n{json.dumps(solution.plan.data_requirement, ensure_ascii=False)}"
            f"\n\n## 部署建议\n{json.dumps(solution.plan.deployment, ensure_ascii=False)}"
            f"\n\n## 评估建议\n{json.dumps(solution.plan.evaluation, ensure_ascii=False)}"
            f"\n\n## 主要风险\n{risks}"
        )
    return (
        "# 推荐算法方案\n\n"
        f"## 推荐方案\n{solution.name}\n\n"
        f"## 推荐原因\n{solution.fit_reason}\n\n"
        f"## 核心技术路线\n{solution.core_approach}\n\n"
        f"## 适用前提\n{solution.preconditions}\n\n"
        f"## 后续需要确认\n{questions}"
        f"{plan_sections}\n\n"
        "请确认这个推荐方案是否符合你的预期。你可以继续补充修改，确认后我会形成正式方案总结。"
    )


def heuristic_slots(text: str) -> AlgorithmSlots:
    goal = None
    if "抽烟" in text:
        goal = "识别视频或图像中的抽烟行为"
    elif "安全帽" in text:
        goal = "识别人员是否佩戴安全帽"
    elif "跌倒" in text:
        goal = "识别人员跌倒事件"
    elif "入侵" in text:
        goal = "识别指定区域入侵事件"
    elif "烟雾" in text or "明火" in text:
        goal = "识别明火或烟雾事件"

    inputs = None
    if any(word in text for word in ["单张图片", "单张图像", "单帧图片", "单帧图像"]):
        inputs = "单张图片"
    elif any(word in text for word in ["视频流", "实时视频"]):
        inputs = "视频流"
    elif any(word in text for word in ["图片序列", "图像序列"]):
        inputs = "图片序列"
    elif any(word in text for word in ["视频", "图像", "图片", "摄像头"]):
        inputs = "视频流或图片序列"

    outputs = None
    if any(word in text for word in ["输出", "是否有人", "判断的证据", "证据帧", "置信度", "目标位置"]):
        outputs = text.strip()
    elif any(word in text for word in ["识别", "检测", "算法"]):
        outputs = "事件结果、目标位置、置信度和证据帧"

    constraints = text.strip() if any(
        word in text
        for word in ["实时", "延迟", "ms", "毫秒", "gpu", "GPU", "服务器", "边缘", "不能漏报", "宁可多报"]
    ) else None
    metrics = text.strip() if any(
        word in text
        for word in ["Accuracy", "accuracy", "准确率", "precision", "recall", "F1", "误报率", "漏报率"]
    ) else None

    return AlgorithmSlots(
        problem_goal=goal,
        inputs=inputs,
        outputs=outputs,
        constraints=constraints,
        metrics=metrics,
    )


def fallback_followup(missing_slots: list[str]) -> str:
    intro = "我已经记录了你刚刚补充的关键信息。"
    labels = {
        "problem_goal": "算法要解决的具体业务目标是什么？",
        "inputs": "输入数据是什么，例如视频流、单张图片、传感器数据，是否有摄像头场景信息？",
        "outputs": "希望输出哪些结果，例如事件标签、目标框、时间段、置信度或证据帧？",
        "core_steps": "你期望的核心处理流程有哪些，是否需要检测、跟踪、时序判断或规则融合？",
        "constraints": "有哪些部署约束，例如实时性、边缘设备、低误报、低光照或遮挡？",
        "metrics": "验收指标希望关注哪些，例如 precision、recall、F1、误报率或事件级准确率？",
    }
    question = next((labels[key] for key in missing_slots if key in labels), "请确认是否可以基于当前信息生成第一版算法方案总结。")
    return f"{intro}\n\n接下来我还需要确认一个关键点：{question}"


def fallback_summary(state: AgentState) -> str:
    slots = state.slots
    references = "\n".join(f"- {hit.source}" for hit in state.kb_context[:3]) or "- 暂无知识库引用"
    return f"""# 算法方案总结

## 目标
{slots.problem_goal or "待确认"}

## 输入输出
- 输入：{slots.inputs or "待确认"}
- 输出：{slots.outputs or "待确认"}

## 核心流程
{slots.core_steps or "待确认"}

## 关键约束
{slots.constraints or "待确认"}

## 假设与边界
- 假设：{slots.assumptions or "待确认"}
- 边界场景：{slots.edge_cases or "待确认"}

## 评估指标
{slots.metrics or "待确认"}

## 知识库引用
{references}
"""
