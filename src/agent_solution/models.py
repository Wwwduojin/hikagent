from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


SLOT_KEYS = [
    "problem_goal",
    "inputs",
    "outputs",
    "core_steps",
    "constraints",
    "assumptions",
    "metrics",
    "edge_cases",
]

REQUIRED_SLOT_KEYS = [
    "problem_goal",
    "inputs",
    "outputs",
    "core_steps",
    "constraints",
    "metrics",
]

OPTIONAL_SLOT_KEYS = [
    "assumptions",
    "edge_cases",
]

RECOMMENDATION_REQUIREMENT_KEYS = [
    "problem_goal",
    "inputs",
    "outputs",
    "constraints",
    "metrics",
]


class AlgorithmSlots(BaseModel):
    problem_goal: str | None = None
    inputs: str | None = None
    outputs: str | None = None
    core_steps: str | None = None
    constraints: str | None = None
    assumptions: str | None = None
    metrics: str | None = None
    edge_cases: str | None = None

    def merge(self, other: "AlgorithmSlots") -> "AlgorithmSlots":
        data = self.model_dump()
        for key, value in other.model_dump().items():
            if value:
                existing = data.get(key)
                if key in {"constraints", "metrics"} and existing:
                    if existing in value:
                        data[key] = value
                    elif value not in existing:
                        data[key] = f"{existing}；{value}"
                else:
                    data[key] = value
        return AlgorithmSlots(**data)

    def missing(self) -> list[str]:
        return [key for key in SLOT_KEYS if not getattr(self, key)]

    def missing_required(self) -> list[str]:
        return [key for key in REQUIRED_SLOT_KEYS if not getattr(self, key)]

    def missing_optional(self) -> list[str]:
        return [key for key in OPTIONAL_SLOT_KEYS if not getattr(self, key)]

    def is_ready(self) -> bool:
        return all(getattr(self, key) for key in REQUIRED_SLOT_KEYS)

    def missing_recommendation_requirements(self) -> list[str]:
        return [key for key in RECOMMENDATION_REQUIREMENT_KEYS if not getattr(self, key)]

    def is_ready_for_recommendation(self) -> bool:
        return all(getattr(self, key) for key in RECOMMENDATION_REQUIREMENT_KEYS)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class KnowledgeHit(BaseModel):
    chunk_id: int
    document_id: int
    source: str
    content: str
    vector_score: float = 0.0
    keyword_score: float = 0.0
    hybrid_score: float = 0.0


class KnowledgeCase(BaseModel):
    title: str
    scenario: str
    core_flow: str
    match_reason: str
    source: str


class IntentResult(BaseModel):
    intent_type: Literal[
        "single_app_question",
        "compare_apps",
        "general_recommendation",
        "kb_feedback",
        "solution_confirmation",
        "ambiguous",
    ] = "single_app_question"
    switch_type: Literal[
        "same_app",
        "new_app",
        "compare_apps",
        "general_recommendation",
        "ambiguous",
    ] = "same_app"
    detected_app_id: str | None = None
    detected_app_name: str | None = None
    detected_apps: list[dict[str, str]] = Field(default_factory=list)
    confidence: float = 0.0
    is_kb_satisfaction_feedback: bool = False
    kb_satisfaction: Literal["satisfied", "unsatisfied", "unclear"] | None = None
    is_solution_confirmation: bool = False
    reason: str | None = None


class UserProfile(BaseModel):
    industry: str | None = None
    role: str | None = None
    company_size: str | None = None
    tech_level: str | None = None
    deployment_preference: str | None = None
    budget_level: str | None = None
    latency_requirement: str | None = None
    compliance_requirement: str | None = None
    acceptance_preference: str | None = None

    def merge(self, other: "UserProfile") -> "UserProfile":
        data = self.model_dump()
        for key, value in other.model_dump().items():
            if value:
                data[key] = value
        return UserProfile(**data)


class AlgorithmPlanStep(BaseModel):
    step_name: str
    description: str
    input: list[str] = Field(default_factory=list)
    output: list[str] = Field(default_factory=list)


class AlgorithmPlanRequest(BaseModel):
    algorithm_name: str
    algorithm_description: str
    input_data: list[str] = Field(default_factory=list)
    output_target: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    kb_context: list[str] = Field(default_factory=list)
    user_profile: dict[str, Any] = Field(default_factory=dict)


class AlgorithmPlan(BaseModel):
    solution_name: str
    overall_idea: str
    pipeline: list[AlgorithmPlanStep] = Field(default_factory=list)
    model_recommendation: dict[str, Any] = Field(default_factory=dict)
    data_requirement: dict[str, Any] = Field(default_factory=dict)
    deployment: dict[str, Any] = Field(default_factory=dict)
    evaluation: dict[str, Any] = Field(default_factory=dict)
    risks: list[str] = Field(default_factory=list)


class RecommendedSolution(BaseModel):
    name: str
    fit_reason: str
    core_approach: str
    preconditions: str
    open_questions: list[str] = Field(default_factory=list)
    plan: AlgorithmPlan | None = None


class AppSession(BaseModel):
    session_id: str
    user_id: str
    thread_id: str
    app_id: str | None = None
    app_name: str | None = None
    session_type: Literal["single_app", "compare_apps", "general_recommendation"] = "single_app"
    status: str = "collecting"
    stage: str = "intake"
    summary: str = ""
    created_at: str | None = None
    updated_at: str | None = None


class ChatResponse(BaseModel):
    message: str
    user_id: str
    thread_id: str | None = None
    current_app_id: str | None = None
    current_app_name: str | None = None
    session_type: str | None = None
    stage: str | None = None
    status: str | None = None
    need_clarification: bool = False


class ToolCallRecord(BaseModel):
    tool_name: str
    arguments: dict[str, Any]
    result: dict[str, Any] | None = None
    error: str | None = None


class AgentState(BaseModel):
    messages: list[ChatMessage] = Field(default_factory=list)
    user_id: str = "u_001"
    thread_id: str | None = None
    session_type: Literal["single_app", "compare_apps", "general_recommendation"] = "single_app"
    current_app_id: str | None = None
    current_app_name: str | None = None
    intent_type: str | None = None
    switch_type: str | None = None
    detected_app_id: str | None = None
    detected_app_name: str | None = None
    detected_apps: list[dict[str, str]] = Field(default_factory=list)
    intent_confidence: float | None = None
    intent_reason: str | None = None
    is_kb_satisfaction_feedback: bool = False
    kb_satisfaction: str | None = None
    is_solution_confirmation: bool = False
    user_profile: UserProfile = Field(default_factory=UserProfile)
    user_constraints: dict[str, Any] = Field(default_factory=dict)
    slots: AlgorithmSlots = Field(default_factory=AlgorithmSlots)
    missing_slots: list[str] = Field(default_factory=lambda: SLOT_KEYS.copy())
    missing_required_slots: list[str] = Field(default_factory=lambda: REQUIRED_SLOT_KEYS.copy())
    missing_optional_slots: list[str] = Field(default_factory=lambda: OPTIONAL_SLOT_KEYS.copy())
    stage: Literal[
        "intake",
        "kb_retrieval",
        "kb_feedback",
        "kb_satisfaction_check",
        "recommendation",
        "solution_refinement",
        "reviewing",
        "confirmed",
        "tool_orchestration",
    ] = "intake"
    active_agent: Literal[
        "supervisor",
        "knowledge_base_agent",
        "recommendation_agent",
        "tool_orchestration_agent",
    ] = "supervisor"
    kb_context: list[KnowledgeHit] = Field(default_factory=list)
    kb_cases: list[KnowledgeCase] = Field(default_factory=list)
    kb_has_match: bool = False
    user_satisfaction: Literal["unknown", "satisfied", "unsatisfied"] = "unknown"
    recommended_solution: RecommendedSolution | None = None
    recommended_solution_reason: str | None = None
    algorithm_main_description: str | None = None
    algorithm_plan_request: AlgorithmPlanRequest | None = None
    algorithm_plan_raw_response: dict[str, Any] | None = None
    algorithm_plan_normalized: AlgorithmPlan | None = None
    comparison_result: dict[str, Any] | None = None
    general_recommendations: list[dict[str, Any]] = Field(default_factory=list)
    draft_summary_markdown: str | None = None
    draft_solution_json: dict[str, Any] | None = None
    assistant_reply: str | None = None
    status: Literal["collecting", "reviewing", "confirmed"] = "collecting"
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    tool_recall_result: dict[str, Any] | None = None
    selected_tools: list[dict[str, Any]] = Field(default_factory=list)
    tool_topology: dict[str, Any] | None = None
    tool_parameters: dict[str, Any] | None = None
    orchestration_judgement: dict[str, Any] | None = None
    summary: str = ""
    unresolved_slots: list[str] = Field(default_factory=list)
    next_agent: Literal[
        "knowledge_base_agent",
        "recommendation_agent",
        "tool_orchestration_agent",
        "end",
    ] | None = None
    turn_complete: bool = False
