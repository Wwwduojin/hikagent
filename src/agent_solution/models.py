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


class RecommendedSolution(BaseModel):
    name: str
    fit_reason: str
    core_approach: str
    preconditions: str
    open_questions: list[str] = Field(default_factory=list)


class ToolCallRecord(BaseModel):
    tool_name: str
    arguments: dict[str, Any]
    result: dict[str, Any] | None = None
    error: str | None = None


class AgentState(BaseModel):
    messages: list[ChatMessage] = Field(default_factory=list)
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
    ] = "intake"
    active_agent: Literal["supervisor", "knowledge_base_agent", "recommendation_agent"] = "supervisor"
    kb_context: list[KnowledgeHit] = Field(default_factory=list)
    kb_cases: list[KnowledgeCase] = Field(default_factory=list)
    kb_has_match: bool = False
    user_satisfaction: Literal["unknown", "satisfied", "unsatisfied"] = "unknown"
    recommended_solution: RecommendedSolution | None = None
    recommended_solution_reason: str | None = None
    draft_summary_markdown: str | None = None
    draft_solution_json: dict[str, Any] | None = None
    assistant_reply: str | None = None
    status: Literal["collecting", "reviewing", "confirmed"] = "collecting"
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
