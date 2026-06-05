from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, RemoveMessage
from langgraph.checkpoint.memory import MemorySaver

from agent_solution.algorithm_planner import AlgorithmPlanner
from agent_solution.app_router import extract_user_profile
from agent_solution.graph import build_graph, build_intent_router_graph, initial_state
from agent_solution.kb import KnowledgeBase
from agent_solution.llm import OpenAICompatibleClient
from agent_solution.models import AppSession, ChatResponse, IntentResult
from agent_solution.storage import BusinessStore
from agent_solution.thread_manager import ThreadManager
from agent_solution.tools import ToolRegistry
from agent_solution.config import ensure_parent


def create_sqlite_checkpointer(path: str):
    from langgraph.checkpoint.sqlite import SqliteSaver

    ensure_parent(Path(path))
    connection = sqlite3.connect(path, check_same_thread=False)
    return SqliteSaver(connection)


class AgentSolutionService:
    def __init__(
        self,
        llm: OpenAICompatibleClient,
        kb: KnowledgeBase,
        tools: ToolRegistry,
        store: BusinessStore,
        planner: AlgorithmPlanner,
        checkpointer: Any | None = None,
    ):
        self.llm = llm
        self.kb = kb
        self.tools = tools
        self.store = store
        self.planner = planner
        self.checkpointer = checkpointer or MemorySaver()
        self.graph = build_graph(llm, kb, tools, planner=planner, checkpointer=self.checkpointer)
        self.intent_graph = build_intent_router_graph(llm)
        self.thread_manager = ThreadManager(store)

    def chat(
        self,
        user_id: str,
        user_input: str,
        current_thread_id: str | None = None,
    ) -> ChatResponse:
        current_session = self.store.get_session(current_thread_id) if current_thread_id else None
        if current_session and current_session.user_id != user_id:
            current_session = None
        summary = current_session.summary if current_session else ""
        intent_result = self.intent_graph.invoke(
            {
                "user_input": user_input,
                "current_session": current_session.model_dump() if current_session else None,
                "summary": summary,
            }
        )
        intent = IntentResult.model_validate(intent_result["intent"])
        if intent.switch_type in {"compare_apps", "general_recommendation"}:
            return ChatResponse(
                message="当前版本暂时只支持单个算法应用的方案咨询。请告诉我你希望先深入哪个具体算法应用。",
                user_id=user_id,
                thread_id=current_thread_id,
                current_app_id=current_session.app_id if current_session else None,
                current_app_name=current_session.app_name if current_session else None,
                session_type=current_session.session_type if current_session else None,
                stage=current_session.stage if current_session else None,
                status=current_session.status if current_session else None,
                need_clarification=True,
            )
        if intent.switch_type == "ambiguous":
            return ChatResponse(
                message="你是想继续当前算法应用，还是想咨询一个新的算法应用？请补充算法名称或业务场景。",
                user_id=user_id,
                thread_id=current_thread_id,
                current_app_id=current_session.app_id if current_session else None,
                current_app_name=current_session.app_name if current_session else None,
                session_type=current_session.session_type if current_session else None,
                stage=current_session.stage if current_session else None,
                status=current_session.status if current_session else None,
                need_clarification=True,
            )

        target_session = self.thread_manager.resolve_thread(user_id, intent, current_session)
        if not target_session:
            raise RuntimeError("Thread manager did not resolve a target session.")
        profile = self.store.get_user_profile(user_id).merge(extract_user_profile(user_input))
        self.store.save_user_profile(user_id, profile)
        config = {"configurable": {"thread_id": target_session.thread_id, "user_id": user_id}}
        existing_checkpoint = self.graph.get_state(config)
        common = {
            "user_id": user_id,
            "thread_id": target_session.thread_id,
            "session_type": target_session.session_type,
            "current_app_id": target_session.app_id,
            "current_app_name": target_session.app_name,
            "intent_type": intent.intent_type,
            "switch_type": intent.switch_type,
            "detected_app_id": intent.detected_app_id,
            "detected_app_name": intent.detected_app_name,
            "detected_apps": intent.detected_apps,
            "intent_confidence": intent.confidence,
            "intent_reason": intent.reason,
            "is_kb_satisfaction_feedback": intent.is_kb_satisfaction_feedback,
            "kb_satisfaction": intent.kb_satisfaction,
            "is_solution_confirmation": intent.is_solution_confirmation,
            "user_profile": profile.model_dump(),
        }
        if existing_checkpoint.values:
            graph_input = {**common, "messages": [HumanMessage(content=user_input)]}
        else:
            graph_input = initial_state()
            graph_input.update(common)
            graph_input["messages"] = [HumanMessage(content=user_input)]

        result = self.graph.invoke(graph_input, config=config)
        result = self._compact_messages(config, result)
        self.store.save_state(target_session, result)
        self.store.record_history(user_id, "ask", target_session.session_id, target_session.app_id)
        self._record_stage_history(user_id, target_session, result)
        return ChatResponse(
            message=str(result.get("assistant_reply") or ""),
            user_id=user_id,
            thread_id=target_session.thread_id,
            current_app_id=target_session.app_id,
            current_app_name=target_session.app_name,
            session_type=target_session.session_type,
            stage=str(result.get("stage") or target_session.stage),
            status=str(result.get("status") or target_session.status),
        )

    def list_sessions(self, user_id: str) -> list[AppSession]:
        return self.store.list_sessions(user_id)

    def get_session(self, user_id: str, thread_id: str) -> AppSession | None:
        session = self.store.get_session(thread_id)
        if session and session.user_id == user_id:
            return session
        return None

    def _compact_messages(self, config: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        messages = list(result.get("messages", []))
        if len(messages) <= 16:
            return result
        older = messages[:-8]
        lines = []
        removals = []
        for message in older:
            content = getattr(message, "content", "")
            role = getattr(message, "type", "message")
            lines.append(f"{role}: {content}")
            message_id = getattr(message, "id", None)
            if message_id:
                removals.append(RemoveMessage(id=message_id))
        summary = (str(result.get("summary") or "") + "\n" + "\n".join(lines)).strip()
        update: dict[str, Any] = {"summary": summary}
        if removals:
            update["messages"] = removals
        self.graph.update_state(config, update)
        return dict(self.graph.get_state(config).values)

    def _record_stage_history(self, user_id: str, session: AppSession, result: dict[str, Any]) -> None:
        stage = result.get("stage")
        if stage == "kb_satisfaction_check":
            self.store.record_history(user_id, "retrieve_kb", session.session_id, session.app_id)
        if result.get("user_satisfaction") in {"satisfied", "unsatisfied"}:
            self.store.record_history(user_id, result["user_satisfaction"], session.session_id, session.app_id)
        if result.get("recommended_solution"):
            self.store.record_history(user_id, "recommend", session.session_id, session.app_id)
        if result.get("status") == "confirmed":
            self.store.record_history(user_id, "confirm", session.session_id, session.app_id)
