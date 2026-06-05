from __future__ import annotations

import re
from datetime import datetime

from agent_solution.models import AppSession, IntentResult
from agent_solution.storage import BusinessStore


class ThreadManager:
    def __init__(self, store: BusinessStore):
        self.store = store

    def resolve_thread(
        self,
        user_id: str,
        intent: IntentResult,
        current_session: AppSession | None = None,
    ) -> AppSession | None:
        if intent.switch_type == "ambiguous":
            return None
        if intent.switch_type == "same_app" and current_session:
            return current_session

        if intent.switch_type in {"compare_apps", "general_recommendation"}:
            return None

        app_id = intent.detected_app_id or "unknown_app"
        app_name = intent.detected_app_name or "未命名算法应用"
        return self._create_session(user_id, app_id, app_name, "single_app", app_id=app_id)

    def _create_session(
        self,
        user_id: str,
        key: str,
        app_name: str,
        session_type: str,
        app_id: str | None = None,
    ) -> AppSession:
        timestamp = datetime.now().strftime("%Y%m%d")
        sequence = len(self.store.list_sessions(user_id)) + 1
        safe_user = re.sub(r"[^a-zA-Z0-9_-]+", "_", user_id)
        safe_key = re.sub(r"[^a-zA-Z0-9_-]+", "_", key)
        session_id = f"{safe_user}__{safe_key}__{timestamp}_{sequence:03d}"
        return self.store.create_session(
            AppSession(
                session_id=session_id,
                user_id=user_id,
                thread_id=session_id,
                app_id=app_id,
                app_name=app_name,
                session_type=session_type,
            )
        )
