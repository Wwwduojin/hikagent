from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_solution.config import ensure_parent
from agent_solution.models import AppSession, UserProfile


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class BusinessStore:
    """SQLite persistence for user-level preferences and application sessions."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        ensure_parent(db_path)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS user_profile (
                    user_id TEXT PRIMARY KEY,
                    profile_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS app_session (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    thread_id TEXT UNIQUE NOT NULL,
                    app_id TEXT,
                    app_name TEXT,
                    session_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    summary TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_app_session_user_updated
                ON app_session(user_id, updated_at DESC);

                CREATE TABLE IF NOT EXISTS app_session_state (
                    session_id TEXT PRIMARY KEY,
                    state_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES app_session(session_id)
                );

                CREATE TABLE IF NOT EXISTS user_app_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    app_id TEXT,
                    session_id TEXT,
                    action_type TEXT NOT NULL,
                    score REAL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def get_user_profile(self, user_id: str) -> UserProfile:
        with self._connect() as conn:
            row = conn.execute("SELECT profile_json FROM user_profile WHERE user_id = ?", (user_id,)).fetchone()
        if not row:
            return UserProfile()
        return UserProfile.model_validate(json.loads(row["profile_json"]))

    def save_user_profile(self, user_id: str, profile: UserProfile) -> None:
        now = utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO user_profile(user_id, profile_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    profile_json = excluded.profile_json,
                    updated_at = excluded.updated_at
                """,
                (user_id, json.dumps(profile.model_dump(), ensure_ascii=False), now),
            )

    def create_session(self, session: AppSession) -> AppSession:
        now = utc_now()
        data = session.model_copy(
            update={
                "created_at": session.created_at or now,
                "updated_at": now,
            }
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO app_session(
                    session_id, user_id, thread_id, app_id, app_name, session_type,
                    status, stage, summary, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data.session_id,
                    data.user_id,
                    data.thread_id,
                    data.app_id,
                    data.app_name,
                    data.session_type,
                    data.status,
                    data.stage,
                    data.summary,
                    data.created_at,
                    data.updated_at,
                ),
            )
        return data

    def get_session(self, thread_id: str) -> AppSession | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM app_session WHERE thread_id = ?", (thread_id,)).fetchone()
        return AppSession.model_validate(dict(row)) if row else None

    def list_sessions(self, user_id: str) -> list[AppSession]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM app_session WHERE user_id = ? ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
        return [AppSession.model_validate(dict(row)) for row in rows]

    def save_state(self, session: AppSession, state: dict[str, Any]) -> None:
        now = utc_now()
        summary = str(state.get("summary") or "")
        stage = str(state.get("stage") or session.stage)
        status = str(state.get("status") or session.status)
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE app_session
                SET stage = ?, status = ?, summary = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (stage, status, summary, now, session.session_id),
            )
            conn.execute(
                """
                INSERT INTO app_session_state(session_id, state_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    state_json = excluded.state_json,
                    updated_at = excluded.updated_at
                """,
                (session.session_id, json.dumps(state, ensure_ascii=False, default=str), now),
            )

    def load_state(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT state_json FROM app_session_state WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return json.loads(row["state_json"]) if row else None

    def record_history(
        self,
        user_id: str,
        action_type: str,
        session_id: str | None = None,
        app_id: str | None = None,
        score: float | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO user_app_history(user_id, app_id, session_id, action_type, score, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (user_id, app_id, session_id, action_type, score, utc_now()),
            )
