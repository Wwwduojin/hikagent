from __future__ import annotations

from pathlib import Path

from agent_solution.config import Settings
from agent_solution.web import build_web_runtime, chat_payload, chat_stream_events, sessions_payload, use_session_payload


def test_web_runtime_defaults_to_simulated_chat(tmp_path: Path) -> None:
    runtime = build_web_runtime(_settings(tmp_path), user_id="u_web")

    result = chat_payload(runtime, {"message": "我想做一个抽烟识别算法方案"})

    response = result["response"]
    assert response["message"]
    assert response["thread_id"]
    assert response["stage"] == "kb_satisfaction_check"
    assert response["status"] in {"collecting", "reviewing"}
    assert result["current_thread_id"] == response["thread_id"]
    assert result["sessions"][0]["thread_id"] == response["thread_id"]


def test_web_sessions_payload_lists_existing_sessions(tmp_path: Path) -> None:
    runtime = build_web_runtime(_settings(tmp_path), user_id="u_web")
    chat_payload(runtime, {"message": "我想做一个抽烟识别算法方案"})

    result = sessions_payload(runtime)

    assert result["user_id"] == "u_web"
    assert result["current_thread_id"]
    assert len(result["sessions"]) == 1
    assert result["sessions"][0]["app_id"] == "smoking_detection"


def test_web_use_session_rejects_missing_thread(tmp_path: Path) -> None:
    runtime = build_web_runtime(_settings(tmp_path), user_id="u_web")

    result = use_session_payload(runtime, {"thread_id": "missing-thread"})

    assert result == {"error": "session not found"}


def test_web_use_session_returns_history_messages(tmp_path: Path) -> None:
    runtime = build_web_runtime(_settings(tmp_path), user_id="u_web")
    chat = chat_payload(runtime, {"message": "我想做一个抽烟识别算法方案"})

    result = use_session_payload(runtime, {"thread_id": chat["current_thread_id"]})

    assert result["messages"][0]["role"] == "user"
    assert "抽烟识别" in result["messages"][0]["content"]
    assert result["messages"][1]["role"] == "assistant"
    assert result["messages"][1]["content"]


def test_web_chat_accepts_image_attachment_without_text(tmp_path: Path) -> None:
    runtime = build_web_runtime(_settings(tmp_path), user_id="u_web")

    result = chat_payload(
        runtime,
        {
            "attachments": [
                {"name": "smoking-scene.png", "type": "image/png", "size": 2048},
            ]
        },
    )

    assert result["response"]["message"]
    assert result["current_thread_id"]


def test_web_stream_does_not_parse_image_without_attachment(tmp_path: Path) -> None:
    runtime = build_web_runtime(_settings(tmp_path), user_id="u_web")

    events = list(chat_stream_events(runtime, {"message": "我想做一个抽烟识别算法方案"}))

    labels = [event.get("label") for event in events if event["type"] == "status"]
    assert "正在解析图片" not in labels
    assert "正在检索知识库" in labels
    assert events[-1]["type"] == "result"


def test_web_stream_parses_image_before_agent_flow(tmp_path: Path) -> None:
    runtime = build_web_runtime(_settings(tmp_path), user_id="u_web")

    events = list(
        chat_stream_events(
            runtime,
            {
                "attachments": [
                    {
                        "name": "smoking-scene.png",
                        "type": "image/png",
                        "size": 2048,
                        "data_url": "data:image/png;base64,abc",
                    },
                ]
            },
        )
    )

    labels = [event.get("label") for event in events if event["type"] == "status"]
    assert labels[0] == "正在解析图片"
    assert "正在检索知识库" in labels
    assert events[-1]["type"] == "result"
    assert events[-1]["response"]["thread_id"]


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        db_path=tmp_path / "agent.sqlite",
        checkpoint_db_path=tmp_path / "checkpoints.sqlite",
        model_base_url="https://endpoint.greatrouter.com",
        chat_model="gpt-5.4-nano",
        embed_model="text-embedding-3-small",
        api_key=None,
    )
