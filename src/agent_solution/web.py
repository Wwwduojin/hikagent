from __future__ import annotations

import ast
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any
from urllib.parse import urlparse

from agent_solution.algorithm_planner import build_algorithm_planner
from agent_solution.config import Settings
from agent_solution.kb import KnowledgeBase
from agent_solution.llm import OpenAICompatibleClient
from agent_solution.models import AppSession, ChatResponse
from agent_solution.service import AgentSolutionService, create_sqlite_checkpointer
from agent_solution.simulation import SimulatedLLMClient
from agent_solution.storage import BusinessStore
from agent_solution.tools import build_default_registry


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7860


class WebRuntime:
    def __init__(self, service: AgentSolutionService, user_id: str, current_thread_id: str | None = None):
        self.service = service
        self.user_id = user_id
        self.current_thread_id = current_thread_id


def build_web_runtime(
    settings: Settings,
    *,
    user_id: str = "u_001",
    thread_id: str | None = None,
    real_model: bool = False,
) -> WebRuntime:
    llm = OpenAICompatibleClient(settings) if real_model else SimulatedLLMClient()
    kb = KnowledgeBase.from_settings(settings, embedder=llm)
    import_seed_documents(settings, kb)
    store = BusinessStore(settings.db_path)
    planner = build_algorithm_planner(settings.plan_mode, llm)
    service = AgentSolutionService(
        llm,
        kb,
        build_default_registry(),
        store,
        planner,
        checkpointer=create_sqlite_checkpointer(str(settings.checkpoint_db_path)),
    )
    return WebRuntime(service, user_id=user_id, current_thread_id=thread_id)


def run_web_server(
    settings: Settings,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    user_id: str = "u_001",
    thread_id: str | None = None,
    real_model: bool = False,
) -> int:
    runtime = build_web_runtime(settings, user_id=user_id, thread_id=thread_id, real_model=real_model)
    handler = make_handler(runtime)
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Agent Solution web demo is running at http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped Agent Solution web demo.")
    finally:
        server.server_close()
    return 0


def chat_payload(runtime: WebRuntime, payload: dict[str, Any]) -> dict[str, Any]:
    result = None
    for event in chat_stream_events(runtime, payload):
        if event["type"] == "error":
            return {"error": event["message"]}
        if event["type"] == "result":
            result = event
    if result is None:
        return {"error": "chat did not produce a response"}
    return {
        "response": result["response"],
        "sessions": result["sessions"],
        "current_thread_id": result["current_thread_id"],
    }


def chat_stream_events(runtime: WebRuntime, payload: dict[str, Any]):
    message = str(payload.get("message") or "").strip()
    attachments = _attachment_summaries(payload.get("attachments", []))
    if not message and not attachments:
        yield {"type": "error", "message": "message or image attachment is required"}
        return
    if attachments:
        yield {"type": "status", "label": "正在解析图片", "active_agent": "vision"}
        try:
            image_description = runtime.service.llm.describe_images(attachments, user_text=message)
        except Exception as exc:
            yield {"type": "error", "message": f"图片解析失败：{exc}"}
            return
        for attachment in attachments:
            attachment["description"] = image_description
    user_message = _message_with_attachments(message, attachments)
    thread_id = str(payload.get("thread_id") or runtime.current_thread_id or "").strip() or None
    for event in runtime.service.stream_chat(runtime.user_id, user_message, thread_id):
        if event["type"] == "result":
            response = ChatResponse.model_validate(event["response"])
            runtime.current_thread_id = response.thread_id or runtime.current_thread_id
            yield {
                "type": "result",
                "response": _chat_response_dict(response),
                "sessions": [_session_dict(session) for session in runtime.service.list_sessions(runtime.user_id)],
                "current_thread_id": runtime.current_thread_id,
            }
        else:
            yield event


def _attachment_summaries(raw_attachments: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_attachments, list):
        return []
    attachments = []
    for item in raw_attachments[:4]:
        if not isinstance(item, dict):
            continue
        mime_type = str(item.get("type") or "")
        if not mime_type.startswith("image/"):
            continue
        attachments.append(
            {
                "name": str(item.get("name") or "未命名图片"),
                "type": mime_type,
                "size": int(item.get("size") or 0),
                "data_url": str(item.get("data_url") or item.get("preview") or ""),
            }
        )
    return attachments


def _message_with_attachments(message: str, attachments: list[dict[str, Any]]) -> str:
    if not attachments:
        return message
    lines = [message] if message else ["用户上传了图片，请结合图片附件理解需求。"]
    lines.append("")
    lines.append("图片附件：")
    for index, item in enumerate(attachments, 1):
        size_kb = item["size"] / 1024 if item["size"] else 0
        lines.append(f"{index}. {item['name']}（{item['type']}，{size_kb:.1f} KB）")
    descriptions = [str(item.get("description") or "").strip() for item in attachments if item.get("description")]
    if descriptions:
        lines.append("")
        lines.append("图片解析结果：")
        for index, description in enumerate(descriptions, 1):
            lines.append(f"{index}. {description}")
    lines.append("请根据用户文字和图片解析结果判断算法应用需求。")
    return "\n".join(lines)


def sessions_payload(runtime: WebRuntime) -> dict[str, Any]:
    sessions = runtime.service.list_sessions(runtime.user_id)
    return {
        "sessions": [_session_dict(session) for session in sessions],
        "current_thread_id": runtime.current_thread_id,
        "user_id": runtime.user_id,
    }


def use_session_payload(runtime: WebRuntime, payload: dict[str, Any]) -> dict[str, Any]:
    thread_id = str(payload.get("thread_id") or "").strip()
    if not thread_id:
        return {"error": "thread_id is required"}
    session = runtime.service.get_session(runtime.user_id, thread_id)
    if not session:
        return {"error": "session not found"}
    runtime.current_thread_id = thread_id
    return {
        "session": _session_dict(session),
        "messages": session_messages(runtime, session),
        "sessions": [_session_dict(item) for item in runtime.service.list_sessions(runtime.user_id)],
        "current_thread_id": runtime.current_thread_id,
    }


def session_messages(runtime: WebRuntime, session: AppSession) -> list[dict[str, str]]:
    state = runtime.service.store.load_state(session.session_id) or {}
    messages = [_coerce_history_message(message, index) for index, message in enumerate(state.get("messages", []))]
    return [message for message in messages if message["content"].strip()]


def _coerce_history_message(message: Any, index: int) -> dict[str, str]:
    if isinstance(message, dict):
        role = str(message.get("role") or message.get("type") or _role_from_index(index))
        return {"role": _normalize_role(role), "content": str(message.get("content") or "")}
    if hasattr(message, "content"):
        role = str(getattr(message, "type", _role_from_index(index)))
        return {"role": _normalize_role(role), "content": str(getattr(message, "content", ""))}
    text = str(message)
    return {"role": _role_from_repr(text, index), "content": _content_from_repr(text)}


def _normalize_role(role: str) -> str:
    return {"human": "user", "ai": "assistant"}.get(role, role if role in {"user", "assistant", "system"} else "user")


def _role_from_repr(text: str, index: int) -> str:
    if "tool_calls=" in text or "invalid_tool_calls=" in text:
        return "assistant"
    return _role_from_index(index)


def _role_from_index(index: int) -> str:
    return "user" if index % 2 == 0 else "assistant"


def _content_from_repr(text: str) -> str:
    match = re.search(r"content=(?P<value>(?:'[^']*(?:\\.[^']*)*'|\"[^\"]*(?:\\.[^\"]*)*\"))", text, re.DOTALL)
    if not match:
        return text
    try:
        return str(ast.literal_eval(match.group("value")))
    except (SyntaxError, ValueError):
        return text


def make_handler(runtime: WebRuntime) -> type[BaseHTTPRequestHandler]:
    class AgentSolutionWebHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                self._send_html(INDEX_HTML)
                return
            if path == "/api/sessions":
                self._send_json(sessions_payload(runtime))
                return
            self._send_json({"error": "not found"}, status=404)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            payload = self._read_json()
            if path == "/api/chat/stream":
                self._send_event_stream(chat_stream_events(runtime, payload))
                return
            if path == "/api/chat":
                result = chat_payload(runtime, payload)
                self._send_json(result, status=400 if "error" in result else 200)
                return
            if path == "/api/use":
                result = use_session_payload(runtime, payload)
                self._send_json(result, status=404 if result.get("error") == "session not found" else 400 if "error" in result else 200)
                return
            self._send_json({"error": "not found"}, status=404)

        def log_message(self, format: str, *args: Any) -> None:
            print(f"{self.address_string()} - {format % args}")

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length <= 0:
                return {}
            raw = self.rfile.read(length).decode("utf-8")
            return json.loads(raw) if raw else {}

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_event_stream(self, events) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            for event in events:
                data = json.dumps(event, ensure_ascii=False)
                self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                self.wfile.flush()

        def _send_html(self, html: str) -> None:
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return AgentSolutionWebHandler


def import_seed_documents(settings: Settings, kb: KnowledgeBase) -> int:
    count = 0
    seed_root = resources.files("agent_solution").joinpath("knowledge_base/seeds")
    for seed in seed_root.iterdir():
        if seed.name.endswith(".md"):
            kb.add_document(
                source=f"seed:{seed.name}",
                title=seed.name.removesuffix(".md"),
                content=seed.read_text(encoding="utf-8"),
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
            count += 1
    return count


def _chat_response_dict(response: ChatResponse) -> dict[str, Any]:
    return response.model_dump()


def _session_dict(session: AppSession) -> dict[str, Any]:
    return session.model_dump()


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Agent Solution Demo</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7fb;
      --panel: #ffffff;
      --line: #d8dee9;
      --text: #172033;
      --muted: #647084;
      --accent: #1677ff;
      --accent-strong: #0f5fd0;
      --ok: #1b8f5a;
      --shadow: 0 10px 30px rgba(23, 32, 51, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .app {
      display: grid;
      grid-template-columns: 300px minmax(0, 1fr);
      min-height: 100vh;
    }
    aside {
      border-right: 1px solid var(--line);
      background: #eef2f8;
      padding: 18px;
    }
    main {
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
      min-height: 100vh;
    }
    .brand {
      font-size: 18px;
      font-weight: 700;
      margin-bottom: 4px;
    }
    .subtle { color: var(--muted); font-size: 12px; }
    .new-chat {
      width: 100%;
      margin-top: 14px;
      border: 1px solid var(--accent);
      border-radius: 8px;
      background: #fff;
      color: var(--accent);
      cursor: pointer;
      font-weight: 650;
      padding: 10px 12px;
      text-align: center;
    }
    .new-chat:hover {
      background: #e9f2ff;
    }
    .toolbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      padding: 14px 18px;
    }
    .status {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      justify-content: flex-end;
    }
    .pill {
      border: 1px solid var(--line);
      border-radius: 999px;
      color: var(--muted);
      padding: 4px 9px;
      white-space: nowrap;
      background: #fff;
      font-size: 12px;
    }
    .sessions {
      display: grid;
      gap: 8px;
      margin-top: 18px;
    }
    .session {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      color: var(--text);
      cursor: pointer;
      padding: 10px;
      text-align: left;
    }
    .session.active {
      border-color: var(--accent);
      box-shadow: 0 0 0 2px rgba(22, 119, 255, 0.12);
    }
    .session-title { font-weight: 650; }
    .session-meta { color: var(--muted); font-size: 12px; margin-top: 4px; word-break: break-all; }
    .messages {
      overflow: auto;
      padding: 22px;
    }
    .message {
      max-width: 860px;
      margin: 0 auto 14px;
      display: grid;
      gap: 6px;
    }
    .bubble {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 14px;
      background: var(--panel);
      box-shadow: var(--shadow);
      white-space: pre-wrap;
    }
    .message.user .bubble {
      margin-left: auto;
      max-width: 70%;
      background: #e9f2ff;
      border-color: #c7dcff;
      box-shadow: none;
    }
    .meta {
      color: var(--muted);
      font-size: 12px;
    }
    .bubble.thinking {
      color: var(--muted);
      background: #fffdf5;
      border-color: #ead89a;
      box-shadow: none;
    }
    .dots::after {
      content: "";
      animation: dots 1.2s steps(4, end) infinite;
    }
    @keyframes dots {
      0% { content: ""; }
      25% { content: "."; }
      50% { content: ".."; }
      75%, 100% { content: "..."; }
    }
    .composer {
      border-top: 1px solid var(--line);
      background: var(--panel);
      padding: 16px 18px;
    }
    .composer-inner {
      display: grid;
      grid-template-columns: 42px minmax(0, 1fr) auto;
      gap: 8px;
      max-width: 980px;
      margin: 0 auto;
      align-items: center;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 8px;
      box-shadow: 0 8px 22px rgba(23, 32, 51, 0.06);
      transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }
    .composer-inner:focus-within {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(22, 119, 255, 0.12);
    }
    textarea {
      min-height: 38px;
      max-height: 132px;
      resize: none;
      border: 0;
      border-radius: 6px;
      padding: 8px 10px;
      color: var(--text);
      font: inherit;
      outline: none;
      background: transparent;
      line-height: 1.45;
    }
    textarea::placeholder { color: #8a93a3; }
    button.primary {
      border: 0;
      border-radius: 6px;
      background: var(--accent);
      color: white;
      font-weight: 650;
      padding: 0 16px;
      min-width: 78px;
      height: 38px;
      cursor: pointer;
    }
    button.primary:hover { background: var(--accent-strong); }
    .icon-button {
      width: 38px;
      height: 38px;
      border: 0;
      border-radius: 6px;
      background: #f0f4fa;
      color: #3d4a5f;
      cursor: pointer;
      font-size: 22px;
      line-height: 1;
    }
    .icon-button:hover { background: #e4efff; color: var(--accent); }
    button:disabled { cursor: not-allowed; opacity: 0.6; }
    .attachments {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      max-width: 980px;
      margin: 0 auto 10px;
    }
    .attachment {
      display: flex;
      align-items: center;
      gap: 8px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 6px 8px;
      max-width: 260px;
    }
    .attachment img {
      width: 36px;
      height: 36px;
      object-fit: cover;
      border-radius: 6px;
      background: #eef2f8;
    }
    .attachment-name {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      color: var(--muted);
      font-size: 12px;
    }
    .attachment-remove {
      border: 0;
      background: transparent;
      color: var(--muted);
      cursor: pointer;
      font-size: 16px;
      line-height: 1;
    }
    .empty {
      color: var(--muted);
      text-align: center;
      margin-top: 22vh;
    }
    @media (max-width: 780px) {
      .app { grid-template-columns: 1fr; }
      aside {
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }
      main { min-height: calc(100vh - 210px); }
      .message.user .bubble { max-width: 92%; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <div class="brand">Agent Solution</div>
      <div class="subtle">本地演示页面</div>
      <button id="newChatButton" class="new-chat" type="button">新对话</button>
      <div id="sessions" class="sessions"></div>
    </aside>
    <main>
      <div class="toolbar">
        <div>
          <strong id="currentApp">未选择应用</strong>
          <div class="subtle" id="threadLine">等待开始对话</div>
        </div>
        <div class="status">
          <span class="pill" id="stagePill">stage: -</span>
          <span class="pill" id="statusPill">status: -</span>
        </div>
      </div>
      <section id="messages" class="messages">
        <div class="empty">输入一句算法需求开始本地演示。</div>
      </section>
      <form id="composer" class="composer">
        <div id="attachments" class="attachments"></div>
        <div class="composer-inner">
          <button id="attachButton" class="icon-button" type="button" title="上传图片">+</button>
          <input id="imageInput" type="file" accept="image/*" multiple hidden />
          <textarea id="messageInput" placeholder="例如：我想做抽烟识别算法方案"></textarea>
          <button id="sendButton" class="primary" type="submit">发送</button>
        </div>
      </form>
    </main>
  </div>
  <script>
    const state = {
      currentThreadId: null,
      sessions: [],
      messages: [],
      attachments: []
    };

    const sessionsEl = document.getElementById("sessions");
    const messagesEl = document.getElementById("messages");
    const attachmentsEl = document.getElementById("attachments");
    const inputEl = document.getElementById("messageInput");
    const imageInputEl = document.getElementById("imageInput");
    const attachButton = document.getElementById("attachButton");
    const newChatButton = document.getElementById("newChatButton");
    const sendButton = document.getElementById("sendButton");
    const currentAppEl = document.getElementById("currentApp");
    const threadLineEl = document.getElementById("threadLine");
    const stagePillEl = document.getElementById("stagePill");
    const statusPillEl = document.getElementById("statusPill");

    async function api(path, options = {}) {
      const response = await fetch(path, {
        headers: { "Content-Type": "application/json" },
        ...options
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "请求失败");
      return data;
    }

    function renderSessions() {
      sessionsEl.innerHTML = "";
      if (!state.sessions.length) {
        const empty = document.createElement("div");
        empty.className = "subtle";
        empty.textContent = "暂无会话";
        sessionsEl.appendChild(empty);
        return;
      }
      for (const session of state.sessions) {
        const button = document.createElement("button");
        button.className = "session" + (session.thread_id === state.currentThreadId ? " active" : "");
        button.type = "button";
        button.onclick = () => useSession(session.thread_id);
        button.innerHTML = `
          <div class="session-title">${escapeHtml(session.app_name || session.session_type || "未命名应用")}</div>
          <div class="session-meta">${escapeHtml(session.stage)} · ${escapeHtml(session.status)}</div>
          <div class="session-meta">${escapeHtml(session.thread_id)}</div>
        `;
        sessionsEl.appendChild(button);
      }
    }

    function renderMessages() {
      messagesEl.innerHTML = "";
      if (!state.messages.length) {
        const empty = document.createElement("div");
        empty.className = "empty";
        empty.textContent = "输入一句算法需求开始本地演示。";
        messagesEl.appendChild(empty);
        return;
      }
      for (const message of state.messages) {
        const wrapper = document.createElement("div");
        wrapper.className = `message ${message.role}`;
        const meta = message.meta ? `<div class="meta">${escapeHtml(message.meta)}</div>` : "";
        const thinkingClass = message.loading ? " thinking" : "";
        const dotsClass = message.loading ? " dots" : "";
        wrapper.innerHTML = `<div class="bubble${thinkingClass}"><span class="${dotsClass}">${escapeHtml(message.content)}</span></div>${meta}`;
        messagesEl.appendChild(wrapper);
      }
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function renderAttachments() {
      attachmentsEl.innerHTML = "";
      for (const attachment of state.attachments) {
        const item = document.createElement("div");
        item.className = "attachment";
        item.innerHTML = `
          <img src="${attachment.preview}" alt="">
          <div class="attachment-name" title="${escapeHtml(attachment.name)}">${escapeHtml(attachment.name)}</div>
          <button class="attachment-remove" type="button" aria-label="移除图片">x</button>
        `;
        item.querySelector("button").onclick = () => {
          state.attachments = state.attachments.filter(existing => existing.id !== attachment.id);
          renderAttachments();
        };
        attachmentsEl.appendChild(item);
      }
    }

    function updateHeader(response) {
      if (!response) return;
      currentAppEl.textContent = response.current_app_name || response.app_name || response.session_type || "未选择应用";
      threadLineEl.textContent = response.thread_id ? `thread: ${response.thread_id}` : "等待开始对话";
      stagePillEl.textContent = `stage: ${response.stage || "-"}`;
      statusPillEl.textContent = `status: ${response.status || "-"}`;
    }

    async function sendMessage(event) {
      event.preventDefault();
      const text = inputEl.value.trim();
      if (!text && !state.attachments.length) return;
      const attachments = state.attachments.map(({ name, type, size, preview }) => ({ name, type, size, data_url: preview }));
      const attachmentLabel = attachments.length
        ? `\n\n[已上传图片：${attachments.map(item => item.name).join("、")}]`
        : "";
      state.messages.push({ role: "user", content: `${text || "上传图片"}${attachmentLabel}` });
      const loadingMessage = {
        role: "assistant",
        content: "正在识别意图",
        loading: true,
        meta: "真实模型响应可能需要一些时间"
      };
      state.messages.push(loadingMessage);
      inputEl.value = "";
      resizeInput();
      state.attachments = [];
      renderAttachments();
      renderMessages();
      sendButton.disabled = true;
      sendButton.textContent = "处理中";
      try {
        await apiStream("/api/chat/stream", { message: text, attachments, thread_id: state.currentThreadId }, event => {
          if (event.type === "status") {
            loadingMessage.content = event.label || "正在处理";
            loadingMessage.meta = [event.active_agent, event.stage].filter(Boolean).join(" · ") || "真实节点状态";
            renderMessages();
          }
          if (event.type === "result") {
            state.currentThreadId = event.current_thread_id;
            state.sessions = event.sessions || [];
            const response = event.response;
            updateHeader(response);
            loadingMessage.content = response.message || "";
            loadingMessage.loading = false;
            loadingMessage.meta = `${response.current_app_name || response.session_type || "应用"} · stage=${response.stage || "-"} · status=${response.status || "-"}`;
            renderSessions();
            renderMessages();
          }
          if (event.type === "error") {
            throw new Error(event.message || "请求失败");
          }
        });
      } catch (error) {
        loadingMessage.content = `请求失败：${error.message}`;
        loadingMessage.loading = false;
        loadingMessage.meta = "";
        renderMessages();
      } finally {
        sendButton.disabled = false;
        sendButton.textContent = "发送";
        inputEl.focus();
      }
    }

    async function apiStream(path, payload, onEvent) {
      const response = await fetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      if (!response.ok || !response.body) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.error || "请求失败");
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop() || "";
        for (const part of parts) {
          const line = part.split("\n").find(item => item.startsWith("data: "));
          if (!line) continue;
          onEvent(JSON.parse(line.slice(6)));
        }
      }
    }

    async function addImageFiles(files) {
      const selected = Array.from(files).filter(file => file.type.startsWith("image/")).slice(0, 4);
      const loaded = await Promise.all(selected.map(file => readImageAttachment(file)));
      state.attachments = [...state.attachments, ...loaded].slice(0, 4);
      renderAttachments();
      imageInputEl.value = "";
    }

    function readImageAttachment(file) {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve({
          id: `${Date.now()}-${Math.random()}`,
          name: file.name,
          type: file.type,
          size: file.size,
          preview: String(reader.result)
        });
        reader.onerror = () => reject(reader.error);
        reader.readAsDataURL(file);
      });
    }

    async function useSession(threadId) {
      try {
        const data = await api("/api/use", {
          method: "POST",
          body: JSON.stringify({ thread_id: threadId })
        });
        state.currentThreadId = data.current_thread_id;
        state.sessions = data.sessions || [];
        state.messages = data.messages || [];
        updateHeader(data.session);
        renderSessions();
        renderMessages();
      } catch (error) {
        state.messages.push({ role: "assistant", content: `切换失败：${error.message}` });
        renderMessages();
      }
    }

    function startNewChat() {
      state.currentThreadId = null;
      state.messages = [];
      state.attachments = [];
      inputEl.value = "";
      resizeInput();
      renderAttachments();
      renderSessions();
      renderMessages();
      currentAppEl.textContent = "新对话";
      threadLineEl.textContent = "发送第一条消息后创建 thread";
      stagePillEl.textContent = "stage: -";
      statusPillEl.textContent = "status: -";
      inputEl.focus();
    }

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    async function boot() {
      const data = await api("/api/sessions");
      state.currentThreadId = data.current_thread_id;
      state.sessions = data.sessions || [];
      renderSessions();
    }

    document.getElementById("composer").addEventListener("submit", sendMessage);
    inputEl.addEventListener("input", resizeInput);
    newChatButton.addEventListener("click", startNewChat);
    attachButton.addEventListener("click", () => imageInputEl.click());
    imageInputEl.addEventListener("change", event => addImageFiles(event.target.files).catch(error => {
      state.messages.push({ role: "assistant", content: `图片读取失败：${error.message}` });
      renderMessages();
    }));
    inputEl.addEventListener("keydown", event => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage(event);
      }
    });
    function resizeInput() {
      inputEl.style.height = "auto";
      inputEl.style.height = `${Math.min(inputEl.scrollHeight, 132)}px`;
    }
    boot().catch(error => {
      state.messages.push({ role: "assistant", content: `初始化失败：${error.message}` });
      renderMessages();
    });
  </script>
</body>
</html>
"""
