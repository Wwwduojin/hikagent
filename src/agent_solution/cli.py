from __future__ import annotations

import argparse
import sys
from importlib import resources
from pathlib import Path

from agent_solution.config import Settings
from agent_solution.algorithm_planner import build_algorithm_planner
from agent_solution.kb import KnowledgeBase
from agent_solution.llm import OpenAICompatibleClient
from agent_solution.service import AgentSolutionService, create_sqlite_checkpointer
from agent_solution.simulation import SMOKING_DEMO_MESSAGES, SimulatedLLMClient
from agent_solution.storage import BusinessStore
from agent_solution.tools import build_default_registry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent-solution")
    subparsers = parser.add_subparsers(dest="command", required=True)

    chat_parser = subparsers.add_parser("chat")
    chat_parser.add_argument(
        "--simulate",
        action="store_true",
        help="Use deterministic local simulation instead of the configured model API.",
    )
    chat_parser.add_argument("--user-id", default="u_001", help="User identifier used for profile and session isolation.")
    chat_parser.add_argument("--thread-id", default=None, help="Resume an existing application thread.")

    demo_parser = subparsers.add_parser("demo")
    demo_parser.add_argument("demo_name", choices=["smoking"])

    kb_parser = subparsers.add_parser("kb")
    kb_subparsers = kb_parser.add_subparsers(dest="kb_command", required=True)
    kb_subparsers.add_parser("seed")
    kb_add = kb_subparsers.add_parser("add")
    kb_add.add_argument("path")
    kb_search = kb_subparsers.add_parser("search")
    kb_search.add_argument("query")
    kb_search.add_argument("--mode", choices=["keyword", "vector", "hybrid"], default="hybrid")
    kb_search.add_argument("--top-k", type=int, default=5)

    tools_parser = subparsers.add_parser("tools")
    tools_parser.add_argument("tools_command", choices=["list"])

    args = parser.parse_args(argv)
    settings = Settings.from_env()
    llm = (
        SimulatedLLMClient()
        if getattr(args, "simulate", False) or args.command == "demo"
        else OpenAICompatibleClient(settings)
    )
    kb = KnowledgeBase.from_settings(settings, embedder=llm)
    tools = build_default_registry()
    planner = build_algorithm_planner(settings.plan_mode, llm)

    if args.command == "chat":
        return run_chat(settings, llm, kb, tools, planner, args.user_id, args.thread_id)
    if args.command == "demo":
        return run_demo(args.demo_name, settings, llm, kb, tools)
    if args.command == "kb":
        return run_kb(args, settings, kb)
    if args.command == "tools":
        for tool in tools.list_tools():
            print(f"{tool.name}\t{tool.description}\tpermission={tool.permission}")
        return 0
    return 1


def run_chat(settings: Settings, llm: OpenAICompatibleClient, kb: KnowledgeBase, tools, planner, user_id: str, thread_id: str | None) -> int:
    store = BusinessStore(settings.db_path)
    checkpointer = create_sqlite_checkpointer(str(settings.checkpoint_db_path))
    service = AgentSolutionService(llm, kb, tools, store, planner, checkpointer=checkpointer)
    current_thread_id = thread_id
    print("Agent Solution chat started. Type /sessions, /use <thread_id>, or /exit.")
    while True:
        try:
            user_text = input("\nUser> ").strip()
        except EOFError:
            break
        if user_text in {"/exit", "exit", "quit"}:
            break
        if user_text == "/sessions":
            sessions = service.list_sessions(user_id)
            for session in sessions:
                marker = "*" if session.thread_id == current_thread_id else " "
                print(
                    f"{marker} {session.thread_id}\t{session.app_name or session.session_type}"
                    f"\tstage={session.stage}\tstatus={session.status}"
                )
            continue
        if user_text.startswith("/use "):
            target = user_text.removeprefix("/use ").strip()
            session = service.get_session(user_id, target)
            if not session:
                print("Session not found for this user.")
            else:
                current_thread_id = session.thread_id
                print(f"Switched to {session.app_name or session.session_type}: {session.thread_id}")
            continue
        if not user_text:
            continue
        response = service.chat(user_id, user_text, current_thread_id)
        current_thread_id = response.thread_id or current_thread_id
        print(
            f"\n[{response.current_app_name or response.session_type}] "
            f"stage={response.stage} status={response.status} thread_id={response.thread_id}"
        )
        print(f"\nAssistant> {response.message}")
    return 0


def run_demo(demo_name: str, settings: Settings, llm, kb: KnowledgeBase, tools) -> int:
    if demo_name != "smoking":
        raise ValueError(f"Unsupported demo: {demo_name}")
    import_seed_documents(settings, kb)

    from agent_solution.graph import append_user_message, build_graph, initial_state

    graph = build_graph(llm, kb, tools)
    state = initial_state()
    print("Running smoking detection simulation demo.\n")
    for user_text in SMOKING_DEMO_MESSAGES:
        print(f"User> {user_text}")
        state = append_user_message(state, user_text)
        state = graph.invoke(state)
        print(f"Assistant> {state.get('assistant_reply')}\n")
    print(f"Final status: {state.get('status')}")
    return 0


def run_kb(args, settings: Settings, kb: KnowledgeBase) -> int:
    if args.kb_command == "seed":
        count = import_seed_documents(settings, kb)
        print(f"Imported {count} seed documents into {settings.db_path}")
        return 0
    if args.kb_command == "add":
        path = Path(args.path)
        if path.is_dir():
            files = [item for item in path.rglob("*") if item.suffix.lower() in {".md", ".txt"}]
        else:
            files = [path]
        for file_path in files:
            kb.add_file(file_path, settings.chunk_size, settings.chunk_overlap)
        print(f"Imported {len(files)} document(s)")
        return 0
    if args.kb_command == "search":
        hits = kb.search(args.query, mode=args.mode, top_k=args.top_k)
        for idx, hit in enumerate(hits, 1):
            print(f"\n[{idx}] {hit.source}")
            print(f"score={hit.hybrid_score:.4f} vector={hit.vector_score:.4f} keyword={hit.keyword_score:.4f}")
            print(hit.content[:500])
        return 0
    return 1


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


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
