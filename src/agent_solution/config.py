from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    db_path: Path
    model_base_url: str
    chat_model: str
    embed_model: str
    vision_model: str | None = None
    api_key: str | None = None
    checkpoint_db_path: Path = Path(".agent_solution") / "checkpoints.sqlite"
    chunk_size: int = 900
    chunk_overlap: int = 120
    plan_mode: str = "simulate"

    @property
    def vllm_base_url(self) -> str:
        return self.model_base_url

    @property
    def vllm_chat_model(self) -> str:
        return self.chat_model

    @property
    def vllm_embed_model(self) -> str:
        return self.embed_model

    @property
    def vllm_vision_model(self) -> str:
        return self.vision_model or self.chat_model

    @property
    def vllm_api_key(self) -> str | None:
        return self.api_key

    @classmethod
    def from_env(cls) -> "Settings":
        default_db = Path(".agent_solution") / "agent_solution.sqlite"
        default_checkpoint_db = Path(".agent_solution") / "checkpoints.sqlite"
        api_key = (
            os.getenv("GREATROUTER_API_KEY")
            or os.getenv("GREATEROUTE_API_KEY")
            or os.getenv("VLLM_API_KEY")
            or None
        )
        return cls(
            db_path=Path(os.getenv("AGENT_SOLUTION_DB", default_db)),
            checkpoint_db_path=Path(os.getenv("AGENT_SOLUTION_CHECKPOINT_DB", default_checkpoint_db)),
            model_base_url=(
                os.getenv("GREATROUTER_BASE_URL")
                or os.getenv("GREATEROUTE_BASE_URL")
                or os.getenv("VLLM_BASE_URL")
                or "https://endpoint.greatrouter.com"
            ).rstrip("/"),
            chat_model=(
                os.getenv("GREATROUTER_CHAT_MODEL")
                or os.getenv("GREATEROUTE_CHAT_MODEL")
                or os.getenv("VLLM_CHAT_MODEL")
                or "gpt-5.4-nano"
            ),
            embed_model=(
                os.getenv("GREATROUTER_EMBED_MODEL")
                or os.getenv("GREATEROUTE_EMBED_MODEL")
                or os.getenv("VLLM_EMBED_MODEL")
                or "text-embedding-3-small"
            ),
            vision_model=(
                os.getenv("GREATROUTER_VISION_MODEL")
                or os.getenv("GREATEROUTE_VISION_MODEL")
                or os.getenv("VLLM_VISION_MODEL")
                or None
            ),
            api_key=api_key,
            chunk_size=int(os.getenv("AGENT_SOLUTION_CHUNK_SIZE", "900")),
            chunk_overlap=int(os.getenv("AGENT_SOLUTION_CHUNK_OVERLAP", "120")),
            plan_mode=os.getenv("AGENT_SOLUTION_PLAN_MODE", "simulate").lower(),
        )


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
