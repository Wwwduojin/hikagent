from __future__ import annotations

import json
from typing import Any

import httpx

from agent_solution.config import Settings


class OpenAICompatibleClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        return headers

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.2) -> str:
        payload = {
            "model": self.settings.chat_model,
            "messages": messages,
            "temperature": temperature,
        }
        with httpx.Client(timeout=60) as client:
            response = client.post(
                f"{self.settings.model_base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        return data["choices"][0]["message"]["content"]

    def embed(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": self.settings.embed_model, "input": texts}
        with httpx.Client(timeout=60) as client:
            response = client.post(
                f"{self.settings.model_base_url}/embeddings",
                headers=self._headers(),
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        return [item["embedding"] for item in data["data"]]


GreatRouterClient = OpenAICompatibleClient
VLLMClient = OpenAICompatibleClient


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json\n", "", 1).replace("JSON\n", "", 1)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end >= start:
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)
