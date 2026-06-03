from __future__ import annotations

from pathlib import Path

import pytest

from agent_solution.kb import KnowledgeBase, split_text


class FakeEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            smoke = 1.0 if "抽烟" in text or "手口" in text or "烟雾" in text else 0.0
            helmet = 1.0 if "安全帽" in text else 0.0
            vectors.append([smoke, helmet, 0.1])
        return vectors


def test_split_text_respects_overlap() -> None:
    chunks = split_text("abcdef", chunk_size=4, chunk_overlap=2)
    assert chunks == ["abcd", "cdef"]


def test_keyword_search_uses_like_fallback_for_chinese(tmp_path: Path) -> None:
    kb = KnowledgeBase(tmp_path / "kb.sqlite", embedder=FakeEmbedder())
    kb.add_document(
        "seed:smoking_detection.md",
        "smoking_detection",
        "抽烟识别算法需要判断手口动作关系，并结合烟雾误报过滤。",
    )

    hits = kb.search("抽烟识别", mode="keyword", top_k=3)

    assert hits
    assert hits[0].source == "seed:smoking_detection.md"
    assert "抽烟" in hits[0].content


def test_hybrid_search_merges_vector_and_keyword_scores(tmp_path: Path) -> None:
    kb = KnowledgeBase(tmp_path / "kb.sqlite", embedder=FakeEmbedder())
    kb.add_document("seed:smoking_detection.md", "smoking", "抽烟识别 手口动作 烟雾误报")
    kb.add_document("seed:helmet_detection.md", "helmet", "安全帽佩戴识别")

    hits = kb.search("抽烟识别 手口动作", mode="hybrid", top_k=2)

    assert hits[0].source == "seed:smoking_detection.md"
    assert hits[0].hybrid_score > 0


def test_vector_search_skips_embeddings_with_different_dimensions(tmp_path: Path) -> None:
    kb = KnowledgeBase(tmp_path / "kb.sqlite", embedder=FakeEmbedder())
    kb.add_document("seed:smoking_detection.md", "smoking", "抽烟识别 手口动作 烟雾误报")

    class DifferentDimensionEmbedder:
        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

    kb.embedder = DifferentDimensionEmbedder()

    assert kb.search("抽烟识别", mode="vector", top_k=2) == []


def test_rejects_unsupported_file(tmp_path: Path) -> None:
    file_path = tmp_path / "bad.pdf"
    file_path.write_text("content", encoding="utf-8")
    kb = KnowledgeBase(tmp_path / "kb.sqlite")

    with pytest.raises(ValueError):
        kb.add_file(file_path)
