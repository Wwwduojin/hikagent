from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Protocol

import numpy as np

from agent_solution.config import Settings, ensure_parent
from agent_solution.models import KnowledgeHit


class EmbeddingClient(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class KnowledgeBase:
    def __init__(self, db_path: Path, embedder: EmbeddingClient | None = None):
        self.db_path = db_path
        self.embedder = embedder
        ensure_parent(db_path)
        self._init_schema()

    @classmethod
    def from_settings(cls, settings: Settings, embedder: EmbeddingClient | None = None) -> "KnowledgeBase":
        return cls(settings.db_path, embedder)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL UNIQUE,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding TEXT,
                    FOREIGN KEY(document_id) REFERENCES documents(id)
                )
                """
            )
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
                USING fts5(content, source, content='chunks', content_rowid='id')
                """
            )

    def add_file(self, path: Path, chunk_size: int = 900, chunk_overlap: int = 120) -> int:
        if path.suffix.lower() not in {".md", ".txt"}:
            raise ValueError(f"Only .md and .txt files are supported: {path}")
        content = path.read_text(encoding="utf-8")
        return self.add_document(str(path), path.stem, content, chunk_size, chunk_overlap)

    def add_document(
        self,
        source: str,
        title: str,
        content: str,
        chunk_size: int = 900,
        chunk_overlap: int = 120,
    ) -> int:
        chunks = split_text(content, chunk_size, chunk_overlap)
        embeddings: list[list[float] | None]
        if self.embedder and chunks:
            try:
                embeddings = self.embedder.embed(chunks)
            except Exception:
                embeddings = [None] * len(chunks)
        else:
            embeddings = [None] * len(chunks)
        with self._connect() as conn:
            existing = conn.execute("SELECT id FROM documents WHERE source = ?", (source,)).fetchone()
            if existing:
                document_id = int(existing["id"])
                conn.execute("DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE document_id = ?)", (document_id,))
                conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
                conn.execute("UPDATE documents SET title = ?, content = ? WHERE id = ?", (title, content, document_id))
            else:
                cursor = conn.execute(
                    "INSERT INTO documents(source, title, content) VALUES (?, ?, ?)",
                    (source, title, content),
                )
                document_id = int(cursor.lastrowid)

            for chunk, embedding in zip(chunks, embeddings):
                cursor = conn.execute(
                    "INSERT INTO chunks(document_id, source, content, embedding) VALUES (?, ?, ?, ?)",
                    (document_id, source, chunk, json.dumps(embedding) if embedding else None),
                )
                chunk_id = int(cursor.lastrowid)
                conn.execute(
                    "INSERT INTO chunks_fts(rowid, content, source) VALUES (?, ?, ?)",
                    (chunk_id, chunk, source),
                )
        return document_id

    def keyword_search(self, query: str, top_k: int = 5) -> list[KnowledgeHit]:
        with self._connect() as conn:
            rows = []
            try:
                rows = conn.execute(
                    """
                    SELECT c.id, c.document_id, c.source, c.content, bm25(chunks_fts) AS score
                    FROM chunks_fts
                    JOIN chunks c ON c.id = chunks_fts.rowid
                    WHERE chunks_fts MATCH ?
                    ORDER BY score
                    LIMIT ?
                    """,
                    (query, top_k),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []

            hits = [
                KnowledgeHit(
                    chunk_id=int(row["id"]),
                    document_id=int(row["document_id"]),
                    source=row["source"],
                    content=row["content"],
                    keyword_score=1.0 / (1.0 + abs(float(row["score"]))),
                    hybrid_score=1.0 / (1.0 + abs(float(row["score"]))),
                )
                for row in rows
            ]
            if hits:
                return hits

            like_terms = [term for term in query.split() if term]
            if not like_terms:
                like_terms = [query]
            where = " OR ".join(["content LIKE ?"] * len(like_terms))
            params = [f"%{term}%" for term in like_terms] + [top_k]
            rows = conn.execute(
                f"SELECT id, document_id, source, content FROM chunks WHERE {where} LIMIT ?",
                params,
            ).fetchall()
            return [
                KnowledgeHit(
                    chunk_id=int(row["id"]),
                    document_id=int(row["document_id"]),
                    source=row["source"],
                    content=row["content"],
                    keyword_score=0.6,
                    hybrid_score=0.6,
                )
                for row in rows
            ]

    def vector_search(self, query: str, top_k: int = 5) -> list[KnowledgeHit]:
        if not self.embedder:
            return []
        query_embedding = np.array(self.embedder.embed([query])[0], dtype=np.float32)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, document_id, source, content, embedding FROM chunks WHERE embedding IS NOT NULL"
            ).fetchall()
        scored = []
        for row in rows:
            embedding = np.array(json.loads(row["embedding"]), dtype=np.float32)
            if query_embedding.shape != embedding.shape:
                continue
            score = cosine_similarity(query_embedding, embedding)
            scored.append(
                KnowledgeHit(
                    chunk_id=int(row["id"]),
                    document_id=int(row["document_id"]),
                    source=row["source"],
                    content=row["content"],
                    vector_score=score,
                    hybrid_score=score,
                )
            )
        return sorted(scored, key=lambda hit: hit.vector_score, reverse=True)[:top_k]

    def hybrid_search(self, query: str, top_k: int = 5) -> list[KnowledgeHit]:
        merged: dict[int, KnowledgeHit] = {}
        for hit in self.keyword_search(query, top_k=top_k * 2):
            merged[hit.chunk_id] = hit
        for hit in self.vector_search(query, top_k=top_k * 2):
            existing = merged.get(hit.chunk_id)
            if existing:
                existing.vector_score = hit.vector_score
                existing.hybrid_score = 0.55 * hit.vector_score + 0.45 * existing.keyword_score
            else:
                hit.hybrid_score = 0.55 * hit.vector_score
                merged[hit.chunk_id] = hit
        return sorted(merged.values(), key=lambda item: item.hybrid_score, reverse=True)[:top_k]

    def search(self, query: str, mode: str = "hybrid", top_k: int = 5) -> list[KnowledgeHit]:
        if mode == "keyword":
            return self.keyword_search(query, top_k)
        if mode == "vector":
            return self.vector_search(query, top_k)
        if mode == "hybrid":
            return self.hybrid_search(query, top_k)
        raise ValueError("mode must be one of: keyword, vector, hybrid")


def split_text(text: str, chunk_size: int = 900, chunk_overlap: int = 120) -> list[str]:
    normalized = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    if not normalized:
        return []
    chunks = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(normalized):
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if math.isclose(denom, 0.0):
        return 0.0
    return float(np.dot(left, right) / denom)
