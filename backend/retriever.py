"""Semantic retrieval service backed by the persistent Chroma collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from backend.config import Settings


@dataclass(frozen=True)
class RetrievedSource:
    """A retrieved document and its searchable metadata."""

    document: str
    metadata: dict[str, Any]
    distance: float


def select_diverse_sources(
    sources: list[RetrievedSource], top_k: int, max_per_group: int = 2
) -> list[RetrievedSource]:
    """Reduce repetitive evidence while preserving nearest-result ordering."""
    selected: list[RetrievedSource] = []
    group_counts: dict[tuple[str, str, str], int] = {}
    deferred: list[RetrievedSource] = []
    for source in sources:
        metadata = source.metadata
        group = (
            str(metadata.get("region", "Unknown")),
            str(metadata.get("operator", "Unknown")),
            str(metadata.get("call_category", "Unknown")),
        )
        if group_counts.get(group, 0) < max_per_group:
            selected.append(source)
            group_counts[group] = group_counts.get(group, 0) + 1
        else:
            deferred.append(source)
        if len(selected) == top_k:
            return selected
    selected.extend(deferred[: top_k - len(selected)])
    return selected


class ChromaRetriever:
    """Load the embedding model and query the configured Chroma collection."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = chromadb.PersistentClient(path=str(settings.chroma_path))
        self.collection = self.client.get_collection(settings.chroma_collection)
        self.model = SentenceTransformer(settings.embedding_model, device="cpu")

    @property
    def count(self) -> int:
        """Return the number of indexed records."""
        return self.collection.count()

    def retrieve_logs(self, query: str, top_k: int) -> list[RetrievedSource]:
        """Return the nearest indexed telecom records for a natural-language query."""
        embedding = self.model.encode([query], normalize_embeddings=True).tolist()
        candidate_count = min(max(top_k * 4, top_k), 50)
        result = self.collection.query(
            query_embeddings=embedding,
            n_results=candidate_count,
            include=["documents", "metadatas", "distances"],
        )
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        candidates = [
            RetrievedSource(
                document=str(document),
                metadata=dict(metadata or {}),
                distance=float(distance),
            )
            for document, metadata, distance in zip(documents, metadatas, distances)
        ]
        return select_diverse_sources(candidates, top_k)
