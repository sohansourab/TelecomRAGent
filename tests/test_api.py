"""Focused API contract tests using an in-memory fake retriever."""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from backend.main import create_app
from backend.retriever import RetrievedSource


class FakeRetriever:
    """Small deterministic retriever for API tests."""

    def __init__(self, settings: object) -> None:
        self.settings = settings

    def retrieve_logs(self, query: str, top_k: int) -> list[RetrievedSource]:
        return [
            RetrievedSource(
                document=f"Matched query: {query}",
                metadata={"region": "Uttar Pradesh", "severity": "critical"},
                distance=0.2,
            )
        ][:top_k]


class ApiContractTests(unittest.TestCase):
    """Verify health, validation, and query response contracts."""

    def setUp(self) -> None:
        self.client = TestClient(create_app(retriever_factory=FakeRetriever))

    def test_health(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_query_returns_sources_and_trace(self) -> None:
        response = self.client.post("/query", json={"question": "call drops in UP"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(
            payload["tool_trace"],
            [
                "retrieve_logs",
                "analyze_root_cause",
                "generate_resolution",
                "generate_report",
            ],
        )
        self.assertEqual(payload["sources"][0]["metadata"]["region"], "Uttar Pradesh")

    def test_query_rejects_short_question(self) -> None:
        response = self.client.post("/query", json={"question": "no"})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()