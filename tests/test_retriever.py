"""Unit tests for retrieval result shaping."""

from __future__ import annotations

import unittest

from backend.retriever import RetrievedSource, select_diverse_sources


class RetrievalSelectionTests(unittest.TestCase):
    """Ensure repeated evidence does not dominate the displayed sources."""

    def test_caps_repeated_region_operator_issue_groups(self) -> None:
        sources = [
            RetrievedSource(
                document=f"record-{index}",
                metadata={
                    "region": "Uttar Pradesh",
                    "operator": "Airtel",
                    "call_category": "Call Dropped",
                },
                distance=float(index),
            )
            for index in range(4)
        ]
        sources.append(
            RetrievedSource(
                document="other-record",
                metadata={
                    "region": "Karnataka",
                    "operator": "Jio",
                    "call_category": "Poor Voice Quality",
                },
                distance=4.0,
            )
        )
        selected = select_diverse_sources(sources, top_k=3)
        self.assertEqual([source.document for source in selected], ["record-0", "record-1", "other-record"])


if __name__ == "__main__":
    unittest.main()