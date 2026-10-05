"""Deterministic LangChain tool chain for telecom incident analysis."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from backend.retriever import ChromaRetriever, RetrievedSource
from backend.llm import OllamaReasoner


class RetrieveLogsInput(BaseModel):
    """Input contract for semantic log retrieval."""

    query: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=10)


class AnalyzeRootCauseInput(BaseModel):
    """Input contract for evidence aggregation."""

    question: str = Field(min_length=3, max_length=1000)
    sources_json: str = Field(min_length=2)


class GenerateResolutionInput(BaseModel):
    """Input contract for remediation generation."""

    root_cause_json: str = Field(min_length=2)


class GenerateReportInput(BaseModel):
    """Input contract for final report formatting."""

    question: str = Field(min_length=3, max_length=1000)
    root_cause_json: str = Field(min_length=2)
    resolution_json: str = Field(min_length=2)


@dataclass(frozen=True)
class AgentResult:
    """Complete result returned by the ordered tool chain."""

    answer: str
    sources: list[RetrievedSource]
    tool_trace: list[str]


class TelecomAgent:
    """Run the four TelecomRAGent tools in a fixed, auditable order."""

    def __init__(self, retriever: ChromaRetriever, reasoner: OllamaReasoner | None = None) -> None:
        self.retriever = retriever
        self.reasoner = reasoner
        self.retrieve_tool = StructuredTool.from_function(
            func=self._retrieve_logs,
            name="retrieve_logs",
            description="Retrieve semantically similar telecom voice-quality records.",
            args_schema=RetrieveLogsInput,
        )
        self.analyze_tool = StructuredTool.from_function(
            func=self._analyze_root_cause,
            name="analyze_root_cause",
            description="Aggregate evidence into a ranked telecom root-cause finding.",
            args_schema=AnalyzeRootCauseInput,
        )
        self.resolution_tool = StructuredTool.from_function(
            func=self._generate_resolution,
            name="generate_resolution",
            description="Generate operational remediation actions from the finding.",
            args_schema=GenerateResolutionInput,
        )
        self.report_tool = StructuredTool.from_function(
            func=self._generate_report,
            name="generate_report",
            description="Format the final telecom investigation report.",
            args_schema=GenerateReportInput,
        )

    def _retrieve_logs(self, query: str, top_k: int = 5) -> list[RetrievedSource]:
        return self.retriever.retrieve_logs(query, top_k)

    @staticmethod
    def _analyze_root_cause(question: str, sources_json: str) -> dict[str, Any]:
        sources = json.loads(sources_json)
        metadata = [source["metadata"] for source in sources]
        severity_counts = Counter(item.get("severity", "unknown") for item in metadata)
        operator_counts = Counter(item.get("operator", "unknown") for item in metadata)
        region_counts = Counter(item.get("region", "Unknown") for item in metadata)
        category_counts = Counter(item.get("call_category", "unknown") for item in metadata)
        dominant_region = region_counts.most_common(1)[0][0] if region_counts else "Unknown"
        dominant_operator = operator_counts.most_common(1)[0][0] if operator_counts else "Unknown"
        dominant_category = category_counts.most_common(1)[0][0] if category_counts else "Unknown"
        return {
            "question": question,
            "evidence_count": len(metadata),
            "dominant_region": dominant_region,
            "dominant_operator": dominant_operator,
            "dominant_issue": dominant_category,
            "severity_counts": dict(severity_counts),
            "operator_counts": dict(operator_counts),
            "root_cause": (
                f"The retrieved evidence is concentrated in {dominant_region}, with "
                f"{dominant_category.lower()} as the dominant issue and {dominant_operator} "
                "as the most represented operator."
            ),
        }

    @staticmethod
    def _generate_resolution(root_cause_json: str) -> dict[str, Any]:
        finding = json.loads(root_cause_json)
        issue = str(finding.get("dominant_issue", "")).lower()
        if "dropped" in issue:
            actions = [
                "Compare drop rates by cell and hour for the affected region.",
                "Inspect handover failures, radio coverage, and congestion counters.",
                "Prioritize drive tests around the highest-severity coordinates.",
            ]
        else:
            actions = [
                "Compare quality ratings by cell, network type, and time period.",
                "Inspect coverage, interference, and capacity metrics in the affected region.",
                "Recheck the result after remediation using the same query window.",
            ]
        return {"priority": "high", "actions": actions}

    def _generate_report(
        self, question: str, root_cause_json: str, resolution_json: str
    ) -> str:
        finding = json.loads(root_cause_json)
        resolution = json.loads(resolution_json)
        actions = " ".join(
            f"{index}. {action}" for index, action in enumerate(resolution["actions"], 1)
        )
        fallback = (
            f"Investigation question: {question}\n"
            f"Root cause assessment: {finding['root_cause']}\n"
            f"Evidence reviewed: {finding['evidence_count']} records.\n"
            f"Recommended actions: {actions}"
        )
        if self.reasoner is None:
            return fallback
        return self.reasoner.generate_report(question, finding, resolution, fallback)

    def run(self, question: str, top_k: int) -> AgentResult:
        """Execute all four tools in the required order."""
        sources = self.retrieve_tool.invoke({"query": question, "top_k": top_k})
        source_payload = json.dumps([asdict(source) for source in sources])
        root_cause = self.analyze_tool.invoke(
            {"question": question, "sources_json": source_payload}
        )
        resolution = self.resolution_tool.invoke(
            {"root_cause_json": json.dumps(root_cause)}
        )
        answer = self.report_tool.invoke(
            {
                "question": question,
                "root_cause_json": json.dumps(root_cause),
                "resolution_json": json.dumps(resolution),
            }
        )
        return AgentResult(
            answer=answer,
            sources=sources,
            tool_trace=[
                "retrieve_logs",
                "analyze_root_cause",
                "generate_resolution",
                "generate_report",
            ],
        )
