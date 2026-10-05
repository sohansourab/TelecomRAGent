"""Optional Ollama reasoning adapter with a local deterministic fallback."""

from __future__ import annotations

import logging
from typing import Any

from langchain_ollama import ChatOllama

from backend.config import Settings


LOGGER = logging.getLogger("telecomragent.llm")


class OllamaReasoner:
    """Generate report prose with a local Ollama model when available."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=settings.ollama_temperature,
            keep_alive=settings.ollama_keep_alive,
            num_predict=settings.ollama_num_predict,
            num_ctx=settings.ollama_num_ctx,
        )

    def generate_report(
        self,
        question: str,
        root_cause: dict[str, Any],
        resolution: dict[str, Any],
        fallback: str,
    ) -> str:
        """Ask Ollama for concise report prose, returning fallback on any failure."""
        prompt = (
            "You are a telecom operations analyst. Use only supplied evidence. "
            "Do not invent facts. Return at most 120 words with headings "
            "Finding, Evidence, Recommended actions.\n\n"
            f"Question: {question}\n"
            f"Root-cause evidence: {root_cause}\n"
            f"Resolution actions: {resolution}"
        )
        try:
            response = self.client.invoke(prompt)
            content = getattr(response, "content", response)
            report = str(content).strip()
            return report or fallback
        except Exception as error:
            LOGGER.warning(
                "Ollama unavailable at %s using model %s: %s",
                self.settings.ollama_base_url,
                self.settings.ollama_model,
                error,
            )
            return fallback
