"""FastAPI application for TelecomRAGent retrieval."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Callable

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from backend.agent import TelecomAgent
from backend.config import Settings
from backend.llm import OllamaReasoner
from backend.retriever import ChromaRetriever


LOGGER = logging.getLogger("telecomragent.api")


class QueryRequest(BaseModel):
    """Validated semantic-query request."""

    question: str = Field(min_length=3, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=10)


class SourceResponse(BaseModel):
    """A source returned with a query response."""

    document: str
    metadata: dict[str, object]
    distance: float


class QueryResponse(BaseModel):
    """Stable API response for retrieval and future agent traces."""

    answer: str
    sources: list[SourceResponse]
    tool_trace: list[str]


class HealthResponse(BaseModel):
    """API health response."""

    status: str
    indexed_records: int | None = None


class StatsResponse(BaseModel):
    """Aggregated network statistics for the frontend sidebar."""

    indexed_records: int
    regions: int
    operators: list[str]
    network_types: list[str]
    severity_counts: dict[str, int]


def create_app(
    settings: Settings | None = None,
    retriever_factory: Callable[[Settings], ChromaRetriever] = ChromaRetriever,
    agent_factory: Callable[[ChromaRetriever], TelecomAgent] | None = None,
) -> FastAPI:
    """Create the API application with an injectable retriever factory."""
    runtime_settings = settings or Settings()
    make_agent = agent_factory or (
        lambda service: TelecomAgent(service, OllamaReasoner(runtime_settings))
    )
    retriever: ChromaRetriever | None = None

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.retriever = None
        yield
        application.state.retriever = None

    app = FastAPI(title="TelecomRAGent API", version="0.1.0", lifespan=lifespan)

    def get_retriever() -> ChromaRetriever:
        nonlocal retriever
        if retriever is None:
            try:
                retriever = retriever_factory(runtime_settings)
            except Exception as error:
                LOGGER.exception("Unable to initialize semantic retriever")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Semantic index is unavailable",
                ) from error
        return retriever

    agent: TelecomAgent | None = None

    def get_agent() -> TelecomAgent:
        nonlocal agent
        if agent is None:
            agent = make_agent(get_retriever())
        return agent

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Return API status without forcing model initialization."""
        return HealthResponse(status="ok")

    @app.get("/stats", response_model=StatsResponse)
    def stats() -> StatsResponse:
        """Return lightweight statistics without initializing the embedding model."""
        try:
            frame = pd.read_csv(runtime_settings.processed_data_path)
        except (OSError, pd.errors.ParserError) as error:
            LOGGER.exception("Unable to load processed statistics")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Processed dataset is unavailable",
            ) from error
        return StatsResponse(
            indexed_records=int(len(frame)),
            regions=int(frame["state_name"].nunique()),
            operators=sorted(frame["operator"].dropna().astype(str).unique()),
            network_types=sorted(frame["network_type"].dropna().astype(str).unique()),
            severity_counts={
                key: int(value) for key, value in frame["severity"].value_counts().items()
            },
        )

    @app.post("/query", response_model=QueryResponse)
    def query(
        request: QueryRequest,
        service: TelecomAgent = Depends(get_agent),
    ) -> QueryResponse:
        """Run the ordered telecom analysis tool chain for a user question."""
        top_k = request.top_k or runtime_settings.default_top_k
        result = service.run(request.question.strip(), top_k)
        source_models = [SourceResponse(**asdict(source)) for source in result.sources]
        return QueryResponse(
            answer=result.answer,
            sources=source_models,
            tool_trace=result.tool_trace,
        )

    return app


app = create_app()
