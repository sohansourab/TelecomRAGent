"""Validated backend configuration loaded from environment variables."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Runtime settings for the retrieval API."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    chroma_path: Path = Field(default=PROJECT_ROOT / "data" / "chroma")
    chroma_collection: str = "mycall_voice_quality"
    processed_data_path: Path = Field(
        default=PROJECT_ROOT / "data" / "processed" / "mycall_clean.csv"
    )
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b-instruct-q4_K_M"
    ollama_temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    ollama_keep_alive: str = "10m"
    ollama_num_predict: int = Field(default=180, ge=64, le=512)
    ollama_num_ctx: int = Field(default=2048, ge=512, le=8192)
    default_top_k: int = Field(default=3, ge=1, le=10)
    max_top_k: int = Field(default=10, ge=1, le=20)
