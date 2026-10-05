"""Embed cleaned telecom records and persist them in ChromaDB."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer


LOGGER = logging.getLogger("telecomragent.index")
DEFAULT_INPUT = Path("data/processed/mycall_clean.csv")
DEFAULT_DB = Path("data/chroma")
DEFAULT_COLLECTION = "mycall_voice_quality"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REQUIRED_COLUMNS = {
    "document",
    "event_date",
    "state_name",
    "operator",
    "network_type",
    "severity",
    "rating",
}


def metadata_for_row(row: pd.Series) -> dict[str, str | float | int]:
    """Build Chroma-compatible metadata while omitting null coordinates."""
    metadata: dict[str, str | float | int] = {
        "region": str(row["state_name"]),
        "timestamp": str(row["event_date"]),
        "severity": str(row["severity"]),
        "operator": str(row["operator"]),
        "network_type": str(row["network_type"]),
        "call_category": str(row["calldrop_category"]),
        "rating": float(row["rating"]),
    }
    for column in ("latitude", "longitude"):
        if pd.notna(row[column]):
            metadata[column] = float(row[column])
    return metadata


def index_dataset(
    input_path: Path,
    db_path: Path,
    collection_name: str,
    model_name: str,
    batch_size: int,
    rebuild: bool,
) -> dict[str, object]:
    """Create or update a persistent Chroma collection and return its report."""
    frame = pd.read_csv(input_path)
    missing_columns = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    if frame.empty:
        raise ValueError("Cannot index an empty dataset")
    if batch_size < 1:
        raise ValueError("Batch size must be at least 1")

    client = chromadb.PersistentClient(path=str(db_path))
    if rebuild:
        try:
            client.delete_collection(collection_name)
        except Exception as error:
            LOGGER.debug("Collection did not need deletion: %s", error)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine", "embedding_model": model_name},
    )
    model = SentenceTransformer(model_name, device="cpu")
    documents = frame["document"].astype(str).tolist()
    ids = [f"mycall-{index:06d}" for index in range(len(frame))]
    total = len(documents)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        embeddings = model.encode(
            documents[start:end],
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).tolist()
        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings,
            metadatas=[metadata_for_row(row) for _, row in frame.iloc[start:end].iterrows()],
        )
        LOGGER.info("Indexed %d/%d records", end, total)

    dimension = len(collection.peek(1)["embeddings"][0])
    return {
        "collection": collection_name,
        "database_path": str(db_path),
        "embedding_model": model_name,
        "embedding_dimension": dimension,
        "distance": "cosine",
        "records_indexed": total,
        "collection_count": collection.count(),
        "batch_size": batch_size,
    }


def main() -> int:
    """Parse CLI arguments, index the dataset, and print an audit report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO if arguments.verbose else logging.WARNING)

    if not arguments.input.is_file():
        parser.error(f"Input file does not exist: {arguments.input}")
    try:
        report = index_dataset(
            arguments.input,
            arguments.db,
            arguments.collection,
            arguments.model,
            arguments.batch_size,
            arguments.rebuild,
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())