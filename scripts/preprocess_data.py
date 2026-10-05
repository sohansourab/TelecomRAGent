"""Clean the TRAI MyCall CSV and create retrieval-ready telecom records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path("data/raw/mycall/Voice_Qaulity.csv")
DEFAULT_OUTPUT = Path("data/processed/mycall_clean.csv")
DEFAULT_REPORT = Path("data/processed/mycall_preprocessing.json")
REQUIRED_COLUMNS = {
    "inout_travelling",
    "operator",
    "network_type",
    "rating",
    "calldrop_category",
    "latitude",
    "longitude",
    "state_name",
    "month",
    "year",
}
OPERATOR_NAMES = {
    "RJio": "Jio",
    "VI": "Vi",
    "Vodafone": "Vodafone",
    "\ufeffRJio": "Jio",
    "\ufeffIdea": "Idea",
}


def normalize_text(series: pd.Series, replacements: dict[str, str] | None = None) -> pd.Series:
    """Strip text values, replace blanks, and preserve missing values for filling."""
    normalized = series.astype("string").str.strip()
    missing_tokens = normalized.str.lower().isin({"", "na", "nan", "none", "null"})
    normalized = normalized.mask(normalized.isna() | missing_tokens, "Unknown")
    if replacements:
        normalized = normalized.replace(replacements)
    return normalized


def severity_for_row(row: pd.Series) -> str:
    """Assign an explainable severity from call category and user rating."""
    category = str(row["calldrop_category"]).strip().lower()
    rating = float(row["rating"])
    if "dropped" in category or rating <= 1:
        return "critical"
    if "poor" in category or rating <= 2:
        return "high"
    if rating <= 3:
        return "medium"
    return "low"


def clean_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Normalize records and return the cleaned frame with an audit summary."""
    missing_columns = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    cleaned = frame.copy()
    input_rows = len(cleaned)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    cleaned["operator"] = normalize_text(cleaned["operator"], OPERATOR_NAMES)
    cleaned["network_type"] = normalize_text(cleaned["network_type"])
    cleaned["state_name"] = normalize_text(cleaned["state_name"])
    cleaned["inout_travelling"] = normalize_text(cleaned["inout_travelling"])
    cleaned["calldrop_category"] = normalize_text(cleaned["calldrop_category"])

    for column in ("rating", "latitude", "longitude", "month", "year"):
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    invalid_coordinates = ~cleaned["latitude"].between(6, 38) | ~cleaned["longitude"].between(68, 98)
    invalid_coordinates |= cleaned[["latitude", "longitude"]].isna().any(axis=1)
    cleaned.loc[invalid_coordinates, ["latitude", "longitude"]] = pd.NA

    cleaned["event_date"] = pd.to_datetime(
        {
            "year": cleaned["year"],
            "month": cleaned["month"],
            "day": 1,
        },
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")
    cleaned["severity"] = cleaned.apply(severity_for_row, axis=1)
    cleaned["document"] = cleaned.apply(
        lambda row: (
            f"{row['severity'].title()} telecom voice-quality event in {row['state_name']}. "
            f"Operator: {row['operator']}. Network: {row['network_type']}. "
            f"Category: {row['calldrop_category']}. Rating: {row['rating']}/5. "
            f"Travel context: {row['inout_travelling']}. Date: {row['event_date']}."
        ),
        axis=1,
    )

    output_columns = [
        "document",
        "event_date",
        "year",
        "month",
        "state_name",
        "operator",
        "network_type",
        "calldrop_category",
        "severity",
        "rating",
        "latitude",
        "longitude",
        "inout_travelling",
    ]
    cleaned = cleaned[output_columns].drop_duplicates().reset_index(drop=True)
    report: dict[str, object] = {
        "input_rows": int(input_rows),
        "output_rows": int(len(cleaned)),
        "duplicates_removed": int(input_rows - len(cleaned) if input_rows else 0),
        "invalid_coordinates_replaced_with_null": int(cleaned["latitude"].isna().sum()),
        "null_counts": {
            column: int(value) for column, value in cleaned.isna().sum().items() if value
        },
        "severity_counts": {
            key: int(value) for key, value in cleaned["severity"].value_counts().items()
        },
    }
    return cleaned, report


def main() -> int:
    """Read, clean, and write the processed dataset and audit report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    arguments = parser.parse_args()

    if not arguments.input.is_file():
        parser.error(f"Input file does not exist: {arguments.input}")
    try:
        cleaned, report = clean_dataset(pd.read_csv(arguments.input))
    except (OSError, ValueError, pd.errors.ParserError) as error:
        parser.error(str(error))

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(cleaned.to_csv(index=False), encoding="utf-8")
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())