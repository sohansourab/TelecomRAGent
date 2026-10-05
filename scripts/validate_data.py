"""Validate the selected TRAI MyCall CSV and emit a JSON quality report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


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
DEFAULT_INPUT = Path("data/raw/mycall/Voice_Qaulity.csv")
DEFAULT_REPORT = Path("data/processed/mycall_validation.json")


def validate_dataset(input_path: Path) -> dict[str, object]:
    """Validate schema, ranges, nulls, and duplicate rows in the CSV."""
    frame = pd.read_csv(input_path)
    missing_columns = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    numeric_columns = ["rating", "latitude", "longitude", "month", "year"]
    numeric_frame = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    report: dict[str, object] = {
        "source": str(input_path),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "missing_values": {
            column: int(value) for column, value in frame.isna().sum().items() if value
        },
        "duplicate_rows": int(frame.duplicated().sum()),
        "invalid_numeric_values": {
            column: int(value)
            for column, value in numeric_frame.isna().sum().items()
            if value
        },
        "distinct_operators": sorted(frame["operator"].dropna().astype(str).unique()),
        "distinct_states": int(frame["state_name"].nunique(dropna=True)),
        "distinct_network_types": sorted(
            frame["network_type"].dropna().astype(str).unique()
        ),
        "rating_range": [
            float(numeric_frame["rating"].min()),
            float(numeric_frame["rating"].max()),
        ],
        "latitude_range": [
            float(numeric_frame["latitude"].min()),
            float(numeric_frame["latitude"].max()),
        ],
        "longitude_range": [
            float(numeric_frame["longitude"].min()),
            float(numeric_frame["longitude"].max()),
        ],
    }
    return report


def main() -> int:
    """Parse CLI arguments, validate the CSV, and write a JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    arguments = parser.parse_args()

    if not arguments.input.is_file():
        parser.error(f"Input file does not exist: {arguments.input}")
    try:
        report = validate_dataset(arguments.input)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        parser.error(str(error))

    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())