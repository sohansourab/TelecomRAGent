"""Download the selected TRAI MyCall voice-quality dataset."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


DATASET_URL = (
    "https://raw.githubusercontent.com/VaishnaviTale/Voice-Call-Quality-Analysis/"
    "main/Voice_Qaulity.csv"
)
DEFAULT_OUTPUT = Path("data/raw/mycall/Voice_Qaulity.csv")
MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024


def download_dataset(url: str, output_path: Path) -> int:
    """Download a CSV atomically and return its byte size."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "TelecomRAGent/1.0"})

    with tempfile.NamedTemporaryFile(
        mode="wb", dir=output_path.parent, prefix=".download-", delete=False
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        try:
            with urlopen(request, timeout=60) as response:
                total_bytes = 0
                while chunk := response.read(1024 * 1024):
                    total_bytes += len(chunk)
                    if total_bytes > MAX_DOWNLOAD_BYTES:
                        raise ValueError("Dataset exceeds the 50 MB safety limit")
                    temporary_file.write(chunk)
            if total_bytes == 0:
                raise ValueError("Downloaded dataset is empty")
            temporary_path.replace(output_path)
            return total_bytes
        except (OSError, URLError, ValueError):
            temporary_path.unlink(missing_ok=True)
            raise


def main() -> int:
    """Parse CLI arguments, download the dataset, and report its location."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.getenv("MYCALL_DATASET_URL", DATASET_URL))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()

    try:
        size = download_dataset(arguments.url, arguments.output)
    except (OSError, URLError, ValueError) as error:
        parser.error(str(error))
    print(f"Downloaded {size:,} bytes to {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())