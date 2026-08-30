"""Shared CSV result-logging helper for the seqattention benchmark script."""

import csv
from datetime import datetime, timezone
from pathlib import Path


def write_csv(rows: list[dict], path: str) -> None:
    """Write a list of flat dict rows to a CSV file, creating parent dirs as
    needed. All rows must share the same set of keys. Existing files at
    `path` are overwritten, not appended."""
    if not rows:
        raise ValueError("write_csv called with an empty rows list")

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def default_output_path(prefix: str) -> str:
    """A timestamped default path under seq-attention/examples/results/."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return str(Path(__file__).parent / "results" / f"{prefix}_{timestamp}.csv")
