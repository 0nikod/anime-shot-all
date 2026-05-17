"""CSV logging helpers."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    output_fieldnames = ["saved_at", *fieldnames] if "saved_at" not in fieldnames else fieldnames
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({"saved_at": timestamp, **row})
    return path
