"""Project summary statistics for the log/output tab."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .config import resolve_work_path
from .files import collect_images
from .ignore_ranges import load_ignore_state


def summarize_project(work_dir: Path, config: dict[str, Any]) -> dict[str, object]:
    paths = config["paths"]
    frames_raw = collect_images(resolve_work_path(work_dir, paths["frames_raw"]))
    crops_dir = resolve_work_path(work_dir, paths["crops"])
    ignore_state = load_ignore_state(work_dir)
    crop_counts = _crop_counts(crops_dir)
    return {
        "work_dir": str(work_dir),
        "ignore_ranges": sum(len(item.get("ignore_ranges", [])) for item in ignore_state.get("episodes", [])),
        "frames_raw": len(frames_raw),
        "crops_total": sum(crop_counts.values()),
        **crop_counts,
    }


def format_summary(summary: dict[str, object]) -> str:
    lines = [f"{key}: {value}" for key, value in summary.items()]
    return "\n".join(lines)


def recent_log_text(path: Path, max_rows: int = 20) -> str:
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    return "\n".join(",".join(row) for row in rows[-max_rows:])


def _crop_counts(crops_dir: Path) -> dict[str, int]:
    keys = ["face", "body", "background", "full", "random_crop"]
    counts = {key: len(collect_images(crops_dir / key)) for key in keys}
    counts["hard_split"] = len(collect_images(crops_dir / "hard"))
    return counts
