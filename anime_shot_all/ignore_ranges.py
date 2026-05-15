"""Ignore-range persistence, CSV conversion, and validation."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import resolve_work_path
from .timecode import parse_timecode


@dataclass(frozen=True)
class IgnoreRange:
    start: str
    end: str
    label: str = ""
    enabled: bool = True
    notes: str = ""

    @property
    def start_sec(self) -> float:
        return parse_timecode(self.start)

    @property
    def end_sec(self) -> float:
        return parse_timecode(self.end)


def state_path(work_dir: Path) -> Path:
    return work_dir / "states" / "ignore_ranges.json"


def load_ignore_state(work_dir: Path) -> dict[str, Any]:
    path = state_path(work_dir)
    if not path.exists():
        return {"episodes": []}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("episodes", []), list):
        raise ValueError(f"invalid ignore range state: {path}")
    return data


def save_ignore_state(work_dir: Path, state: dict[str, Any]) -> Path:
    path = state_path(work_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return path


def rows_to_state(rows: list[list[Any]], work_dir: Path, video_lookup: dict[str, str] | None = None) -> dict[str, Any]:
    episodes: dict[str, dict[str, Any]] = {}
    video_lookup = video_lookup or {}
    for row in rows:
        if not row or not row[0]:
            continue
        episode_id = str(row[0]).strip()
        video_name = str(row[1]).strip() if len(row) > 1 and row[1] else video_lookup.get(episode_id, "")
        entry = episodes.setdefault(
            episode_id,
            {"episode_id": episode_id, "video_path": video_name, "ignore_ranges": []},
        )
        if video_name:
            entry["video_path"] = video_name
        entry["ignore_ranges"].append(
            {
                "start": str(row[2]).strip() if len(row) > 2 else "",
                "end": str(row[3]).strip() if len(row) > 3 else "",
                "label": str(row[4]).strip() if len(row) > 4 and row[4] is not None else "",
                "enabled": _bool(row[5]) if len(row) > 5 else True,
                "notes": str(row[6]).strip() if len(row) > 6 and row[6] is not None else "",
            }
        )
    return {"episodes": list(episodes.values())}


def state_to_rows(state: dict[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for episode in state.get("episodes", []):
        episode_id = episode.get("episode_id", "")
        video_path = episode.get("video_path", "")
        video_name = Path(video_path).name if video_path else ""
        for item in episode.get("ignore_ranges", []):
            rows.append(
                [
                    episode_id,
                    video_name,
                    item.get("start", ""),
                    item.get("end", ""),
                    item.get("label", ""),
                    bool(item.get("enabled", True)),
                    item.get("notes", ""),
                ]
            )
    return rows


def import_csv(path: Path) -> list[list[Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            [
                row.get("episode_id", ""),
                row.get("video_name", ""),
                row.get("ignore_start", ""),
                row.get("ignore_end", ""),
                row.get("label", ""),
                _bool(row.get("enabled", True)),
                row.get("notes", ""),
            ]
            for row in reader
        ]


def export_csv(path: Path, rows: list[list[Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["episode_id", "video_name", "ignore_start", "ignore_end", "label", "enabled", "notes"])
        writer.writerows(rows)
    return path


def normalize_ranges(
    state: dict[str, Any],
    work_dir: Path,
    durations: dict[str, float] | None = None,
    auto_merge: bool = True,
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Validate and optionally merge enabled ranges.

    Returns ``(new_state, warnings, errors)`` so GUI callers can display
    warnings while blocking only hard errors.
    """

    durations = durations or {}
    warnings: list[str] = []
    errors: list[str] = []
    normalized_episodes: list[dict[str, Any]] = []

    for episode in state.get("episodes", []):
        episode_id = str(episode.get("episode_id", "")).strip()
        video_path = str(episode.get("video_path", "")).strip()
        if not episode_id:
            errors.append("episode_id is required")
            continue
        if video_path:
            resolved = resolve_work_path(work_dir, video_path)
            if not resolved.exists():
                warnings.append(f"{episode_id}: video does not exist: {video_path}")
        active: list[dict[str, Any]] = []
        disabled: list[dict[str, Any]] = []
        for item in episode.get("ignore_ranges", []):
            copied = dict(item)
            copied["enabled"] = _bool(copied.get("enabled", True))
            try:
                start = parse_timecode(copied.get("start", ""))
                end = parse_timecode(copied.get("end", ""))
            except ValueError as exc:
                errors.append(f"{episode_id}: {exc}")
                continue
            if start >= end:
                errors.append(f"{episode_id}: ignore_start must be less than ignore_end")
                continue
            duration = durations.get(episode_id)
            if duration is not None and end > duration:
                errors.append(f"{episode_id}: ignore range exceeds duration {duration:.3f}s")
                continue
            copied["_start_sec"] = start
            copied["_end_sec"] = end
            if copied["enabled"]:
                active.append(copied)
            else:
                disabled.append(_strip_internal(copied))
        active.sort(key=lambda value: value["_start_sec"])
        if auto_merge:
            merged = _merge_active_ranges(active, episode_id, warnings)
        else:
            merged = _check_overlaps(active, episode_id, warnings)
        normalized_episodes.append(
            {
                "episode_id": episode_id,
                "video_path": video_path,
                "ignore_ranges": [_strip_internal(item) for item in merged] + disabled,
            }
        )
    return {"episodes": normalized_episodes}, warnings, errors


def active_ranges_for_episode(state: dict[str, Any], episode_id: str) -> list[dict[str, Any]]:
    for episode in state.get("episodes", []):
        if episode.get("episode_id") != episode_id:
            continue
        ranges = []
        for item in episode.get("ignore_ranges", []):
            if not _bool(item.get("enabled", True)):
                continue
            ranges.append({**item, "start_sec": parse_timecode(item["start"]), "end_sec": parse_timecode(item["end"])})
        return sorted(ranges, key=lambda value: value["start_sec"])
    return []


def match_ignore(timestamp: float, ranges: list[dict[str, Any]]) -> dict[str, Any] | None:
    for item in ranges:
        if item["start_sec"] <= timestamp < item["end_sec"]:
            return item
    return None


def _merge_active_ranges(active: list[dict[str, Any]], episode_id: str, warnings: list[str]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for item in active:
        if not merged or item["_start_sec"] > merged[-1]["_end_sec"]:
            merged.append(item)
            continue
        previous = merged[-1]
        warnings.append(f"{episode_id}: merged overlapping ignore ranges")
        if item["_end_sec"] > previous["_end_sec"]:
            previous["end"] = item["end"]
            previous["_end_sec"] = item["_end_sec"]
        labels = [value for value in [previous.get("label", ""), item.get("label", "")] if value]
        previous["label"] = "+".join(dict.fromkeys(labels))
    return merged


def _check_overlaps(active: list[dict[str, Any]], episode_id: str, warnings: list[str]) -> list[dict[str, Any]]:
    for previous, current in zip(active, active[1:]):
        if current["_start_sec"] <= previous["_end_sec"]:
            warnings.append(f"{episode_id}: overlapping ignore ranges")
    return active


def _strip_internal(item: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in item.items() if not key.startswith("_")}


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}
