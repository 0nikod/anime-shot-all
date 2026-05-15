"""pHash based duplicate analysis and export."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import imagehash
from PIL import Image

from .config import resolve_work_path
from .files import collect_images, parse_episode_id, relative_to_or_absolute
from .logging_utils import write_csv


DEDUP_LOG_FIELDS = [
    "source",
    "output",
    "episode_id",
    "scope_key",
    "status",
    "group_id",
    "hash",
    "nearest_distance",
    "nearest_kept",
    "manual_override",
    "dedup_scope",
    "hash_threshold",
    "hash_size",
    "hash_crop",
    "hash_resize_width",
    "error",
]


KEEP_STATUSES = {"kept", "manually_kept", "kept_unique"}
REJECT_STATUSES = {"duplicate", "manually_removed"}


def dedup_state_path(work_dir: Path) -> Path:
    return work_dir / "states" / "dedup_state.json"


def load_dedup_state(work_dir: Path) -> dict[str, Any]:
    path = dedup_state_path(work_dir)
    if not path.exists():
        return {"config": {}, "groups": [], "unique_images": []}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_dedup_state(work_dir: Path, state: dict[str, Any]) -> Path:
    path = dedup_state_path(work_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return path


def analyze_duplicates(work_dir: Path, config: dict[str, Any], input_dir: Path | None = None) -> tuple[dict[str, Any], Path]:
    params = config["dedup"]
    input_dir = input_dir or resolve_work_path(work_dir, config["paths"]["frames_raw"])
    images = _filter_images(collect_images(input_dir), params.get("episode_filter", "all"))
    records = [_hash_record(path, work_dir, params) for path in images]
    scope_map: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        scope_key = record["episode_id"] if params["dedup_scope"] == "per_episode" else "global"
        scope_map.setdefault(scope_key, []).append(record)

    groups: list[dict[str, Any]] = []
    unique_images: list[dict[str, Any]] = []
    for scope_key, scoped_records in scope_map.items():
        scoped_groups, scoped_unique = _cluster_scope(scope_key, scoped_records, params)
        groups.extend(scoped_groups)
        unique_images.extend(scoped_unique)

    state = {
        "config": {
            "hash_threshold": params["hash_threshold"],
            "hash_size": params["hash_size"],
            "hash_crop": params["hash_crop"],
            "hash_resize_width": params["hash_resize_width"],
            "dedup_scope": params["dedup_scope"],
            "episode_filter": params.get("episode_filter", "all"),
        },
        "groups": groups,
        "unique_images": unique_images,
    }
    path = save_dedup_state(work_dir, state)
    return state, path


def update_group_decision(state: dict[str, Any], group_id: str, keep_paths: list[str]) -> dict[str, Any]:
    keep_set = set(keep_paths)
    for group in state.get("groups", []):
        if group.get("group_id") != group_id:
            continue
        for image in group.get("images", []):
            original_keep = image.get("status") in KEEP_STATUSES
            should_keep = image.get("path") in keep_set
            image["status"] = "manually_kept" if should_keep else "manually_removed"
            image["manual_override"] = original_keep != should_keep
        break
    return state


def keep_all_in_group(state: dict[str, Any], group_id: str) -> dict[str, Any]:
    for group in state.get("groups", []):
        if group.get("group_id") == group_id:
            return update_group_decision(state, group_id, [item["path"] for item in group.get("images", [])])
    return state


def export_dedup_results(work_dir: Path, config: dict[str, Any], state: dict[str, Any] | None = None) -> tuple[int, int, Path]:
    params = config["dedup"]
    state = state or load_dedup_state(work_dir)
    output_dir = resolve_work_path(work_dir, config["paths"]["frames_dedup"])
    rejected_dir = resolve_work_path(work_dir, config["paths"]["rejected_duplicates"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if params.get("export_rejected_duplicates", True):
        rejected_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    kept_count = 0
    rejected_count = 0
    for record in _iter_state_records(state):
        source = resolve_work_path(work_dir, record["path"])
        output = ""
        error = ""
        try:
            if record["status"] in KEEP_STATUSES:
                target = output_dir / source.name
                shutil.copy2(source, target)
                output = relative_to_or_absolute(target, work_dir)
                kept_count += 1
            elif record["status"] in REJECT_STATUSES and params.get("export_rejected_duplicates", True):
                target = rejected_dir / source.name
                shutil.copy2(source, target)
                output = relative_to_or_absolute(target, work_dir)
                rejected_count += 1
        except OSError as exc:
            error = str(exc)
            record["status"] = "error"
        rows.append(_log_row(record, output, state.get("config", {}), error))
    log_path = resolve_work_path(work_dir, config["logging"]["dedup_log"])
    write_csv(log_path, DEDUP_LOG_FIELDS, rows)
    return kept_count, rejected_count, log_path


def group_gallery_items(work_dir: Path, state: dict[str, Any], group_id: str) -> tuple[list[tuple[str, str]], list[str]]:
    for group in state.get("groups", []):
        if group.get("group_id") != group_id:
            continue
        gallery = []
        keep_values = []
        for item in group.get("images", []):
            path = resolve_work_path(work_dir, item["path"])
            label = f"{Path(item['path']).name} | {item.get('status')} | d={item.get('nearest_distance', '')}"
            gallery.append((str(path), label))
            if item.get("status") in KEEP_STATUSES:
                keep_values.append(item["path"])
        return gallery, keep_values
    return [], []


def _filter_images(images: list[Path], episode_filter: Any) -> list[Path]:
    if not episode_filter or episode_filter == "all":
        return images
    allowed = {episode_filter} if isinstance(episode_filter, str) else set(episode_filter)
    return [path for path in images if parse_episode_id(path) in allowed]


def _hash_record(path: Path, work_dir: Path, params: dict[str, Any]) -> dict[str, Any]:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image = _crop_for_hash(image, str(params.get("hash_crop", "center")))
        resize_width = int(params.get("hash_resize_width", 256))
        if resize_width > 0 and image.width > resize_width:
            height = max(1, round(image.height * resize_width / image.width))
            image = image.resize((resize_width, height))
        value = imagehash.phash(image, hash_size=int(params.get("hash_size", 8)))
    episode_id = parse_episode_id(path)
    return {
        "path": relative_to_or_absolute(path, work_dir),
        "episode_id": episode_id,
        "hash": str(value),
        "_hash_obj": value,
    }


def _crop_for_hash(image: Image.Image, mode: str) -> Image.Image:
    if mode != "center":
        return image
    width, height = image.size
    crop_width = int(width * 0.6)
    crop_height = int(height * 0.6)
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    return image.crop((left, top, left + crop_width, top + crop_height))


def _cluster_scope(
    scope_key: str,
    records: list[dict[str, Any]],
    params: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    threshold = int(params.get("hash_threshold", 5))
    groups: list[list[dict[str, Any]]] = []
    unique: list[dict[str, Any]] = []
    consumed: set[int] = set()
    for index, record in enumerate(records):
        if index in consumed:
            continue
        cluster = [record]
        consumed.add(index)
        for other_index in range(index + 1, len(records)):
            if other_index in consumed:
                continue
            distance = record["_hash_obj"] - records[other_index]["_hash_obj"]
            if distance <= threshold:
                records[other_index]["nearest_distance"] = int(distance)
                records[other_index]["nearest_kept"] = record["path"]
                cluster.append(records[other_index])
                consumed.add(other_index)
        if len(cluster) > 1:
            groups.append(cluster)
        else:
            unique.append(_public_record(record, "kept_unique"))

    serialized_groups = []
    for group_index, cluster in enumerate(groups, start=1):
        scope_prefix = scope_key if scope_key != "global" else "global"
        group_id = f"{scope_prefix}_g{group_index:06d}"
        images = []
        for item_index, item in enumerate(cluster):
            status = "kept" if item_index == 0 else "duplicate"
            if item_index == 0:
                item["nearest_distance"] = 0
                item["nearest_kept"] = ""
            images.append(_public_record(item, status, group_id=group_id, scope_key=scope_key))
        serialized_groups.append(
            {
                "group_id": group_id,
                "episode_id": cluster[0]["episode_id"] if scope_key != "global" else "",
                "scope_key": scope_key,
                "images": images,
            }
        )
    return serialized_groups, unique


def _public_record(
    record: dict[str, Any],
    status: str,
    *,
    group_id: str = "",
    scope_key: str = "",
) -> dict[str, Any]:
    return {
        "path": record["path"],
        "episode_id": record["episode_id"],
        "scope_key": scope_key or record["episode_id"],
        "group_id": group_id,
        "hash": record["hash"],
        "nearest_distance": record.get("nearest_distance", 0),
        "nearest_kept": record.get("nearest_kept", ""),
        "status": status,
        "manual_override": False,
    }


def _iter_state_records(state: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for group in state.get("groups", []):
        for image in group.get("images", []):
            merged = dict(image)
            merged["group_id"] = group.get("group_id", merged.get("group_id", ""))
            merged["scope_key"] = group.get("scope_key", merged.get("scope_key", ""))
            records.append(merged)
    records.extend(state.get("unique_images", []))
    return records


def _log_row(record: dict[str, Any], output: str, cfg: dict[str, Any], error: str) -> dict[str, object]:
    return {
        "source": record.get("path", ""),
        "output": output,
        "episode_id": record.get("episode_id", ""),
        "scope_key": record.get("scope_key", ""),
        "status": record.get("status", ""),
        "group_id": record.get("group_id", ""),
        "hash": record.get("hash", ""),
        "nearest_distance": record.get("nearest_distance", ""),
        "nearest_kept": record.get("nearest_kept", ""),
        "manual_override": record.get("manual_override", False),
        "dedup_scope": cfg.get("dedup_scope", ""),
        "hash_threshold": cfg.get("hash_threshold", ""),
        "hash_size": cfg.get("hash_size", ""),
        "hash_crop": cfg.get("hash_crop", ""),
        "hash_resize_width": cfg.get("hash_resize_width", ""),
        "error": error,
    }
