"""Built-in configuration defaults.

The GUI writes these defaults to ``work_dir/configs/default.yaml`` when a
project is opened for the first time. They intentionally mirror the public
YAML contract so missing user fields can be filled by a plain deep merge.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


BUILTIN_DEFAULTS: dict[str, Any] = {
    "project": {
        "work_dir": "work_dir",
        "video_dir": "../video_dir",
        "output_dir": ".",
        "supported_video_ext": [
            ".mp4",
            ".mkv",
            ".avi",
            ".mov",
            ".webm",
            ".flv",
            ".wmv",
            ".m4v",
        ],
    },
    "paths": {
        "frames_raw": "frames_raw",
        "rejected_duplicates": "rejected_duplicates",
        "crops": "crops",
        "logs": "logs",
        "states": "states",
        "configs": "configs",
    },
    "extract": {
        "interval": 0.25,
        "keyframe_only": False,
        "png_compression": 3,
        "min_width": 0,
        "crop_bottom": 0,
        "phash_threshold": 5,
        "phash_size": 8,
        "phash_crop": "center",
        "phash_resize_width": 256,
        "group_seconds_per_keep": 5.0,
        "group_max_duration": 60.0,
        "extract_random_seed": 42,
    },
    "ignore_ranges": {
        "enabled": True,
        "auto_merge_overlaps": True,
        "require_valid_ranges": True,
        "allow_empty_ranges": True,
    },
    "dedup": {
        "hash_threshold": 5,
        "hash_size": 8,
        "hash_crop": "center",
        "hash_resize_width": 256,
        "dedup_scope": "per_episode",
        "episode_filter": "all",
        "export_rejected_duplicates": True,
        "num_workers": -1,
    },
    "dedup_preview": {
        "review_mode": "single_group",
        "default_keep_strategy": "first",
        "save_state_path": "states/dedup_state.json",
    },
    "crop": {
        "input_dir": "frames_raw",
        "output_dir": "crops",
        "png_compression": 3,
        "min_crop_size": 128,
        "output_strategy": "fixed",
        "target_crops_per_image": 3,
        "random_seed": 42,
    },
    "crop_types": {
        "full": True,
        "face": True,
        "body": True,
        "halfbody": True,
        "random_crop": True,
    },
    "full_crop": {
        "min_area": 1048576,
        "max_area": 2359296,
        "max_upscale": 2.0,
    },
    "ratio_selection": {
        "sigma": 0.45,
        "max_ratio_change": 2.2,
        "always_allow_square": True,
        "base_weights": {
            "9:16": 1.0,
            "3:4": 1.2,
            "1:1": 1.5,
            "4:3": 1.2,
            "16:9": 1.0,
        },
    },
    "detection": {
        "conf_threshold": 0.35,
        "iou_threshold": 0.7,
        "face_level": "s",
        "face_version": "v1.4",
        "person_level": "m",
        "person_version": "v1.1",
        "halfbody_level": "s",
        "halfbody_version": "v1.0",
    },
    "body_crop": {
        "expand_top": 1.15,
        "expand_bottom": 1.15,
        "expand_left": 1.2,
        "expand_right": 1.2,
        "min_size": 256,
        "max_count_per_image": 3,
    },
    "halfbody_crop": {
        "expand_top": 1.2,
        "expand_bottom": 1.25,
        "expand_left": 1.2,
        "expand_right": 1.2,
        "min_size": 192,
        "max_count_per_image": 3,
    },
    "face_crop": {
        "expand_top": 1.5,
        "expand_bottom": 2.0,
        "expand_left": 1.4,
        "expand_right": 1.4,
        "min_size": 128,
        "max_count_per_image": 5,
    },
    "random_crop": {
        "count_per_image": 2,
        "min_scale": 0.35,
        "max_scale": 0.85,
        "avoid_body": False,
    },
    "random_output_weights": {
        "full": 0,
        "face": 25,
        "body": 25,
        "halfbody": 25,
        "random_crop": 10,
    },
    "logging": {
        "extract_log": "logs/extract_log.csv",
        "dedup_log": "logs/dedup_log.csv",
        "crop_log": "logs/crop_log.csv",
        "verbose": True,
    },
}


def builtin_defaults() -> dict[str, Any]:
    """Return a deep copy so callers can mutate safely."""

    return deepcopy(BUILTIN_DEFAULTS)
