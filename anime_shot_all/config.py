"""YAML configuration loading and project directory management."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .defaults import builtin_defaults


REQUIRED_DIR_KEYS = (
    "frames_raw",
    "frames_dedup",
    "rejected_duplicates",
    "crops",
    "logs",
    "states",
    "configs",
)


def deep_merge(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    """Recursively merge dictionaries without treating lists as mergeable."""

    if not override:
        return deepcopy(base)
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)


def resolve_work_path(work_dir: Path, value: str | Path) -> Path:
    """Resolve a possibly relative project path against ``work_dir``."""

    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return work_dir / path


def project_paths(work_dir: str | Path, config: dict[str, Any]) -> dict[str, Path]:
    root = Path(work_dir).expanduser().resolve()
    paths = config.get("paths", {})
    return {key: resolve_work_path(root, paths.get(key, key)) for key in REQUIRED_DIR_KEYS}


def ensure_work_dir(work_dir: str | Path, create: bool = True) -> Path:
    root = Path(work_dir).expanduser().resolve()
    if root.exists() and not root.is_dir():
        raise NotADirectoryError(root)
    if not root.exists():
        if not create:
            raise FileNotFoundError(root)
        root.mkdir(parents=True, exist_ok=True)
    return root


def initialize_work_dir(work_dir: str | Path) -> tuple[dict[str, Any], list[str]]:
    """Create project directories and config files, then return merged config."""

    root = ensure_work_dir(work_dir, create=True)
    defaults = builtin_defaults()
    defaults["project"]["work_dir"] = str(root)

    config_dir = root / defaults["paths"]["configs"]
    default_path = config_dir / "default.yaml"
    params_path = config_dir / "params.yaml"

    messages: list[str] = []
    if not default_path.exists():
        write_yaml(default_path, defaults)
        messages.append(f"created {default_path}")
    default_config = deep_merge(defaults, read_yaml(default_path))

    if not params_path.exists():
        params_config = deepcopy(default_config)
        params_config["project"]["work_dir"] = str(root)
        write_yaml(params_path, params_config)
        messages.append(f"created {params_path}")

    config = load_project_config(root)
    for path in project_paths(root, config).values():
        path.mkdir(parents=True, exist_ok=True)
    video_dir = resolve_work_path(root, config["project"].get("video_dir", "videos"))
    video_dir.mkdir(parents=True, exist_ok=True)
    messages.append(f"initialized {root}")
    return config, messages


def load_project_config(work_dir: str | Path) -> dict[str, Any]:
    root = ensure_work_dir(work_dir, create=False)
    defaults = builtin_defaults()
    config_dir = root / defaults["paths"]["configs"]
    default_config = deep_merge(defaults, read_yaml(config_dir / "default.yaml"))
    params_config = read_yaml(config_dir / "params.yaml")
    config = deep_merge(default_config, params_config)
    config.setdefault("project", {})["work_dir"] = str(root)
    return config


def save_params(work_dir: str | Path, config: dict[str, Any]) -> Path:
    root = ensure_work_dir(work_dir, create=True)
    path = root / "configs" / "params.yaml"
    data = deepcopy(config)
    data.setdefault("project", {})["work_dir"] = str(root)
    write_yaml(path, data)
    return path


def reset_params_from_default(work_dir: str | Path) -> dict[str, Any]:
    root = ensure_work_dir(work_dir, create=False)
    default_config = deep_merge(builtin_defaults(), read_yaml(root / "configs" / "default.yaml"))
    default_config.setdefault("project", {})["work_dir"] = str(root)
    write_yaml(root / "configs" / "params.yaml", default_config)
    return default_config
