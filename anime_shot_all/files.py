"""Small filesystem helpers."""

from __future__ import annotations

import re
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def natural_key(value: str | Path) -> list[object]:
    text = Path(value).name if isinstance(value, Path) else value
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def relative_to_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def relative_path_value(value: str | Path, root: Path) -> str:
    """Store paths inside ``root`` as relative values, external paths as absolute."""

    path = Path(value).expanduser()
    if not path.is_absolute():
        return path.as_posix()
    return relative_to_or_absolute(path, root)


def collect_images(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(
        [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=natural_key,
    )


def parse_episode_id(path: str | Path) -> str:
    name = Path(path).name
    match = re.match(r"(?P<episode>ep\d+)", name, flags=re.IGNORECASE)
    if match:
        return match.group("episode").lower()
    stem = Path(path).stem
    match = re.search(r"(\d+)", stem)
    if match:
        return f"ep{int(match.group(1)):02d}"
    return stem
