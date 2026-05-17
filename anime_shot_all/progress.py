"""Progress message and subprocess output formatting helpers."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path


ProgressCallback = Callable[[str], None]


def normalize_process_output(text: str | bytes | None) -> str:
    """Normalize ffmpeg/ffprobe stderr into newline-delimited text."""

    if text is None:
        return ""
    if isinstance(text, bytes):
        text = text.decode(errors="replace")
    lines = [line.strip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return "\n".join(line for line in lines if line)


def process_error_message(prefix: str, error: BaseException) -> str:
    stderr = normalize_process_output(getattr(error, "stderr", ""))
    if stderr:
        return f"{prefix}: {stderr}"
    return f"{prefix}: {error}"


def format_progress(action: str, current: int, total: int, path: str | Path) -> str:
    return f"{action} {current}/{total}: {Path(path).name}"
