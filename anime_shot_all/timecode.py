"""Time parsing helpers used by ignore ranges and filenames."""

from __future__ import annotations


def parse_timecode(value: str | int | float) -> float:
    """Parse ``HH:MM:SS``, ``MM:SS``, or raw seconds into seconds."""

    if isinstance(value, (int, float)):
        seconds = float(value)
    else:
        text = str(value).strip()
        if not text:
            raise ValueError("empty time value")
        parts = text.split(":")
        if len(parts) == 1:
            seconds = float(parts[0])
        elif len(parts) == 2:
            minutes, secs = parts
            seconds = int(minutes) * 60 + float(secs)
        elif len(parts) == 3:
            hours, minutes, secs = parts
            seconds = int(hours) * 3600 + int(minutes) * 60 + float(secs)
        else:
            raise ValueError(f"invalid time value: {value}")
    if seconds < 0:
        raise ValueError(f"time must be non-negative: {value}")
    return seconds


def format_timecode(seconds: float) -> str:
    whole = int(seconds)
    hours = whole // 3600
    minutes = (whole % 3600) // 60
    secs = whole % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def timestamp_token(seconds: float) -> str:
    return f"{seconds:010.3f}"
