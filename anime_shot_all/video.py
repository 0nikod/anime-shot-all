"""Video directory scanning via ffprobe."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from .files import natural_key, relative_to_or_absolute
from .progress import ProgressCallback, format_progress, process_error_message


@dataclass(frozen=True)
class VideoInfo:
    episode_id: str
    video_path: str
    video_name: str
    duration_sec: float
    fps: float
    width: int
    height: int


def _parse_fps(value: str) -> float:
    if not value or value == "0/0":
        return 0.0
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        denominator_float = float(denominator)
        return float(numerator) / denominator_float if denominator_float else 0.0
    return float(value)


def probe_video(path: Path, episode_id: str, work_dir: Path) -> VideoInfo:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate:format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        payload = json.loads(result.stdout)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(process_error_message(f"ffprobe failed for {path}", error)) from error
    except json.JSONDecodeError as error:
        raise RuntimeError(f"ffprobe returned invalid JSON for {path}: {error}") from error
    stream = (payload.get("streams") or [{}])[0]
    duration = float((payload.get("format") or {}).get("duration") or 0)
    return VideoInfo(
        episode_id=episode_id,
        video_path=relative_to_or_absolute(path, work_dir),
        video_name=path.name,
        duration_sec=duration,
        fps=_parse_fps(stream.get("r_frame_rate", "")),
        width=int(stream.get("width") or 0),
        height=int(stream.get("height") or 0),
    )


def video_candidates(video_dir: Path, supported_ext: list[str]) -> list[Path]:
    extensions = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in supported_ext}
    return sorted(
        [p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in extensions],
        key=natural_key,
    )


def scan_videos(video_dir: Path, work_dir: Path, supported_ext: list[str], progress: ProgressCallback | None = None) -> list[VideoInfo]:
    candidates = video_candidates(video_dir, supported_ext)
    videos: list[VideoInfo] = []
    for index, path in enumerate(candidates, start=1):
        if progress:
            progress(format_progress("scan", index, len(candidates), path))
        try:
            videos.append(probe_video(path, f"ep{len(videos) + 1:02d}", work_dir))
        except RuntimeError as error:
            if progress:
                progress(str(error))
    return videos


def videos_to_rows(videos: list[VideoInfo]) -> list[list[object]]:
    return [
        [
            item.episode_id,
            item.video_path,
            item.video_name,
            round(item.duration_sec, 3),
            round(item.fps, 3),
            item.width,
            item.height,
        ]
        for item in videos
    ]


def videos_as_dicts(videos: list[VideoInfo]) -> list[dict[str, object]]:
    return [asdict(video) for video in videos]
