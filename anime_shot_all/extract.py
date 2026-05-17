"""Frame extraction with ignore-range aware scene-diff sampling."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import imagehash
import numpy as np
from PIL import Image

from .config import resolve_work_path
from .files import relative_to_or_absolute
from .ignore_ranges import active_ranges_for_episode, load_ignore_state, match_ignore
from .logging_utils import write_csv
from .progress import ProgressCallback, format_progress, process_error_message
from .timecode import timestamp_token
from .video import VideoInfo


EXTRACT_LOG_FIELDS = [
    "episode_id",
    "video",
    "image",
    "frame_index",
    "timestamp_sec",
    "diff_score",
    "reason",
    "ignored",
    "ignore_label",
    "ignore_start",
    "ignore_end",
    "output_path",
    "status",
    "error",
]


def extract_frames_for_videos(
    work_dir: Path,
    config: dict[str, Any],
    videos: list[VideoInfo],
    output_dir: Path | None = None,
    stop_state: dict[str, Any] | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[int, Path, list[str]]:
    """Extract frames for all videos and write one combined CSV log."""

    output_dir = output_dir or resolve_work_path(work_dir, config["paths"]["frames_raw"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = resolve_work_path(work_dir, config["logging"]["extract_log"])
    ignore_state = load_ignore_state(work_dir)
    rows: list[dict[str, object]] = []
    messages: list[str] = []
    saved_total = 0
    total = len(videos)
    for index, video in enumerate(videos, start=1):
        if progress:
            progress(format_progress("extract", index, total, video.video_name))
        saved, video_rows = extract_frames_for_video(
            work_dir,
            config,
            video,
            ignore_state,
            output_dir,
            stop_state=stop_state,
            progress=progress,
        )
        saved_total += saved
        rows.extend(video_rows)
        message = f"{video.episode_id}: saved {saved} frames from {video.video_name}"
        messages.append(message)
        if progress:
            progress(message)
        if stop_state and stop_state.get("stop"):
            messages.append("stopped by user")
            if progress:
                progress("stopped by user")
            break
    write_csv(log_path, EXTRACT_LOG_FIELDS, rows)
    return saved_total, log_path, messages


def extract_frames_for_video(
    work_dir: Path,
    config: dict[str, Any],
    video: VideoInfo,
    ignore_state: dict[str, Any],
    output_dir: Path,
    stop_state: dict[str, Any] | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[int, list[dict[str, object]]]:
    params = config["extract"]
    video_path = resolve_work_path(work_dir, video.video_path)
    keyframe_only = bool(params.get("keyframe_only", False))
    if keyframe_only:
        return _extract_keyframes_for_video(
            work_dir,
            params,
            video,
            video_path,
            ignore_state,
            output_dir,
            stop_state,
            progress,
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0, [_error_row(video, f"cannot open video: {video_path}")]

    png_compression = int(params["png_compression"])
    ranges = active_ranges_for_episode(ignore_state, video.episode_id)
    phash_threshold = int(params.get("phash_threshold", 5))
    group_seconds_per_keep = float(params.get("group_seconds_per_keep", 5.0))
    group_max_duration = float(params.get("group_max_duration", 60.0))
    rng = random.Random(int(params.get("extract_random_seed", 42)))

    rows: list[dict[str, object]] = []
    saved_count = 0
    group_frames: list[_PendingFrame] = []
    group_start_ts: float | None = None
    last_group_hash: imagehash.ImageHash | None = None
    group_index = 0

    stopped = False
    fps = cap.get(cv2.CAP_PROP_FPS) or video.fps or 24.0
    progress_reporter = _VideoProgressReporter(
        progress,
        video.video_name,
        "frame",
        _estimate_total_frames(cap, video, fps),
    )
    progress_reporter.emit(0, saved_count, len(group_frames), force=True)
    frame_step = max(1, int(round(float(params["interval"]) * fps)))
    frame_index = 0
    while True:
        if _stop_requested(stop_state):
            stopped = True
            break
        ok = cap.grab()
        if not ok:
            break
        if frame_index % frame_step != 0:
            frame_index += 1
            progress_reporter.emit(frame_index, saved_count, len(group_frames))
            continue
        ok, frame = cap.retrieve()
        if not ok:
            rows.append(_base_row(video, frame_index, frame_index / fps, status="error", error="cannot retrieve frame"))
            frame_index += 1
            progress_reporter.emit(frame_index, saved_count, len(group_frames))
            continue
        timestamp = frame_index / fps
        ignored = match_ignore(timestamp, ranges)
        if ignored:
            rows.append(
                _base_row(
                    video,
                    frame_index,
                    timestamp,
                    ignored=True,
                    ignore_label=ignored.get("label", ""),
                    ignore_start=ignored.get("start", ""),
                    ignore_end=ignored.get("end", ""),
                    status="skipped_ignore",
                )
            )
            frame_index += 1
            progress_reporter.emit(frame_index, saved_count, len(group_frames))
            continue

        processed = _prepare_for_output(frame, int(params["crop_bottom"]), int(params["min_width"]))

        if group_start_ts is not None and (timestamp - group_start_ts) >= group_max_duration:
            group_index, saved_delta = _finalize_group(
                group_frames,
                video,
                work_dir,
                output_dir,
                png_compression,
                rng,
                group_index,
                group_seconds_per_keep,
                rows,
            )
            saved_count += saved_delta
            group_start_ts = None
            last_group_hash = None
            progress_reporter.emit(frame_index, saved_count, len(group_frames))

        frame_hash = _phash_frame(processed, params)
        if last_group_hash is None or (frame_hash - last_group_hash) > phash_threshold:
            group_index, saved_delta = _finalize_group(
                group_frames,
                video,
                work_dir,
                output_dir,
                png_compression,
                rng,
                group_index,
                group_seconds_per_keep,
                rows,
            )
            saved_count += saved_delta
            group_start_ts = timestamp
            last_group_hash = frame_hash
            group_frames = []
            progress_reporter.emit(frame_index, saved_count, len(group_frames))
        else:
            last_group_hash = frame_hash

        group_frames.append(
            _PendingFrame(
                frame_index=frame_index,
                timestamp=timestamp,
                processed=processed,
                diff_score=None,
                reason="phash",
            )
        )

        frame_index += 1
        progress_reporter.emit(frame_index, saved_count, len(group_frames))

    if not stopped:
        group_index, saved_delta = _finalize_group(
            group_frames,
            video,
            work_dir,
            output_dir,
            png_compression,
            rng,
            group_index,
            group_seconds_per_keep,
            rows,
        )
        saved_count += saved_delta

    cap.release()
    progress_reporter.emit(frame_index, saved_count, len(group_frames), force=True)
    return saved_count, rows


@dataclass
class _KeyframeInfo:
    frame_index: int
    timestamp: float


@dataclass
class _PendingFrame:
    frame_index: int
    timestamp: float
    processed: np.ndarray
    diff_score: float | None
    reason: str


class _VideoProgressReporter:
    """Emit throttled, human-readable progress for one video."""

    def __init__(
        self,
        progress: ProgressCallback | None,
        video_name: str,
        unit: str,
        total: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        min_fraction_step: float = 0.05,
        min_seconds: float = 2.0,
    ) -> None:
        self.progress = progress
        self.video_name = video_name
        self.unit = unit
        self.total = max(0, total)
        self.clock = clock
        self.min_fraction_step = min_fraction_step
        self.min_seconds = min_seconds
        if self.total > 0:
            self._min_current_step = max(1, int(math.ceil(self.total * self.min_fraction_step)))
        else:
            self._min_current_step = 0
        self._last_current: int | None = None
        self._last_time: float | None = None

    def emit(self, current: int, saved: int, staged: int = 0, *, force: bool = False) -> None:
        if not self.progress:
            return
        current = max(0, current)
        if self.total > 0:
            current = min(current, self.total)
        now = self.clock()
        fraction = current / self.total if self.total > 0 else 0.0
        should_emit = force or self._last_current is None
        if not should_emit and self.total > 0:
            should_emit = current - self._last_current >= self._min_current_step or current >= self.total
        if not should_emit and self._last_time is not None:
            should_emit = now - self._last_time >= self.min_seconds
        if not should_emit:
            return

        self._last_current = current
        self._last_time = now
        if self.total > 0:
            location = f"{self.unit} {current}/{self.total} ({fraction:.0%})"
        else:
            location = f"{self.unit} {current}"
        details = f"{self.video_name}: {location}, saved {saved}"
        if staged > 0:
            details = f"{details}, staged {staged}"
        self.progress(details)


def _estimate_total_frames(cap: cv2.VideoCapture, video: VideoInfo, fps: float) -> int:
    try:
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    except (TypeError, ValueError):
        frame_count = 0.0
    if math.isfinite(frame_count) and frame_count > 0:
        return int(round(frame_count))
    if video.duration_sec > 0 and fps > 0:
        return int(math.ceil(video.duration_sec * fps))
    return 0


def _finalize_group(
    group_frames: list[_PendingFrame],
    video: VideoInfo,
    work_dir: Path,
    output_dir: Path,
    png_compression: int,
    rng: random.Random,
    group_index: int,
    group_seconds_per_keep: float,
    rows: list[dict[str, object]],
) -> tuple[int, int]:
    if not group_frames:
        return group_index, 0
    group_index += 1
    saved_count = 0
    duration = max(0.0, group_frames[-1].timestamp - group_frames[0].timestamp)
    keep_count = max(1, int(math.ceil(duration / max(0.1, group_seconds_per_keep))))
    keep_count = min(keep_count, len(group_frames))
    keep_indices = set(rng.sample(range(len(group_frames)), keep_count))
    for index, item in enumerate(group_frames):
        filename = f"{video.episode_id}_f{item.frame_index:010d}_t{timestamp_token(item.timestamp)}.png"
        output_path = output_dir / filename
        if index in keep_indices:
            success = cv2.imwrite(str(output_path), item.processed, [cv2.IMWRITE_PNG_COMPRESSION, png_compression])
            status = "saved" if success else "error"
            error = "" if success else "cv2.imwrite failed"
            if success:
                saved_count += 1
                rows.append(
                    _base_row(
                        video,
                        item.frame_index,
                        item.timestamp,
                        image=filename,
                        diff_score=item.diff_score,
                        reason=item.reason,
                        output_path=relative_to_or_absolute(output_path, work_dir),
                        status=status,
                        error=error,
                    )
                )
            else:
                rows.append(
                    _base_row(
                        video,
                        item.frame_index,
                        item.timestamp,
                        diff_score=item.diff_score,
                        reason=item.reason,
                        status=status,
                        error=error,
                    )
                )
        else:
            rows.append(
                _base_row(
                    video,
                    item.frame_index,
                    item.timestamp,
                    diff_score=item.diff_score,
                    reason=item.reason,
                    status="skipped_dedup",
                )
            )
    group_frames.clear()
    return group_index, saved_count


def _phash_frame(frame: np.ndarray, params: dict[str, Any]) -> imagehash.ImageHash:
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = _crop_for_hash(image, str(params.get("phash_crop", "center")))
    resize_width = int(params.get("phash_resize_width", 256))
    if resize_width > 0 and image.width > resize_width:
        height = max(1, round(image.height * resize_width / image.width))
        image = image.resize((resize_width, height))
    return imagehash.phash(image, hash_size=int(params.get("phash_size", 8)))


def _crop_for_hash(image: Image.Image, mode: str) -> Image.Image:
    if mode != "center":
        return image
    width, height = image.size
    crop_width = int(width * 0.6)
    crop_height = int(height * 0.6)
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    return image.crop((left, top, left + crop_width, top + crop_height))


def _extract_keyframes_for_video(
    work_dir: Path,
    params: dict[str, Any],
    video: VideoInfo,
    video_path: Path,
    ignore_state: dict[str, Any],
    output_dir: Path,
    stop_state: dict[str, Any] | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[int, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    saved_count = 0

    output_dir.mkdir(parents=True, exist_ok=True)
    ranges = active_ranges_for_episode(ignore_state, video.episode_id)
    with tempfile.TemporaryDirectory(prefix="anime-shot-keyframes-") as temp_name:
        temp_dir = Path(temp_name)
        temp_pattern = temp_dir / "keyframe_%010d.png"
        ok, output = _export_iframes_with_ffmpeg(video_path, temp_pattern, params, stop_state, progress)
        if not ok:
            if not _stop_requested(stop_state):
                rows.append(_error_row(video, output or f"ffmpeg failed for {video_path}"))
            return 0, rows

        exported = sorted(temp_dir.glob("keyframe_*.png"))
        keyframes = _parse_showinfo_keyframes(output, float(video.fps or 0.0))
        if progress:
            progress(f"{video.video_name}: found {len(keyframes)} keyframes")
        if _stop_requested(stop_state):
            return 0, rows
        if not keyframes:
            rows.append(_error_row(video, f"no keyframes found: {video_path}"))
            return 0, rows
        if len(exported) != len(keyframes):
            rows.append(_error_row(video, f"ffmpeg exported {len(exported)} keyframes, showinfo reported {len(keyframes)}"))

        progress_reporter = _VideoProgressReporter(progress, video.video_name, "keyframe", len(keyframes))
        progress_reporter.emit(0, saved_count, force=True)
        for keyframe_index, (keyframe, exported_path) in enumerate(zip(keyframes, exported, strict=False), start=1):
            if _stop_requested(stop_state):
                break
            ignored = match_ignore(keyframe.timestamp, ranges)
            if ignored:
                rows.append(
                    _base_row(
                        video,
                        keyframe.frame_index,
                        keyframe.timestamp,
                        ignored=True,
                        ignore_label=ignored.get("label", ""),
                        ignore_start=ignored.get("start", ""),
                        ignore_end=ignored.get("end", ""),
                        status="skipped_ignore",
                    )
                )
                progress_reporter.emit(keyframe_index, saved_count, force=keyframe_index == len(keyframes))
                continue

            filename = f"{video.episode_id}_f{keyframe.frame_index:010d}_t{timestamp_token(keyframe.timestamp)}.png"
            output_path = output_dir / filename
            try:
                shutil.move(str(exported_path), output_path)
            except OSError as error:
                rows.append(_base_row(video, keyframe.frame_index, keyframe.timestamp, reason="keyframe", status="error", error=str(error)))
                progress_reporter.emit(keyframe_index, saved_count, force=keyframe_index == len(keyframes))
                continue
            saved_count += 1
            rows.append(
                _base_row(
                    video,
                    keyframe.frame_index,
                    keyframe.timestamp,
                    image=filename,
                    reason="keyframe",
                    output_path=relative_to_or_absolute(output_path, work_dir),
                    status="saved",
                )
            )
            progress_reporter.emit(keyframe_index, saved_count, force=keyframe_index == len(keyframes))
    return saved_count, rows


def _export_iframes_with_ffmpeg(
    video_path: Path,
    output_pattern: Path,
    params: dict[str, Any],
    stop_state: dict[str, Any] | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[bool, str]:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "info",
        "-skip_frame",
        "nokey",
        "-i",
        str(video_path),
        "-vsync",
        "0",
    ]
    filters = _ffmpeg_output_filters(params)
    filters.append("showinfo")
    if filters:
        command.extend(["-vf", ",".join(filters)])
    command.extend(
        [
            "-compression_level",
            str(_png_compression_level(params)),
            str(output_pattern),
        ]
    )
    ok, stdout, stderr = _run_subprocess(command, stop_state=stop_state)
    if not ok and progress and stderr:
        progress(process_error_message(f"ffmpeg failed for {video_path}", _CompletedProcessError(command, stderr)))
    return ok, stderr or stdout


_SHOWINFO_INDEX_PATTERN = re.compile(r"\bn:\s*([^\s]+)")
_SHOWINFO_TIMESTAMP_PATTERN = re.compile(r"\bpts_time:\s*([^\s]+)")


def _parse_showinfo_keyframes(output: str, fps: float) -> list[_KeyframeInfo]:
    keyframes: list[_KeyframeInfo] = []
    fallback_index = 0
    for line in output.splitlines():
        if "showinfo" not in line or "pts_time:" not in line:
            continue
        timestamp_match = _SHOWINFO_TIMESTAMP_PATTERN.search(line)
        if timestamp_match is None:
            continue
        try:
            timestamp = float(timestamp_match.group(1))
        except (TypeError, ValueError):
            continue
        index_match = _SHOWINFO_INDEX_PATTERN.search(line)
        try:
            frame_index = int(round(timestamp * fps)) if fps > 0 else int(index_match.group(1) if index_match else fallback_index)
        except (TypeError, ValueError):
            frame_index = fallback_index
        keyframes.append(_KeyframeInfo(frame_index, timestamp))
        fallback_index += 1
    return keyframes


def _ffmpeg_output_filters(params: dict[str, Any]) -> list[str]:
    filters: list[str] = []
    crop_bottom = int(params.get("crop_bottom", 0))
    if crop_bottom > 0:
        filters.append(f"crop=iw:ih-{crop_bottom}:0:0")
    min_width = int(params.get("min_width", 0))
    if min_width > 0:
        filters.append(f"scale='if(lt(iw\\,{min_width})\\,{min_width}\\,iw)':-1")
    return filters


def _png_compression_level(params: dict[str, Any]) -> int:
    return max(0, min(9, int(params.get("png_compression", 3))))


class _CompletedProcessError(subprocess.CalledProcessError):
    def __init__(self, command: list[str], stderr: str):
        super().__init__(returncode=1, cmd=command, stderr=stderr)


def _run_subprocess(command: list[str], stop_state: dict[str, Any] | None = None) -> tuple[bool, str, str]:
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError as error:
        return False, "", str(error)

    while process.poll() is None:
        if _stop_requested(stop_state):
            process.terminate()
            try:
                stdout, stderr = process.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
            return False, stdout, stderr
        try:
            stdout, stderr = process.communicate(timeout=0.2)
            return process.returncode == 0, stdout, stderr
        except subprocess.TimeoutExpired:
            continue

    stdout, stderr = process.communicate()
    return process.returncode == 0, stdout, stderr


def _stop_requested(stop_state: dict[str, Any] | None) -> bool:
    return bool(stop_state and stop_state.get("stop"))


def _prepare_for_output(frame: np.ndarray, crop_bottom: int, min_width: int) -> np.ndarray:
    output = frame
    if crop_bottom > 0 and crop_bottom < output.shape[0]:
        output = output[: output.shape[0] - crop_bottom, :]
    if min_width > 0 and output.shape[1] < min_width:
        scale = min_width / output.shape[1]
        output = cv2.resize(output, (min_width, int(round(output.shape[0] * scale))), interpolation=cv2.INTER_CUBIC)
    return output



def _base_row(
    video: VideoInfo,
    frame_index: int,
    timestamp: float,
    *,
    image: str = "",
    diff_score: float | None = None,
    reason: str = "",
    ignored: bool = False,
    ignore_label: str = "",
    ignore_start: str = "",
    ignore_end: str = "",
    output_path: str = "",
    status: str,
    error: str = "",
) -> dict[str, object]:
    return {
        "episode_id": video.episode_id,
        "video": video.video_name,
        "image": image,
        "frame_index": frame_index,
        "timestamp_sec": round(timestamp, 3),
        "diff_score": "" if diff_score is None else round(diff_score, 4),
        "reason": reason,
        "ignored": ignored,
        "ignore_label": ignore_label,
        "ignore_start": ignore_start,
        "ignore_end": ignore_end,
        "output_path": output_path,
        "status": status,
        "error": error,
    }


def _error_row(video: VideoInfo, error: str) -> dict[str, object]:
    return _base_row(video, 0, 0.0, status="error", error=error)
