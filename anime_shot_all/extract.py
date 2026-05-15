"""Frame extraction with ignore-range aware scene-diff sampling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import resolve_work_path
from .files import relative_to_or_absolute
from .ignore_ranges import active_ranges_for_episode, load_ignore_state, match_ignore
from .logging_utils import write_csv
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
) -> tuple[int, Path, list[str]]:
    """Extract frames for all videos and write one combined CSV log."""

    output_dir = output_dir or resolve_work_path(work_dir, config["paths"]["frames_raw"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = resolve_work_path(work_dir, config["logging"]["extract_log"])
    ignore_state = load_ignore_state(work_dir)
    rows: list[dict[str, object]] = []
    messages: list[str] = []
    saved_total = 0
    for video in videos:
        saved, video_rows = extract_frames_for_video(work_dir, config, video, ignore_state, output_dir)
        saved_total += saved
        rows.extend(video_rows)
        messages.append(f"{video.episode_id}: saved {saved} frames")
    write_csv(log_path, EXTRACT_LOG_FIELDS, rows)
    return saved_total, log_path, messages


def extract_frames_for_video(
    work_dir: Path,
    config: dict[str, Any],
    video: VideoInfo,
    ignore_state: dict[str, Any],
    output_dir: Path,
) -> tuple[int, list[dict[str, object]]]:
    params = config["extract"]
    video_path = resolve_work_path(work_dir, video.video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0, [_error_row(video, f"cannot open video: {video_path}")]

    fps = cap.get(cv2.CAP_PROP_FPS) or video.fps or 24.0
    frame_step = max(1, int(round(float(params["interval"]) * fps)))
    png_compression = int(params["png_compression"])
    ranges = active_ranges_for_episode(ignore_state, video.episode_id)

    rows: list[dict[str, object]] = []
    saved_count = 0
    previous_diff_frame: np.ndarray | None = None
    previous_saved_timestamp: float | None = None
    was_ignored = False
    first_valid = True

    frame_index = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if frame_index % frame_step != 0:
            frame_index += 1
            continue
        ok, frame = cap.retrieve()
        if not ok:
            rows.append(_base_row(video, frame_index, frame_index / fps, status="error", error="cannot retrieve frame"))
            frame_index += 1
            continue
        timestamp = frame_index / fps
        ignored = match_ignore(timestamp, ranges)
        if ignored:
            was_ignored = True
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
            continue

        if was_ignored and params.get("reset_diff_after_ignore", True):
            previous_diff_frame = None
            previous_saved_timestamp = None
            was_ignored = False

        processed = _prepare_for_output(frame, int(params["crop_bottom"]), int(params["min_width"]))
        diff_frame = _prepare_for_diff(processed, int(params["resize_width_for_diff"]))
        diff_score = None if previous_diff_frame is None else _gray_mean_absdiff(previous_diff_frame, diff_frame)
        force_gap = previous_saved_timestamp is not None and (timestamp - previous_saved_timestamp) >= float(params["max_gap"])

        if first_valid:
            reason = "first"
        elif previous_diff_frame is None:
            reason = "first_after_ignore"
        elif force_gap:
            reason = "force_gap"
        elif diff_score is not None and diff_score >= float(params["diff_threshold"]):
            reason = "diff"
        else:
            reason = ""

        if reason:
            filename = f"{video.episode_id}_f{frame_index:010d}_t{timestamp_token(timestamp)}.png"
            output_path = output_dir / filename
            success = cv2.imwrite(str(output_path), processed, [cv2.IMWRITE_PNG_COMPRESSION, png_compression])
            status = "saved" if success else "error"
            error = "" if success else "cv2.imwrite failed"
            if success:
                saved_count += 1
                previous_saved_timestamp = timestamp
                previous_diff_frame = diff_frame
                first_valid = False
            rows.append(
                _base_row(
                    video,
                    frame_index,
                    timestamp,
                    image=filename if success else "",
                    diff_score=diff_score,
                    reason=reason,
                    output_path=relative_to_or_absolute(output_path, work_dir) if success else "",
                    status=status,
                    error=error,
                )
            )
        else:
            rows.append(_base_row(video, frame_index, timestamp, diff_score=diff_score, status="skipped_diff"))
        frame_index += 1

    cap.release()
    return saved_count, rows


def _prepare_for_output(frame: np.ndarray, crop_bottom: int, min_width: int) -> np.ndarray:
    output = frame
    if crop_bottom > 0 and crop_bottom < output.shape[0]:
        output = output[: output.shape[0] - crop_bottom, :]
    if min_width > 0 and output.shape[1] < min_width:
        scale = min_width / output.shape[1]
        output = cv2.resize(output, (min_width, int(round(output.shape[0] * scale))), interpolation=cv2.INTER_CUBIC)
    return output


def _prepare_for_diff(frame: np.ndarray, resize_width: int) -> np.ndarray:
    if resize_width > 0 and frame.shape[1] != resize_width:
        scale = resize_width / frame.shape[1]
        frame = cv2.resize(frame, (resize_width, max(1, int(round(frame.shape[0] * scale)))), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _gray_mean_absdiff(previous: np.ndarray, current: np.ndarray) -> float:
    if previous.shape != current.shape:
        current = cv2.resize(current, (previous.shape[1], previous.shape[0]), interpolation=cv2.INTER_AREA)
    return float(np.mean(cv2.absdiff(previous, current)))


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
