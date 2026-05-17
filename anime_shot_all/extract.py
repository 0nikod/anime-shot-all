"""Frame extraction with ignore-range aware scene-diff sampling."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import random
import subprocess
from pathlib import Path
from typing import Any

import cv2
import imagehash
import numpy as np
from PIL import Image

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
    stop_state: dict[str, Any] | None = None,
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
        saved, video_rows = extract_frames_for_video(
            work_dir,
            config,
            video,
            ignore_state,
            output_dir,
            stop_state=stop_state,
        )
        saved_total += saved
        rows.extend(video_rows)
        messages.append(f"{video.episode_id}: saved {saved} frames")
        if stop_state and stop_state.get("stop"):
            messages.append("stopped by user")
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
) -> tuple[int, list[dict[str, object]]]:
    params = config["extract"]
    video_path = resolve_work_path(work_dir, video.video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0, [_error_row(video, f"cannot open video: {video_path}")]

    png_compression = int(params["png_compression"])
    ranges = active_ranges_for_episode(ignore_state, video.episode_id)
    keyframe_only = bool(params.get("keyframe_only", False))
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

    if keyframe_only:
        # keyframe mode: interval/max_gap/pHash dedup are ignored.
        keyframes = _probe_keyframe_timestamps(video_path)
        if not keyframes:
            rows.append(_error_row(video, f"no keyframes found: {video_path}"))
        for frame_index, timestamp in keyframes:
            if stop_state and stop_state.get("stop"):
                break
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
                continue
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
            ok, frame = cap.read()
            if not ok:
                rows.append(_base_row(video, frame_index, timestamp, status="error", error="cannot retrieve keyframe"))
                continue
            processed = _prepare_for_output(frame, int(params["crop_bottom"]), int(params["min_width"]))
            filename = f"{video.episode_id}_f{frame_index:010d}_t{timestamp_token(timestamp)}.png"
            output_path = output_dir / filename
            success = cv2.imwrite(str(output_path), processed, [cv2.IMWRITE_PNG_COMPRESSION, png_compression])
            status = "saved" if success else "error"
            error = "" if success else "cv2.imwrite failed"
            if success:
                saved_count += 1
            rows.append(
                _base_row(
                    video,
                    frame_index,
                    timestamp,
                    image=filename if success else "",
                    reason="keyframe",
                    output_path=relative_to_or_absolute(output_path, work_dir) if success else "",
                    status=status,
                    error=error,
                )
            )
    else:
        fps = cap.get(cv2.CAP_PROP_FPS) or video.fps or 24.0
        frame_step = max(1, int(round(float(params["interval"]) * fps)))
        frame_index = 0
        while True:
            if stop_state and stop_state.get("stop"):
                break
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
    return saved_count, rows


@dataclass
class _PendingFrame:
    frame_index: int
    timestamp: float
    processed: np.ndarray
    diff_score: float | None
    reason: str


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


def _probe_keyframe_timestamps(video_path: Path) -> list[tuple[int, float]]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_frames",
        "-show_entries",
        "frame=key_frame,best_effort_timestamp_time,pts_time,pkt_pts_time",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        payload = json.loads(result.stdout)
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
        return []
    timestamps: list[tuple[int, float]] = []
    index = 0
    for frame in payload.get("frames", []):
        if str(frame.get("key_frame", "")) != "1":
            continue
        ts_text = (
            frame.get("best_effort_timestamp_time")
            or frame.get("pts_time")
            or frame.get("pkt_pts_time")
        )
        if ts_text is None:
            continue
        try:
            timestamp = float(ts_text)
        except (TypeError, ValueError):
            continue
        timestamps.append((index, timestamp))
        index += 1
    return timestamps


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
