import json
import pytest
from pathlib import Path
import numpy as np

from anime_shot_all.config import initialize_work_dir
from anime_shot_all.extract import extract_frames_for_video, extract_frames_for_videos, _PendingFrame, _probe_keyframe_timestamps
from anime_shot_all.video import VideoInfo

def test_extract_frames_grouping(tmp_path: Path, monkeypatch):
    config, _ = initialize_work_dir(tmp_path)
    config["extract"]["interval"] = 1.0
    config["extract"]["group_max_duration"] = 5.0
    config["extract"]["group_seconds_per_keep"] = 2.0
    config["extract"]["phash_threshold"] = 5
    config["extract"]["keyframe_only"] = False
    config["extract"]["crop_bottom"] = 0
    config["extract"]["min_width"] = 0

    video = VideoInfo(episode_id="ep01", video_path="fake.mp4", video_name="fake", duration_sec=10.0, fps=1.0, width=1920, height=1080)
    
    # Mock cv2
    class FakeCap:
        def __init__(self):
            self.frames = 10
            self.current = 0
        def isOpened(self): return True
        def get(self, prop): return 1.0
        def grab(self):
            if self.current < self.frames:
                self.current += 1
                return True
            return False
        def retrieve(self):
            return True, np.zeros((1080, 1920, 3), dtype=np.uint8)
        def release(self): pass

    monkeypatch.setattr("anime_shot_all.extract.cv2.VideoCapture", lambda x: FakeCap())
    monkeypatch.setattr("anime_shot_all.extract.cv2.imwrite", lambda p, f, c: True)

    # Mock phash to return same hash
    monkeypatch.setattr("anime_shot_all.extract._phash_frame", lambda f, p: 0)

    saved_count, rows = extract_frames_for_video(tmp_path, config, video, {}, tmp_path / "out")
    
    # We have 10 frames (1 per sec). All have same hash.
    # Group max duration is 5.0. 
    # Frames 0,1,2,3,4 -> duration 4.0. Next frame at 5.0 breaks the group.
    # Group 1 (0 to 4s): keep ceil(4.0 / 2.0) = 2 frames
    # Group 2 (5 to 9s): keep ceil(4.0 / 2.0) = 2 frames
    # Total saved = 4
    
    assert saved_count == 4


def test_probe_keyframe_timestamps_uses_stable_timestamp_fields(tmp_path: Path, monkeypatch):
    payload = {
        "frames": [
            {"key_frame": 1, "best_effort_timestamp_time": "0.000000"},
            {"key_frame": 0, "best_effort_timestamp_time": "1.000000"},
            {"key_frame": 1, "pts_time": "2.000000"},
            {"key_frame": 1, "pkt_pts_time": "3.000000"},
            {"key_frame": 1, "best_effort_timestamp_time": "N/A"},
            {"key_frame": 1},
        ]
    }

    def fake_run(command, check, capture_output, text):
        assert "frame=key_frame,best_effort_timestamp_time,pts_time,pkt_pts_time" in command
        assert "json" in command
        return type("Result", (), {"stdout": json.dumps(payload)})()

    monkeypatch.setattr("anime_shot_all.extract.subprocess.run", fake_run)

    assert _probe_keyframe_timestamps(tmp_path / "fake.mp4") == [(0, 0.0), (1, 2.0), (2, 3.0)]


def test_extract_keyframes_passes_png_compression(tmp_path: Path, monkeypatch):
    config, _ = initialize_work_dir(tmp_path)
    config["extract"]["keyframe_only"] = True
    config["extract"]["png_compression"] = 7
    config["extract"]["crop_bottom"] = 0
    config["extract"]["min_width"] = 0

    video = VideoInfo(
        episode_id="ep01",
        video_path="fake.mp4",
        video_name="fake",
        duration_sec=2.0,
        fps=1.0,
        width=64,
        height=64,
    )

    class FakeCap:
        def isOpened(self):
            return True

        def set(self, prop, value):
            return True

        def read(self):
            return True, np.zeros((64, 64, 3), dtype=np.uint8)

        def release(self):
            pass

    imwrite_calls = []

    def fake_imwrite(path, frame, params):
        imwrite_calls.append((path, params))
        return True

    monkeypatch.setattr("anime_shot_all.extract.cv2.VideoCapture", lambda path: FakeCap())
    monkeypatch.setattr("anime_shot_all.extract._probe_keyframe_timestamps", lambda path, progress=None: [(0, 0.0)])
    monkeypatch.setattr("anime_shot_all.extract.cv2.imwrite", fake_imwrite)

    saved_count, rows = extract_frames_for_video(tmp_path, config, video, {}, tmp_path / "out")

    assert saved_count == 1
    assert rows[0]["status"] == "saved"
    assert imwrite_calls[0][1] == [pytest.importorskip("cv2").IMWRITE_PNG_COMPRESSION, 7]


def test_extract_frames_for_videos_reports_file_progress(tmp_path: Path, monkeypatch):
    config, _ = initialize_work_dir(tmp_path)
    video = VideoInfo(
        episode_id="ep01",
        video_path="fake.mp4",
        video_name="fake.mp4",
        duration_sec=1.0,
        fps=1.0,
        width=64,
        height=64,
    )
    messages = []

    def fake_extract_frames_for_video(*args, **kwargs):
        return 2, []

    monkeypatch.setattr("anime_shot_all.extract.extract_frames_for_video", fake_extract_frames_for_video)

    saved, log_path, summary = extract_frames_for_videos(tmp_path, config, [video], progress=messages.append)

    assert saved == 2
    assert log_path.exists()
    assert summary == ["ep01: saved 2 frames from fake.mp4"]
    assert "extract 1/1: fake.mp4" in messages
