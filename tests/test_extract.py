from pathlib import Path
import numpy as np

from anime_shot_all.config import initialize_work_dir
from anime_shot_all import extract as extract_module
from anime_shot_all.extract import extract_frames_for_video, extract_frames_for_videos, _parse_showinfo_keyframes
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
        def get(self, prop):
            if prop == extract_module.cv2.CAP_PROP_FPS:
                return 1.0
            if prop == extract_module.cv2.CAP_PROP_FRAME_COUNT:
                return self.frames
            return 0.0
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

    messages = []
    saved_count, rows = extract_frames_for_video(tmp_path, config, video, {}, tmp_path / "out", progress=messages.append)
    
    # We have 10 frames (1 per sec). All have same hash.
    # Group max duration is 5.0. 
    # Frames 0,1,2,3,4 -> duration 4.0. Next frame at 5.0 breaks the group.
    # Group 1 (0 to 4s): keep ceil(4.0 / 2.0) = 2 frames
    # Group 2 (5 to 9s): keep ceil(4.0 / 2.0) = 2 frames
    # Total saved = 4
    
    assert saved_count == 4
    assert any("fake: frame 5/10 (50%)" in message for message in messages)
    assert any("saved 4" in message for message in messages)


def test_parse_showinfo_keyframes_uses_timestamp_and_fps():
    output = "\n".join(
        [
            "[Parsed_showinfo_1 @ 0x1] n:   0 pts:0 pts_time:0 pos:1",
            "noise",
            "[Parsed_showinfo_1 @ 0x1] n:   1 pts:1000 pts_time:2.000000 pos:2",
            "[Parsed_showinfo_1 @ 0x1] n:   2 pts:2000 pts_time:N/A pos:3",
        ]
    )

    keyframes = _parse_showinfo_keyframes(output, fps=24.0)

    assert [(item.frame_index, item.timestamp) for item in keyframes] == [(0, 0.0), (48, 2.0)]


def test_extract_keyframes_uses_ffmpeg_iframe_export_and_png_compression(tmp_path: Path, monkeypatch):
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
        fps=24.0,
        width=64,
        height=64,
    )

    commands = []

    def fake_run_subprocess(command, stop_state=None):
        commands.append(command)
        if command[0] == "ffmpeg":
            output_pattern = Path(command[-1])
            assert "-skip_frame" in command
            assert "nokey" in command
            assert "showinfo" in command[command.index("-vf") + 1]
            assert command[command.index("-compression_level") + 1] == "7"
            output_pattern.parent.mkdir(parents=True, exist_ok=True)
            (output_pattern.parent / "keyframe_0000000001.png").write_bytes(b"png")
            return True, "", "[Parsed_showinfo_1 @ 0x1] n: 0 pts:500 pts_time:0.5 pos:1"
        raise AssertionError(command)

    monkeypatch.setattr("anime_shot_all.extract._run_subprocess", fake_run_subprocess)

    saved_count, rows = extract_frames_for_video(tmp_path, config, video, {}, tmp_path / "out")

    assert saved_count == 1
    assert rows[0]["status"] == "saved"
    assert rows[0]["frame_index"] == 12
    assert (tmp_path / "out" / "ep01_f0000000012_t000000.500.png").read_bytes() == b"png"
    assert [command[0] for command in commands] == ["ffmpeg"]


def test_extract_keyframes_reports_video_progress(tmp_path: Path, monkeypatch):
    config, _ = initialize_work_dir(tmp_path)
    config["extract"]["keyframe_only"] = True
    config["extract"]["crop_bottom"] = 0
    config["extract"]["min_width"] = 0

    video = VideoInfo(
        episode_id="ep01",
        video_path="fake.mp4",
        video_name="fake",
        duration_sec=3.0,
        fps=1.0,
        width=64,
        height=64,
    )

    def fake_run_subprocess(command, stop_state=None):
        if command[0] == "ffmpeg":
            output_pattern = Path(command[-1])
            output_pattern.parent.mkdir(parents=True, exist_ok=True)
            for index in range(1, 4):
                (output_pattern.parent / f"keyframe_{index:010d}.png").write_bytes(b"png")
            return True, "", "\n".join(
                [
                    "[Parsed_showinfo_1 @ 0x1] n: 0 pts:0 pts_time:0.000000 pos:1",
                    "[Parsed_showinfo_1 @ 0x1] n: 1 pts:1 pts_time:1.000000 pos:2",
                    "[Parsed_showinfo_1 @ 0x1] n: 2 pts:2 pts_time:2.000000 pos:3",
                ]
            )
        raise AssertionError(command)

    monkeypatch.setattr("anime_shot_all.extract._run_subprocess", fake_run_subprocess)

    messages = []
    saved_count, _rows = extract_frames_for_video(tmp_path, config, video, {}, tmp_path / "out", progress=messages.append)

    assert saved_count == 3
    assert any("fake: keyframe 1/3 (33%)" in message for message in messages)
    assert any("fake: keyframe 3/3 (100%), saved 3" in message for message in messages)


def test_extract_keyframes_filters_ignored_showinfo_frames(tmp_path: Path, monkeypatch):
    config, _ = initialize_work_dir(tmp_path)
    config["extract"]["keyframe_only"] = True
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
    ignore_state = {
        "episodes": [
            {
                "episode_id": "ep01",
                "video_path": "fake.mp4",
                "ignore_ranges": [{"start": "1.0", "end": "2.0", "label": "op", "enabled": True}],
            }
        ]
    }

    def fake_run_subprocess(command, stop_state=None):
        output_pattern = Path(command[-1])
        output_pattern.parent.mkdir(parents=True, exist_ok=True)
        for index in range(1, 3):
            (output_pattern.parent / f"keyframe_{index:010d}.png").write_bytes(f"png{index}".encode())
        return True, "", "\n".join(
            [
                "[Parsed_showinfo_1 @ 0x1] n: 0 pts:0 pts_time:0.000000 pos:1",
                "[Parsed_showinfo_1 @ 0x1] n: 1 pts:1 pts_time:1.000000 pos:2",
            ]
        )

    monkeypatch.setattr("anime_shot_all.extract._run_subprocess", fake_run_subprocess)

    saved_count, rows = extract_frames_for_video(tmp_path, config, video, ignore_state, tmp_path / "out")

    assert saved_count == 1
    assert [row["status"] for row in rows] == ["saved", "skipped_ignore"]
    assert rows[1]["ignore_label"] == "op"
    assert sorted(path.name for path in (tmp_path / "out").glob("*.png")) == ["ep01_f0000000000_t000000.000.png"]


def test_extract_frame_progress_is_throttled(tmp_path: Path, monkeypatch):
    config, _ = initialize_work_dir(tmp_path)
    config["extract"]["interval"] = 1.0
    config["extract"]["group_max_duration"] = 1000.0
    config["extract"]["group_seconds_per_keep"] = 1000.0
    config["extract"]["keyframe_only"] = False
    config["extract"]["crop_bottom"] = 0
    config["extract"]["min_width"] = 0

    video = VideoInfo(
        episode_id="ep01",
        video_path="fake.mp4",
        video_name="fake",
        duration_sec=100.0,
        fps=1.0,
        width=64,
        height=64,
    )

    class FakeCap:
        def __init__(self):
            self.frames = 100
            self.current = 0

        def isOpened(self):
            return True

        def get(self, prop):
            if prop == extract_module.cv2.CAP_PROP_FPS:
                return 1.0
            if prop == extract_module.cv2.CAP_PROP_FRAME_COUNT:
                return self.frames
            return 0.0

        def grab(self):
            if self.current < self.frames:
                self.current += 1
                return True
            return False

        def retrieve(self):
            return True, np.zeros((64, 64, 3), dtype=np.uint8)

        def release(self):
            pass

    monkeypatch.setattr("anime_shot_all.extract.cv2.VideoCapture", lambda path: FakeCap())
    monkeypatch.setattr("anime_shot_all.extract._phash_frame", lambda frame, params: 0)
    monkeypatch.setattr("anime_shot_all.extract.cv2.imwrite", lambda *args: True)

    messages = []
    extract_frames_for_video(tmp_path, config, video, {}, tmp_path / "out", progress=messages.append)

    frame_messages = [message for message in messages if "fake: frame" in message]
    assert len(frame_messages) < 100
    assert any("fake: frame 50/100 (50%)" in message for message in frame_messages)
    assert any("fake: frame 100/100 (100%)" in message for message in frame_messages)


def test_extract_stop_does_not_flush_pending_group(tmp_path: Path, monkeypatch):
    config, _ = initialize_work_dir(tmp_path)
    config["extract"]["interval"] = 1.0
    config["extract"]["keyframe_only"] = False
    config["extract"]["crop_bottom"] = 0
    config["extract"]["min_width"] = 0

    video = VideoInfo(
        episode_id="ep01",
        video_path="fake.mp4",
        video_name="fake",
        duration_sec=10.0,
        fps=1.0,
        width=64,
        height=64,
    )
    stop_state = {"stop": False}

    class FakeCap:
        def __init__(self):
            self.current = 0

        def isOpened(self):
            return True

        def get(self, prop):
            return 1.0

        def grab(self):
            if self.current == 0:
                self.current += 1
                return True
            stop_state["stop"] = True
            return True

        def retrieve(self):
            return True, np.zeros((64, 64, 3), dtype=np.uint8)

        def release(self):
            pass

    imwrite_calls = []

    monkeypatch.setattr("anime_shot_all.extract.cv2.VideoCapture", lambda path: FakeCap())
    monkeypatch.setattr("anime_shot_all.extract._phash_frame", lambda frame, params: 0)
    monkeypatch.setattr("anime_shot_all.extract.cv2.imwrite", lambda *args: imwrite_calls.append(args) or True)

    saved_count, rows = extract_frames_for_video(tmp_path, config, video, {}, tmp_path / "out", stop_state=stop_state)

    assert saved_count == 0
    assert rows == []
    assert imwrite_calls == []


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
