import gradio as gr

from anime_shot_all.config import initialize_work_dir
from anime_shot_all.gui import _request_stop, _run_extract, _scan_videos, _values_from_config, build_app


def test_gui_builds_blocks():
    app = build_app()

    assert isinstance(app, gr.Blocks)


def test_request_stop_mutates_existing_state():
    stop_state = {"stop": False}

    returned, message = _request_stop(stop_state)

    assert returned is stop_state
    assert stop_state == {"stop": True}
    assert message == "stop requested"


def test_scan_videos_reports_missing_directory(tmp_path):
    missing = tmp_path / "missing"

    videos, rows, message, _ = list(_scan_videos(str(tmp_path), str(missing), {}))[-1]

    assert videos == []
    assert rows == []
    assert "视频文件夹不存在" in message
    assert str(missing) in message


def test_scan_videos_uses_defaults_when_config_is_empty(tmp_path, monkeypatch):
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    captured = {}

    def fake_video_candidates(path, supported_ext):
        captured["path"] = path
        captured["supported_ext"] = supported_ext
        return []

    monkeypatch.setattr("anime_shot_all.gui.video_candidates", fake_video_candidates)

    videos, rows, message, _ = list(_scan_videos(str(tmp_path), str(video_dir), {}))[-1]

    assert videos == []
    assert rows == []
    assert "scanned 0 videos" in message
    assert captured["path"] == video_dir
    assert ".mkv" in captured["supported_ext"]


def test_run_extract_prints_progress_to_terminal(tmp_path, monkeypatch, capsys):
    config, _ = initialize_work_dir(tmp_path)
    video = {
        "episode_id": "ep01",
        "video_path": "fake.mp4",
        "video_name": "fake.mp4",
        "duration_sec": 1.0,
        "fps": 1.0,
        "width": 64,
        "height": 64,
    }

    def fake_extract_frames_for_video(*args, **kwargs):
        kwargs["progress"]("fake.mp4: keyframe 1/1 (100%), saved 1")
        return 1, []

    monkeypatch.setattr("anime_shot_all.gui.extract_frames_for_video", fake_extract_frames_for_video)

    outputs = list(_run_extract(str(tmp_path), config, [video], [], {"stop": False}, *_values_from_config(config)))
    final_log = outputs[-1][1]
    terminal_output = capsys.readouterr().out

    assert "fake.mp4: keyframe 1/1 (100%), saved 1" in final_log
    assert "fake.mp4: keyframe 1/1 (100%), saved 1" in terminal_output
