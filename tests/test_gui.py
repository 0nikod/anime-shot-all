import gradio as gr

from anime_shot_all.gui import _scan_videos, build_app


def test_gui_builds_blocks():
    app = build_app()

    assert isinstance(app, gr.Blocks)


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
