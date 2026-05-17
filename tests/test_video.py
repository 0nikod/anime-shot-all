import subprocess
from pathlib import Path

from anime_shot_all.video import scan_videos


def test_scan_videos_reports_ffprobe_error_and_continues(tmp_path: Path, monkeypatch):
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    bad = video_dir / "bad.mkv"
    bad.write_bytes(b"not a video")
    messages = []

    def fake_run(command, check, capture_output, text):
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=command,
            stderr="[matroska,webm @ 0000026fdf687940] Unsupported encoding type\rno newline",
        )

    monkeypatch.setattr("anime_shot_all.video.subprocess.run", fake_run)

    videos = scan_videos(video_dir, tmp_path, [".mkv"], progress=messages.append)

    assert videos == []
    assert any("scan 1/1: bad.mkv" in message for message in messages)
    assert any("Unsupported encoding type\nno newline" in message for message in messages)
