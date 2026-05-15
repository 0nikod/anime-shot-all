from pathlib import Path

from anime_shot_all.config import deep_merge, initialize_work_dir, resolve_work_path
from anime_shot_all.ignore_ranges import normalize_ranges, rows_to_state, state_to_rows
from anime_shot_all.timecode import parse_timecode


def test_deep_merge_keeps_missing_defaults():
    merged = deep_merge({"a": {"b": 1, "c": 2}, "items": [1]}, {"a": {"b": 3}})
    assert merged == {"a": {"b": 3, "c": 2}, "items": [1]}


def test_initialize_work_dir_creates_project_structure(tmp_path: Path):
    config, messages = initialize_work_dir(tmp_path)
    assert (tmp_path / "configs" / "default.yaml").exists()
    assert (tmp_path / "configs" / "params.yaml").exists()
    assert (tmp_path / "frames_raw").is_dir()
    assert (tmp_path / "videos").is_dir()
    assert config["project"]["work_dir"] == str(tmp_path.resolve())
    assert messages


def test_resolve_work_path_uses_work_dir_for_relative_paths(tmp_path: Path):
    assert resolve_work_path(tmp_path, "frames_raw") == tmp_path / "frames_raw"
    assert resolve_work_path(tmp_path, "/tmp/source").as_posix() == "/tmp/source"


def test_parse_timecode_accepts_required_formats():
    assert parse_timecode("00:01:30") == 90
    assert parse_timecode("01:30") == 90
    assert parse_timecode("90") == 90


def test_ignore_ranges_roundtrip_and_merge(tmp_path: Path):
    rows = [
        ["ep01", "ep01.mkv", "00:00:00", "00:01:30", "OP", True, ""],
        ["ep01", "ep01.mkv", "00:01:20", "00:03:00", "ED", True, ""],
    ]
    state = rows_to_state(rows, tmp_path)
    normalized, warnings, errors = normalize_ranges(state, tmp_path, {"ep01": 1440}, auto_merge=True)
    assert not errors
    assert warnings
    result_rows = state_to_rows(normalized)
    assert result_rows == [["ep01", "ep01.mkv", "00:00:00", "00:03:00", "OP+ED", True, ""]]
