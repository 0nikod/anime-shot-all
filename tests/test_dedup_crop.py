from pathlib import Path
import random

from PIL import Image

from anime_shot_all.config import initialize_work_dir
from anime_shot_all.crop import crop_one_image, run_crop
from anime_shot_all.dedup import analyze_duplicates, export_dedup_results, update_group_decision


def _save_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (320, 180), color).save(path)


def test_dedup_analysis_and_manual_export(tmp_path: Path):
    config, _ = initialize_work_dir(tmp_path)
    raw = tmp_path / "frames_raw"
    _save_image(raw / "ep01_f0000000001_t000001.000.png", (255, 0, 0))
    _save_image(raw / "ep01_f0000000002_t000002.000.png", (255, 0, 0))
    _save_image(raw / "ep02_f0000000001_t000001.000.png", (0, 255, 0))

    config["dedup"]["hash_threshold"] = 0
    config["dedup"]["dedup_scope"] = "per_episode"
    state, state_path = analyze_duplicates(tmp_path, config, raw)

    assert state_path.exists()
    assert len(state["groups"]) == 1
    group = state["groups"][0]
    duplicate_path = group["images"][1]["path"]
    state = update_group_decision(state, group["group_id"], [duplicate_path])
    kept, rejected, log_path = export_dedup_results(tmp_path, config, state)

    assert kept == 2
    assert rejected == 1
    assert log_path.exists()
    assert len(list((tmp_path / "frames_dedup").glob("*.png"))) == 2
    assert len(list((tmp_path / "rejected_duplicates").glob("*.png"))) == 1


def test_hard_and_random_crop_outputs_are_png(tmp_path: Path):
    config, _ = initialize_work_dir(tmp_path)
    dedup = tmp_path / "frames_dedup"
    _save_image(dedup / "ep01_f0000000001_t000001.000.png", (10, 20, 30))

    config["crop_types"] = {
        "full": False,
        "hard_split": True,
        "face": False,
        "body": False,
        "halfbody": False,
        "background": False,
        "random_crop": True,
    }
    config["hard_split"] = {
        "left_square": False,
        "center_square": True,
        "right_square": False,
        "center_portrait": False,
        "upper_landscape": False,
        "lower_landscape": False,
    }
    config["crop"]["input_dir"] = "frames_dedup"
    config["crop"]["output_dir"] = "crops"
    config["crop"]["min_crop_size"] = 32
    config["random_crop"]["count_per_image"] = 1

    saved, log_path = run_crop(tmp_path, config)

    assert saved == 2
    assert log_path.exists()
    assert len(list((tmp_path / "crops").rglob("*.png"))) == 2


def test_run_crop_reports_file_progress(tmp_path: Path):
    config, _ = initialize_work_dir(tmp_path)
    image_path = tmp_path / "frames_raw" / "ep01.png"
    _save_image(image_path, (255, 255, 255))
    config["crop"]["input_dir"] = "frames_raw"
    config["crop_types"] = {
        "full": True,
        "hard_split": False,
        "face": False,
        "body": False,
        "halfbody": False,
        "background": False,
        "random_crop": False,
    }
    messages = []

    saved, log_path = run_crop(tmp_path, config, progress=messages.append)

    assert saved == 1
    assert log_path.exists()
    assert "crop 1/1: ep01.png" in messages
    assert "ep01.png: saved 1 crops, total 1" in messages


def test_crop_skips_detection_when_semantic_types_are_disabled(tmp_path: Path):
    config, _ = initialize_work_dir(tmp_path)
    image_path = tmp_path / "frames_dedup" / "ep01_f0000000001_t000001.000.png"
    _save_image(image_path, (10, 20, 30))
    config["crop_types"] = {
        "full": True,
        "hard_split": False,
        "face": False,
        "body": False,
        "halfbody": False,
        "background": False,
        "random_crop": False,
    }

    saved, rows = crop_one_image(
        tmp_path,
        config,
        image_path,
        tmp_path / "crops",
        random.Random(42),
    )

    assert saved == 1
    assert rows[0]["crop_type"] == "full"


def test_imgutils_detection_outputs_body_face_and_halfbody(tmp_path: Path, monkeypatch):
    config, _ = initialize_work_dir(tmp_path)
    image_path = tmp_path / "frames_dedup" / "ep01_f0000000001_t000001.000.png"
    _save_image(image_path, (40, 50, 60))
    config["crop_types"] = {
        "full": False,
        "hard_split": False,
        "face": True,
        "body": True,
        "halfbody": True,
        "background": False,
        "random_crop": False,
    }
    config["crop"]["min_crop_size"] = 16
    config["body_crop"]["min_size"] = 16
    config["face_crop"]["min_size"] = 16
    config["halfbody_crop"]["min_size"] = 16

    calls = []

    def fake_detector(image, kind, level, version, conf_threshold, iou_threshold):
        calls.append(kind)
        if kind == "person":
            return [((20, 10, 220, 170), "person", 0.9)]
        if kind == "face":
            return [((50, 20, 110, 80), "face", 0.8)]
        if kind == "halfbody":
            return [((10, 5, 150, 120), "halfbody", 0.85)]
        return []

    monkeypatch.setattr("anime_shot_all.crop._call_imgutils_detector", fake_detector)

    saved, rows = crop_one_image(tmp_path, config, image_path, tmp_path / "crops", random.Random(42))

    assert saved == 3
    assert {"person", "face", "halfbody"}.issubset(set(calls))
    assert (tmp_path / "crops" / "body").exists()
    assert (tmp_path / "crops" / "face").exists()
    assert (tmp_path / "crops" / "halfbody").exists()
    assert {row["crop_type"] for row in rows if row["status"] == "saved"} == {"body", "face", "halfbody"}
