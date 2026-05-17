from pathlib import Path
import csv
import random
from collections import Counter

from PIL import Image

from anime_shot_all.config import initialize_work_dir
from anime_shot_all.crop import (
    MAX_AREA,
    MIN_AREA,
    CROP_LOG_FIELDS,
    OUTPUT_SIZE_PRESETS,
    _choose_output_size_for_crop,
    _collect_detections,
    _cover_and_clamp_box,
    _rank_ratios_by_bbox,
    _ratio_weight,
    _weighted_ratio_order,
    crop_one_image,
    run_crop,
)
from anime_shot_all.dedup import analyze_duplicates, export_dedup_results, update_group_decision


def _save_image(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (320, 180)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


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
    assert len(list(raw.glob("*.png"))) == 3
    assert len(list((tmp_path / "rejected_duplicates").glob("*.png"))) == 1


def test_random_crop_outputs_preset_area_png(tmp_path: Path):
    config, _ = initialize_work_dir(tmp_path)
    raw = tmp_path / "frames_raw"
    _save_image(raw / "ep01_f0000000001_t000001.000.png", (10, 20, 30), (1280, 720))

    config["crop_types"] = {
        "full": False,
        "face": False,
        "body": False,
        "halfbody": False,
        "random_crop": True,
    }
    config["crop"]["input_dir"] = "frames_raw"
    config["crop"]["output_dir"] = "crops"
    config["crop"]["min_crop_size"] = 32
    config["random_crop"]["count_per_image"] = 1

    saved, log_path = run_crop(tmp_path, config)

    assert saved == 1
    assert log_path.exists()
    outputs = list((tmp_path / "crops" / "random_crop").glob("*.png"))
    assert len(outputs) == 1
    with Image.open(outputs[0]) as output:
        assert MIN_AREA <= output.width * output.height <= MAX_AREA

    with log_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["source_crop_area"]
    assert "source_crop_area" in rows[0]
    assert "expand_left" in rows[0]
    assert "expand_top" in rows[0]
    assert "expand_right" in rows[0]
    assert "expand_bottom" in rows[0]
    assert "producer_type" not in rows[0]
    assert "padding_x" not in rows[0]
    assert "padding_y" not in rows[0]
    assert "fallback_used" not in rows[0]
    assert "fallback_reason" not in rows[0]
    assert "random_seed" not in rows[0]
    assert "aspect_mode" not in rows[0]


def test_run_crop_reports_file_progress(tmp_path: Path):
    config, _ = initialize_work_dir(tmp_path)
    image_path = tmp_path / "frames_raw" / "ep01.png"
    _save_image(image_path, (255, 255, 255), (1920, 1080))
    config["crop"]["input_dir"] = "frames_raw"
    config["crop_types"] = {
        "full": True,
        "face": False,
        "body": False,
        "halfbody": False,
        "random_crop": False,
    }
    messages = []

    saved, log_path = run_crop(tmp_path, config, progress=messages.append)

    assert saved == 1
    assert log_path.exists()
    assert "crop 1/1: ep01.png" in messages
    assert "ep01.png: saved 1 crops, total 1" in messages
    with log_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["reason"] == "full_resize"
    assert int(rows[0]["source_crop_area"]) == 1920 * 1080


def test_crop_skips_detection_when_semantic_types_are_disabled(tmp_path: Path):
    config, _ = initialize_work_dir(tmp_path)
    image_path = tmp_path / "frames_raw" / "ep01_f0000000001_t000001.000.png"
    _save_image(image_path, (10, 20, 30), (1920, 1080))
    config["crop_types"] = {
        "full": True,
        "face": False,
        "body": False,
        "halfbody": False,
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


def test_crop_log_fields_use_bbox_first_contract():
    removed = {
        "producer_type",
        "padding_x",
        "padding_y",
        "fallback_used",
        "fallback_reason",
        "random_seed",
        "aspect_mode",
    }

    assert removed.isdisjoint(CROP_LOG_FIELDS)
    assert {
        "source_crop_area",
        "expand_left",
        "expand_top",
        "expand_right",
        "expand_bottom",
    }.issubset(CROP_LOG_FIELDS)


def test_imgutils_detection_outputs_body_face_and_halfbody(tmp_path: Path, monkeypatch):
    config, _ = initialize_work_dir(tmp_path)
    image_path = tmp_path / "frames_raw" / "ep01_f0000000001_t000001.000.png"
    _save_image(image_path, (40, 50, 60))
    config["crop_types"] = {
        "full": False,
        "face": True,
        "body": True,
        "halfbody": True,
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
    assert all(MIN_AREA <= int(row["output_area"]) <= MAX_AREA for row in rows if row["status"] == "saved")
    assert all(row["selected_ratio"] in {"9:16", "3:4", "1:1", "4:3", "16:9"} for row in rows if row["status"] == "saved")


def test_detection_triggers_only_required_detectors(tmp_path: Path, monkeypatch):
    config, _ = initialize_work_dir(tmp_path)
    image_path = tmp_path / "frames_raw" / "ep01.png"
    _save_image(image_path, (1, 2, 3), (640, 960))
    image = Image.new("RGB", (640, 960), (1, 2, 3))
    calls = []

    def fake_detector(image_or_path, kind, level, version, conf_threshold, iou_threshold):
        calls.append(kind)
        if kind == "person":
            return [((100, 100, 500, 900), "person", 0.9)]
        if kind == "face":
            return [((200, 120, 320, 240), "face", 0.8)]
        if kind == "halfbody":
            return [((40, 40, 300, 520), "halfbody", 0.85)]
        return []

    monkeypatch.setattr("anime_shot_all.crop._call_imgutils_detector", fake_detector)

    config["crop_types"] = {"full": False, "face": True, "body": False, "halfbody": False, "random_crop": False}
    calls.clear()
    _collect_detections(image, image_path, config)
    assert calls == ["face"]

    config["crop_types"] = {"full": False, "face": False, "body": True, "halfbody": False, "random_crop": False}
    calls.clear()
    _collect_detections(image, image_path, config)
    assert calls == ["person"]

    config["crop_types"] = {"full": False, "face": False, "body": False, "halfbody": True, "random_crop": False}
    calls.clear()
    _collect_detections(image, image_path, config)
    assert calls == ["person", "halfbody"]

    config["crop_types"] = {"full": False, "face": False, "body": False, "halfbody": False, "random_crop": True}
    config["random_crop"]["avoid_body"] = True
    calls.clear()
    _collect_detections(image, image_path, config)
    assert calls == ["person"]


def test_ratio_candidates_filter_extreme_reverse_aspects(tmp_path: Path):
    config, _ = initialize_work_dir(tmp_path)

    assert "16:9" not in _rank_ratios_by_bbox(0.5, config)
    assert "9:16" not in _rank_ratios_by_bbox(2.0, config)
    assert "1:1" in _rank_ratios_by_bbox(0.5, config)
    assert "1:1" in _rank_ratios_by_bbox(2.0, config)


def test_ratio_weight_prefers_nearest_bbox_ratio(tmp_path: Path):
    config, _ = initialize_work_dir(tmp_path)
    expected = {
        9 / 16: "9:16",
        3 / 4: "3:4",
        1.0: "1:1",
        4 / 3: "4:3",
        16 / 9: "16:9",
    }

    for bbox_ratio, nearest in expected.items():
        weights = {name: _ratio_weight(name, bbox_ratio, config) for name in _rank_ratios_by_bbox(bbox_ratio, config)}
        assert max(weights, key=weights.get) == nearest


def test_weighted_ratio_order_statistically_prefers_nearest_ratio(tmp_path: Path):
    config, _ = initialize_work_dir(tmp_path)
    rng = random.Random(42)
    first_choices = [
        _weighted_ratio_order(_rank_ratios_by_bbox(3 / 4, config), 3 / 4, config, rng)[0]
        for _ in range(1000)
    ]

    assert Counter(first_choices).most_common(1)[0][0] == "3:4"


def test_size_selection_prefers_source_crop_area_bucket(tmp_path: Path):
    config, _ = initialize_work_dir(tmp_path)
    config["size_selection"]["sigma"] = 0.2
    rng = random.Random(42)

    small_choices = [
        _choose_output_size_for_crop("4:3", (0, 0, 500, 500), config, rng)
        for _ in range(500)
    ]
    medium_choices = [
        _choose_output_size_for_crop("4:3", (0, 0, 1280, 1024), config, rng)
        for _ in range(500)
    ]
    large_choices = [
        _choose_output_size_for_crop("4:3", (0, 0, 2500, 1600), config, rng)
        for _ in range(500)
    ]

    valid_4_3 = [(w, h) for w, h in OUTPUT_SIZE_PRESETS["4:3"] if MIN_AREA <= w * h <= MAX_AREA]
    assert Counter(small_choices).most_common(1)[0][0] == min(valid_4_3, key=lambda size: size[0] * size[1])
    assert Counter(medium_choices).most_common(1)[0][0] == (1280, 1024)
    assert Counter(large_choices).most_common(1)[0][0] == max(valid_4_3, key=lambda size: size[0] * size[1])


def test_cover_and_clamp_box_never_shrinks_float_bbox():
    box = (-2.2, 3.5, 99.1, 100.01)
    covered = _cover_and_clamp_box(box, 80, 90)

    assert covered == (0, 3, 80, 90)
    assert covered[0] <= max(0, box[0])
    assert covered[1] <= box[1]
    assert covered[2] >= min(80, box[2])
    assert covered[3] >= min(90, box[3])


def test_face_does_not_use_body_or_halfbody_when_face_detection_is_empty(tmp_path: Path, monkeypatch):
    config, _ = initialize_work_dir(tmp_path)
    image_path = tmp_path / "frames_raw" / "ep01_f0000000001_t000001.000.png"
    _save_image(image_path, (40, 50, 60), (640, 960))
    config["crop_types"] = {
        "full": False,
        "face": True,
        "body": False,
        "halfbody": False,
        "random_crop": False,
    }
    config["crop"]["min_crop_size"] = 16
    config["halfbody_crop"]["min_size"] = 16
    config["body_crop"]["min_size"] = 16
    config["face_crop"]["min_size"] = 16

    def fake_detector(image, kind, level, version, conf_threshold, iou_threshold):
        if kind == "person":
            return [((160, 120, 480, 900), "person", 0.9)]
        if kind == "face":
            return []
        if kind == "halfbody":
            return [((40, 40, 260, 520), "halfbody", 0.85)]
        return []

    monkeypatch.setattr("anime_shot_all.crop._call_imgutils_detector", fake_detector)

    saved, rows = crop_one_image(tmp_path, config, image_path, tmp_path / "crops", random.Random(42))

    assert saved == 0
    assert rows[-1]["status"] == "skipped"
    assert rows[-1]["reason"] == "no_candidate"
