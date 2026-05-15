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


def test_crop_skips_detection_when_semantic_types_are_disabled(tmp_path: Path):
    config, _ = initialize_work_dir(tmp_path)
    image_path = tmp_path / "frames_dedup" / "ep01_f0000000001_t000001.000.png"
    _save_image(image_path, (10, 20, 30))
    config["crop_types"] = {
        "full": True,
        "hard_split": False,
        "face": False,
        "body": False,
        "background": False,
        "random_crop": False,
    }

    class FailingModel:
        def predict(self, *args, **kwargs):
            raise AssertionError("detection should not run")

    saved, rows = crop_one_image(
        tmp_path,
        config,
        image_path,
        tmp_path / "crops",
        random.Random(42),
        body_model=FailingModel(),
        face_model=FailingModel(),
    )

    assert saved == 1
    assert rows[0]["crop_type"] == "full"
