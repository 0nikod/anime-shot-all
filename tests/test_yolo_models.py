from pathlib import Path

from anime_shot_all.yolo_models import (
    YOLO_MODEL_PRESETS,
    download_selected_presets,
    preset_choices,
    resolve_yolo_model_path,
)


def test_preset_choices_include_bingsu_face_and_person_models():
    face_values = {value for _, value in preset_choices("face")}
    body_values = {value for _, value in preset_choices("body")}

    assert "bingsu/adetailer/face_yolov8n.pt" in face_values
    assert "bingsu/adetailer/face_yolov8m.pt" in face_values
    assert "bingsu/adetailer/person_yolov8n-seg.pt" in body_values
    assert "bingsu/adetailer/person_yolov8m-seg.pt" in body_values


def test_resolve_yolo_model_path_prefers_custom_path(tmp_path: Path):
    custom = tmp_path / "custom.pt"
    custom.write_bytes(b"model")
    config = {
        "body_model_path": str(custom),
        "body_model_preset": "bingsu/adetailer/person_yolov8n-seg.pt",
        "auto_download": True,
        "model_dir": "models/yolo",
    }

    assert resolve_yolo_model_path(tmp_path, config, "body") == custom


def test_download_selected_presets_uses_model_dir(tmp_path: Path, monkeypatch):
    def fake_urlretrieve(url: str, target: Path):
        target.write_bytes(url.encode("utf-8"))
        return str(target), None

    monkeypatch.setattr("anime_shot_all.yolo_models.urlretrieve", fake_urlretrieve)
    config = {
        "body_model_path": "",
        "face_model_path": "",
        "body_model_preset": "bingsu/adetailer/person_yolov8n-seg.pt",
        "face_model_preset": "bingsu/adetailer/face_yolov8n.pt",
        "auto_download": True,
        "model_dir": "models/yolo",
    }

    downloaded = download_selected_presets(tmp_path, config)

    assert downloaded["body"].endswith("models/yolo/person_yolov8n-seg.pt")
    assert downloaded["face"].endswith("models/yolo/face_yolov8n.pt")
    assert (tmp_path / "models" / "yolo" / "person_yolov8n-seg.pt").exists()
    assert YOLO_MODEL_PRESETS["bingsu/adetailer/face_yolov8n.pt"].url.endswith("/face_yolov8n.pt")
