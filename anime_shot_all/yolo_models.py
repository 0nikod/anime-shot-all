"""YOLO model presets and lightweight download helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

from .config import resolve_work_path


BINGSU_ADETAILER_REPO = "Bingsu/adetailer"


@dataclass(frozen=True)
class YoloModelPreset:
    key: str
    label: str
    filename: str
    kind: str
    repo: str = BINGSU_ADETAILER_REPO

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/main/{self.filename}"


YOLO_MODEL_PRESETS: dict[str, YoloModelPreset] = {
    "none": YoloModelPreset("none", "不使用预设", "", "none", ""),
    "bingsu/adetailer/face_yolov8n.pt": YoloModelPreset(
        "bingsu/adetailer/face_yolov8n.pt",
        "Bingsu/adetailer face_yolov8n.pt (6.23 MB)",
        "face_yolov8n.pt",
        "face",
    ),
    "bingsu/adetailer/face_yolov8n_v2.pt": YoloModelPreset(
        "bingsu/adetailer/face_yolov8n_v2.pt",
        "Bingsu/adetailer face_yolov8n_v2.pt (6.24 MB)",
        "face_yolov8n_v2.pt",
        "face",
    ),
    "bingsu/adetailer/face_yolov8s.pt": YoloModelPreset(
        "bingsu/adetailer/face_yolov8s.pt",
        "Bingsu/adetailer face_yolov8s.pt (22.5 MB)",
        "face_yolov8s.pt",
        "face",
    ),
    "bingsu/adetailer/face_yolov8m.pt": YoloModelPreset(
        "bingsu/adetailer/face_yolov8m.pt",
        "Bingsu/adetailer face_yolov8m.pt (52 MB)",
        "face_yolov8m.pt",
        "face",
    ),
    "bingsu/adetailer/face_yolov9c.pt": YoloModelPreset(
        "bingsu/adetailer/face_yolov9c.pt",
        "Bingsu/adetailer face_yolov9c.pt (51.6 MB)",
        "face_yolov9c.pt",
        "face",
    ),
    "bingsu/adetailer/person_yolov8n-seg.pt": YoloModelPreset(
        "bingsu/adetailer/person_yolov8n-seg.pt",
        "Bingsu/adetailer person_yolov8n-seg.pt (6.78 MB)",
        "person_yolov8n-seg.pt",
        "body",
    ),
    "bingsu/adetailer/person_yolov8s-seg.pt": YoloModelPreset(
        "bingsu/adetailer/person_yolov8s-seg.pt",
        "Bingsu/adetailer person_yolov8s-seg.pt",
        "person_yolov8s-seg.pt",
        "body",
    ),
    "bingsu/adetailer/person_yolov8m-seg.pt": YoloModelPreset(
        "bingsu/adetailer/person_yolov8m-seg.pt",
        "Bingsu/adetailer person_yolov8m-seg.pt (54.8 MB)",
        "person_yolov8m-seg.pt",
        "body",
    ),
}


def preset_choices(kind: str) -> list[tuple[str, str]]:
    return [
        (preset.label, preset.key)
        for preset in YOLO_MODEL_PRESETS.values()
        if preset.kind in {"none", kind}
    ]


def resolve_yolo_model_path(work_dir: Path, yolo_config: dict[str, object], kind: str) -> Path | None:
    """Resolve custom path first, otherwise download the selected preset if enabled."""

    custom_key = f"{kind}_model_path"
    preset_key = f"{kind}_model_preset"
    custom_path = str(yolo_config.get(custom_key) or "").strip()
    if custom_path:
        return resolve_work_path(work_dir, custom_path)
    if not bool(yolo_config.get("auto_download", True)):
        return None
    preset_name = str(yolo_config.get(preset_key) or "none")
    preset = YOLO_MODEL_PRESETS.get(preset_name)
    if preset is None or preset.key == "none":
        return None
    return download_yolo_preset(work_dir, yolo_config, preset)


def download_yolo_preset(work_dir: Path, yolo_config: dict[str, object], preset: YoloModelPreset) -> Path:
    model_dir = resolve_work_path(work_dir, str(yolo_config.get("model_dir") or "models/yolo"))
    model_dir.mkdir(parents=True, exist_ok=True)
    target = model_dir / preset.filename
    if target.exists() and target.stat().st_size > 0:
        return target
    temp_target = target.with_suffix(target.suffix + ".download")
    if temp_target.exists():
        temp_target.unlink()
    urlretrieve(preset.url, temp_target)
    temp_target.replace(target)
    return target


def download_selected_presets(work_dir: Path, yolo_config: dict[str, object]) -> dict[str, str]:
    downloaded: dict[str, str] = {}
    for kind in ("body", "face"):
        path = resolve_yolo_model_path(work_dir, yolo_config, kind)
        if path is not None:
            downloaded[kind] = str(path)
    return downloaded
