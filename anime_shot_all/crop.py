"""Image crop generation, including hard splits, random crops, and YOLO crops."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .config import resolve_work_path
from .files import collect_images, parse_episode_id, relative_to_or_absolute
from .logging_utils import write_csv
from .yolo_models import resolve_yolo_model_path


CROP_LOG_FIELDS = [
    "source_image",
    "output_image",
    "episode_id",
    "crop_type",
    "x1",
    "y1",
    "x2",
    "y2",
    "source_width",
    "source_height",
    "output_width",
    "output_height",
    "aspect_mode",
    "padding_x",
    "padding_y",
    "model_path",
    "conf",
    "class_id",
    "score",
    "random_seed",
    "reason",
    "status",
    "error",
]

ASPECTS = {
    "square": 1.0,
    "portrait_2_3": 2 / 3,
    "portrait_3_4": 3 / 4,
    "portrait_9_16": 9 / 16,
    "landscape_16_9": 16 / 9,
    "landscape_4_3": 4 / 3,
    "1:1": 1.0,
    "2:3": 2 / 3,
    "3:4": 3 / 4,
    "9:16": 9 / 16,
    "16:9": 16 / 9,
    "4:3": 4 / 3,
}


@dataclass
class Detection:
    box: tuple[float, float, float, float]
    score: float
    class_id: int


@dataclass
class CropCandidate:
    crop_type: str
    box: tuple[int, int, int, int]
    aspect_mode: str
    reason: str
    model_path: str = ""
    conf: float | str = ""
    class_id: int | str = ""
    score: float | str = ""
    padding_x: float | str = ""
    padding_y: float | str = ""


def run_crop(work_dir: Path, config: dict[str, Any], input_dir: Path | None = None, output_dir: Path | None = None) -> tuple[int, Path]:
    params = config["crop"]
    input_dir = input_dir or resolve_work_path(work_dir, params.get("input_dir", "frames_dedup"))
    output_dir = output_dir or resolve_work_path(work_dir, params.get("output_dir", "crops"))
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(params.get("random_seed", 42)))

    body_model_path = resolve_yolo_model_path(work_dir, config["yolo"], "body") if _needs_body_detection(config) else None
    face_model_path = resolve_yolo_model_path(work_dir, config["yolo"], "face") if _needs_face_detection(config) else None
    config["yolo"]["body_model_path"] = str(body_model_path or config["yolo"].get("body_model_path", ""))
    config["yolo"]["face_model_path"] = str(face_model_path or config["yolo"].get("face_model_path", ""))
    body_model = _load_model(body_model_path) if body_model_path else None
    face_model = _load_model(face_model_path) if face_model_path else None

    rows: list[dict[str, object]] = []
    saved = 0
    for image_path in collect_images(input_dir):
        image_saved, image_rows = crop_one_image(work_dir, config, image_path, output_dir, rng, body_model, face_model)
        saved += image_saved
        rows.extend(image_rows)
    log_path = resolve_work_path(work_dir, config["logging"]["crop_log"])
    write_csv(log_path, CROP_LOG_FIELDS, rows)
    return saved, log_path


def crop_one_image(
    work_dir: Path,
    config: dict[str, Any],
    image_path: Path,
    output_dir: Path,
    rng: random.Random,
    body_model: Any = None,
    face_model: Any = None,
) -> tuple[int, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    params = config["crop"]
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    width, height = image.size
    body_detections = _detect(body_model, image_path, config, "body") if _needs_body_detection(config) else []
    face_detections = _detect(face_model, image_path, config, "face") if _needs_face_detection(config) else []
    candidates = _build_candidates(image, config, rng, body_detections, face_detections)
    selected = _select_candidates(candidates, config, rng)
    saved = 0
    for index, candidate in enumerate(selected, start=1):
        box = _clamp_box(candidate.box, width, height)
        if not _valid_box(box, int(params["min_crop_size"])):
            rows.append(_crop_log_row(work_dir, image_path, "", candidate, image.size, box, "skipped", "too_small"))
            continue
        crop = image.crop(box)
        crop = _limit_max_side(crop, int(params.get("max_side", 2048)))
        target_dir = _target_dir(output_dir, candidate.crop_type)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{image_path.stem}_{candidate.crop_type}_{index:02d}.png"
        crop.save(target, "PNG", compress_level=int(params.get("png_compression", 3)))
        saved += 1
        rows.append(_crop_log_row(work_dir, image_path, target, candidate, image.size, box, "saved", candidate.reason, crop.size))
    if not selected:
        rows.append(
            {
                "source_image": relative_to_or_absolute(image_path, work_dir),
                "episode_id": parse_episode_id(image_path),
                "source_width": width,
                "source_height": height,
                "status": "skipped",
                "reason": "no_candidate",
            }
        )
    return saved, rows


def _build_candidates(
    image: Image.Image,
    config: dict[str, Any],
    rng: random.Random,
    body_detections: list[Detection],
    face_detections: list[Detection],
) -> list[CropCandidate]:
    enabled = config["crop_types"]
    width, height = image.size
    candidates: list[CropCandidate] = []
    if enabled.get("full", True):
        candidates.append(CropCandidate("full", (0, 0, width, height), "original", "full_copy"))
    if enabled.get("hard_split", True):
        candidates.extend(_hard_split_candidates(width, height, config["hard_split"]))
    if enabled.get("body", True):
        candidates.extend(_detection_candidates("body", body_detections, width, height, config))
    if enabled.get("face", True):
        candidates.extend(_detection_candidates("face", face_detections, width, height, config))
    if enabled.get("background", True):
        candidates.extend(_background_candidates(width, height, body_detections, config))
    if enabled.get("random_crop", True):
        candidates.extend(_random_candidates(width, height, config, rng, body_detections))
    return candidates


def _hard_split_candidates(width: int, height: int, enabled: dict[str, bool]) -> list[CropCandidate]:
    side = min(width, height)
    portrait_width = min(width, int(height * 2 / 3))
    landscape_height = min(height, int(width * 9 / 16))
    specs = {
        "left_square": (0, 0, side, side),
        "center_square": ((width - side) // 2, (height - side) // 2, (width + side) // 2, (height + side) // 2),
        "right_square": (width - side, 0, width, side),
        "center_portrait": ((width - portrait_width) // 2, 0, (width + portrait_width) // 2, height),
        "upper_landscape": (0, 0, width, landscape_height),
        "lower_landscape": (0, height - landscape_height, width, height),
    }
    return [
        CropCandidate(f"hard_{name}", box, "fixed", "hard_split")
        for name, box in specs.items()
        if enabled.get(name, True)
    ]


def _detection_candidates(kind: str, detections: list[Detection], width: int, height: int, config: dict[str, Any]) -> list[CropCandidate]:
    if kind == "body":
        params = config["body_crop"]
        padding_x = float(params["padding_x"])
        padding_y = float(params["padding_y"])
        aspect_mode = params["aspect_mode"]
        max_count = int(params["max_count_per_image"])
        min_size = int(params["min_size"])
        model_path = config["yolo"].get("body_model_path", "")
    else:
        params = config["face_crop"]
        padding_x = padding_y = float(params["padding"])
        aspect_mode = params["aspect_mode"]
        max_count = int(params["max_count_per_image"])
        min_size = int(params["min_size"])
        model_path = config["yolo"].get("face_model_path", "")
    candidates = []
    for detection in sorted(detections, key=lambda item: item.score, reverse=True)[:max_count]:
        box = _pad_box(detection.box, padding_x, padding_y)
        box = _fit_aspect(box, aspect_mode, width, height)
        if _box_width_height(box)[0] < min_size or _box_width_height(box)[1] < min_size:
            continue
        candidates.append(
            CropCandidate(
                kind,
                _round_box(box),
                aspect_mode,
                "detected",
                model_path=model_path,
                conf=config["yolo"].get("conf", ""),
                class_id=detection.class_id,
                score=round(detection.score, 4),
                padding_x=padding_x,
                padding_y=padding_y,
            )
        )
    return candidates


def _background_candidates(width: int, height: int, detections: list[Detection], config: dict[str, Any]) -> list[CropCandidate]:
    params = config["background_crop"]
    forbidden = [_pad_box(item.box, float(params["exclusion_padding"]), float(params["exclusion_padding"])) for item in detections]
    if not forbidden and not params.get("allow_no_body", True):
        return []
    base = _hard_split_candidates(
        width,
        height,
        {
            "left_square": False,
            "center_square": True,
            "right_square": False,
            "center_portrait": False,
            "upper_landscape": True,
            "lower_landscape": True,
        },
    )
    accepted = []
    for candidate in base:
        overlap = max([_overlap_ratio(candidate.box, box) for box in forbidden] or [0.0])
        if overlap <= float(params["max_overlap"]):
            accepted.append(CropCandidate("background", candidate.box, params["aspect_mode"], "detected" if forbidden else "no_detection"))
    return accepted[: int(params["max_count_per_image"])]


def _random_candidates(
    width: int,
    height: int,
    config: dict[str, Any],
    rng: random.Random,
    body_detections: list[Detection],
) -> list[CropCandidate]:
    params = config["random_crop"]
    candidates = []
    attempts = max(20, int(params["count_per_image"]) * 10)
    forbidden = [item.box for item in body_detections] if params.get("avoid_body", False) else []
    while len(candidates) < int(params["count_per_image"]) and attempts > 0:
        attempts -= 1
        aspect_label = rng.choice(list(params.get("aspect_pool") or ["1:1"]))
        aspect = ASPECTS.get(aspect_label, 1.0)
        scale = rng.uniform(float(params["min_scale"]), float(params["max_scale"]))
        crop_w = int(width * scale)
        crop_h = int(crop_w / aspect)
        if crop_h > height:
            crop_h = int(height * scale)
            crop_w = int(crop_h * aspect)
        if crop_w <= 0 or crop_h <= 0 or crop_w > width or crop_h > height:
            continue
        x1 = rng.randint(0, width - crop_w)
        y1 = rng.randint(0, height - crop_h)
        box = (x1, y1, x1 + crop_w, y1 + crop_h)
        if forbidden and max(_overlap_ratio(box, item) for item in forbidden) > 0.05:
            continue
        candidates.append(CropCandidate("random_crop", box, aspect_label, "random"))
    return candidates


def _select_candidates(candidates: list[CropCandidate], config: dict[str, Any], rng: random.Random) -> list[CropCandidate]:
    strategy = config["crop"].get("output_strategy", "fixed")
    if strategy == "fixed":
        return candidates
    target = int(config["crop"].get("target_crops_per_image", 3))
    weights = config.get("random_output_weights", {})
    pool = list(candidates)
    selected = []
    while pool and len(selected) < target:
        weighted = [max(0, int(weights.get(_weight_key(item.crop_type), 0))) for item in pool]
        if sum(weighted) <= 0:
            break
        choice = rng.choices(pool, weights=weighted, k=1)[0]
        selected.append(choice)
        pool.remove(choice)
    return selected


def _detect(model: Any, image_path: Path, config: dict[str, Any], kind: str) -> list[Detection]:
    if model is None:
        return []
    yolo = config["yolo"]
    results = model.predict(str(image_path), conf=float(yolo["conf"]), imgsz=int(yolo["imgsz"]), verbose=False)
    detections: list[Detection] = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            class_id = int(box.cls.item())
            if kind == "body" and class_id != int(yolo["body_class_id"]):
                continue
            if kind == "face" and not yolo.get("face_all_classes", True) and class_id != int(yolo.get("face_class_id", 0)):
                continue
            coords = tuple(float(value) for value in box.xyxy[0].tolist())
            detections.append(Detection(coords, float(box.conf.item()), class_id))
    return detections


def _load_model(model_path: str | Path) -> Any:
    if not model_path:
        return None
    from ultralytics import YOLO

    return YOLO(str(model_path))


def _needs_body_detection(config: dict[str, Any]) -> bool:
    enabled = config.get("crop_types", {})
    return bool(
        enabled.get("body", True)
        or enabled.get("background", True)
        or (enabled.get("random_crop", True) and config.get("random_crop", {}).get("avoid_body", False))
    )


def _needs_face_detection(config: dict[str, Any]) -> bool:
    return bool(config.get("crop_types", {}).get("face", True))


def _target_dir(output_dir: Path, crop_type: str) -> Path:
    if crop_type.startswith("hard_"):
        return output_dir / "hard" / crop_type.removeprefix("hard_")
    return output_dir / crop_type


def _weight_key(crop_type: str) -> str:
    return "hard_split" if crop_type.startswith("hard_") else crop_type


def _pad_box(box: tuple[float, float, float, float], padding_x: float, padding_y: float) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    return (x1 - width * padding_x, y1 - height * padding_y, x2 + width * padding_x, y2 + height * padding_y)


def _fit_aspect(box: tuple[float, float, float, float], aspect_mode: str, image_width: int, image_height: int) -> tuple[float, float, float, float]:
    if aspect_mode == "original":
        return box
    target = ASPECTS.get(aspect_mode)
    if not target:
        return box
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    current = width / height if height else target
    if current < target:
        new_width = height * target
        delta = (new_width - width) / 2
        x1 -= delta
        x2 += delta
    else:
        new_height = width / target
        delta = (new_height - height) / 2
        y1 -= delta
        y2 += delta
    return _clamp_float_box((x1, y1, x2, y2), image_width, image_height)


def _clamp_float_box(box: tuple[float, float, float, float], width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    box_width, box_height = x2 - x1, y2 - y1
    x1 = min(max(0, x1), max(0, width - box_width))
    y1 = min(max(0, y1), max(0, height - box_height))
    x2 = min(width, x1 + box_width)
    y2 = min(height, y1 + box_height)
    return (x1, y1, x2, y2)


def _round_box(box: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    return tuple(round(value) for value in box)  # type: ignore[return-value]


def _clamp_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (max(0, x1), max(0, y1), min(width, x2), min(height, y2))


def _valid_box(box: tuple[int, int, int, int], min_size: int) -> bool:
    return box[2] > box[0] and box[3] > box[1] and (box[2] - box[0]) >= min_size and (box[3] - box[1]) >= min_size


def _box_width_height(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return box[2] - box[0], box[3] - box[1]


def _overlap_ratio(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    area = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    return intersection / area


def _limit_max_side(image: Image.Image, max_side: int) -> Image.Image:
    if max(image.size) <= max_side:
        return image
    scale = max_side / max(image.size)
    return image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))))


def _crop_log_row(
    work_dir: Path,
    source: Path,
    target: Path | str,
    candidate: CropCandidate,
    source_size: tuple[int, int],
    box: tuple[int, int, int, int],
    status: str,
    reason: str,
    output_size: tuple[int, int] | None = None,
) -> dict[str, object]:
    output_size = output_size or (0, 0)
    return {
        "source_image": relative_to_or_absolute(source, work_dir),
        "output_image": relative_to_or_absolute(target, work_dir) if target else "",
        "episode_id": parse_episode_id(source),
        "crop_type": candidate.crop_type,
        "x1": box[0],
        "y1": box[1],
        "x2": box[2],
        "y2": box[3],
        "source_width": source_size[0],
        "source_height": source_size[1],
        "output_width": output_size[0],
        "output_height": output_size[1],
        "aspect_mode": candidate.aspect_mode,
        "padding_x": candidate.padding_x,
        "padding_y": candidate.padding_y,
        "model_path": candidate.model_path,
        "conf": candidate.conf,
        "class_id": candidate.class_id,
        "score": candidate.score,
        "random_seed": "",
        "reason": reason,
        "status": status,
        "error": "",
    }
