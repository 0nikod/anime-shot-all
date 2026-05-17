"""Image crop generation with a bbox-first semantic crop pipeline."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .config import resolve_work_path
from .files import collect_images, parse_episode_id, relative_to_or_absolute
from .logging_utils import write_csv
from .progress import ProgressCallback, format_progress

MIN_AREA = 1024 * 1024
MAX_AREA = 1536 * 1536

RATIOS = {
    "9:16": 9 / 16,
    "3:4": 3 / 4,
    "1:1": 1.0,
    "4:3": 4 / 3,
    "16:9": 16 / 9,
}

BASE_RATIO_WEIGHTS = {
    "9:16": 1.0,
    "3:4": 1.2,
    "1:1": 1.5,
    "4:3": 1.2,
    "16:9": 1.0,
}

OUTPUT_SIZE_PRESETS = {
    "1:1": [(1024, 1024), (1152, 1152), (1280, 1280), (1408, 1408), (1536, 1536)],
    "9:16": [(768, 1408), (864, 1536), (1024, 1792), (1152, 2048)],
    "16:9": [(1408, 768), (1536, 864), (1792, 1024), (2048, 1152)],
    "3:4": [(896, 1152), (1024, 1280), (1152, 1536), (1280, 1664)],
    "4:3": [(1152, 896), (1280, 1024), (1536, 1152), (1664, 1280)],
}

CROP_MODES = ("full", "face", "body", "halfbody", "random_crop")

CROP_LOG_FIELDS = [
    "source_image",
    "output_image",
    "episode_id",
    "crop_type",
    "producer_type",
    "x1",
    "y1",
    "x2",
    "y2",
    "raw_x1",
    "raw_y1",
    "raw_x2",
    "raw_y2",
    "semantic_x1",
    "semantic_y1",
    "semantic_x2",
    "semantic_y2",
    "source_width",
    "source_height",
    "output_width",
    "output_height",
    "output_area",
    "aspect_mode",
    "bbox_ratio",
    "selected_ratio",
    "padding_x",
    "padding_y",
    "model_path",
    "conf",
    "class_id",
    "score",
    "random_seed",
    "fallback_used",
    "fallback_reason",
    "reason",
    "status",
    "error",
]


@dataclass
class Detection:
    box: tuple[float, float, float, float]
    score: float
    label: str


@dataclass
class CropCandidate:
    crop_type: str
    producer_type: str
    box: tuple[float, float, float, float]
    reason: str
    model_path: str = ""
    conf: float | str = ""
    class_id: int | str = ""
    score: float | str = ""
    padding_x: float | str = ""
    padding_y: float | str = ""
    fallback_used: bool = False
    fallback_reason: str = ""


@dataclass
class PreparedCrop:
    raw_box: tuple[int, int, int, int]
    semantic_box: tuple[int, int, int, int]
    final_box: tuple[int, int, int, int]
    selected_ratio: str
    output_size: tuple[int, int]
    bbox_ratio: float
    reason: str


def run_crop(
    work_dir: Path,
    config: dict[str, Any],
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    stop_state: dict[str, Any] | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[int, Path]:
    params = config["crop"]
    input_dir = input_dir or resolve_work_path(work_dir, params.get("input_dir", "frames_raw"))
    output_dir = output_dir or resolve_work_path(work_dir, params.get("output_dir", "crops"))
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(params.get("random_seed", 42)))

    rows: list[dict[str, object]] = []
    saved = 0
    images = collect_images(input_dir)
    total = len(images)
    for index, image_path in enumerate(images, start=1):
        if stop_state and stop_state.get("stop"):
            if progress:
                progress("stopped by user")
            break
        if progress:
            progress(format_progress("crop", index, total, image_path))
        image_saved, image_rows = crop_one_image(work_dir, config, image_path, output_dir, rng)
        saved += image_saved
        rows.extend(image_rows)
        if progress:
            progress(f"{image_path.name}: saved {image_saved} crops, total {saved}")
    log_path = resolve_work_path(work_dir, config["logging"]["crop_log"])
    write_csv(log_path, CROP_LOG_FIELDS, rows)
    return saved, log_path


def crop_one_image(
    work_dir: Path,
    config: dict[str, Any],
    image_path: Path,
    output_dir: Path,
    rng: random.Random,
) -> tuple[int, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    params = config["crop"]
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    width, height = image.size
    detections = _collect_detections(image, image_path, config)
    candidates = _build_candidates(image, config, rng, detections)
    selected = _select_candidates(candidates, config, rng)

    saved = 0
    for index, candidate in enumerate(selected, start=1):
        try:
            prepared = _prepare_crop(candidate, image.size, config, rng)
        except ValueError as exc:
            rows.append(_crop_log_row(work_dir, image_path, "", candidate, image.size, None, "skipped", str(exc)))
            continue

        crop = image.crop(prepared.final_box)
        crop = crop.resize(prepared.output_size)
        output_area = crop.width * crop.height
        if not _area_is_valid(output_area):
            rows.append(_crop_log_row(work_dir, image_path, "", candidate, image.size, prepared, "skipped", "invalid_output_area"))
            continue

        target_dir = _target_dir(output_dir, candidate.crop_type)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{image_path.stem}_{candidate.crop_type}_{index:02d}.png"
        crop.save(target, "PNG", compress_level=int(params.get("png_compression", 3)))
        saved += 1
        rows.append(_crop_log_row(work_dir, image_path, target, candidate, image.size, prepared, "saved", prepared.reason))

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


def _collect_detections(image: Image.Image, image_path: Path, config: dict[str, Any]) -> dict[str, list[Detection]]:
    body_detections = _detect(image_path, config, "body") if _needs_body_detection(config) else []
    face_detections = _detect(image_path, config, "face") if _needs_face_detection(config) else []
    halfbody_detections = _detect_halfbody(image, image_path, config, body_detections) if _needs_halfbody_detection(config) else []
    return {"body": body_detections, "face": face_detections, "halfbody": halfbody_detections}


def _build_candidates(
    image: Image.Image,
    config: dict[str, Any],
    rng: random.Random,
    detections: dict[str, list[Detection]],
) -> list[CropCandidate]:
    enabled = {key: bool(value) for key, value in config.get("crop_types", {}).items() if key in CROP_MODES}
    candidates: list[CropCandidate] = []
    for crop_type in CROP_MODES:
        if not enabled.get(crop_type, True):
            continue
        produced = _produce_candidates(crop_type, crop_type, image, config, rng, detections, False, "")
        if produced:
            candidates.extend(produced)
    return candidates


def _produce_candidates(
    crop_type: str,
    producer_type: str,
    image: Image.Image,
    config: dict[str, Any],
    rng: random.Random,
    detections: dict[str, list[Detection]],
    fallback_used: bool,
    fallback_reason: str,
) -> list[CropCandidate]:
    width, height = image.size
    if producer_type == "full":
        return [
            CropCandidate(
                crop_type,
                "full",
                (0, 0, width, height),
                "full_copy" if not fallback_used else "fallback_full",
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
            )
        ]
    if producer_type in {"face", "body", "halfbody"}:
        return _detection_candidates(crop_type, producer_type, detections.get(producer_type, []), config, fallback_used, fallback_reason)
    if producer_type == "random_crop":
        return _random_candidates(crop_type, width, height, config, rng, detections.get("body", []), fallback_used, fallback_reason)
    return []


def _detection_candidates(
    crop_type: str,
    producer_type: str,
    detections: list[Detection],
    config: dict[str, Any],
    fallback_used: bool,
    fallback_reason: str,
) -> list[CropCandidate]:
    params = _semantic_params(config, producer_type)
    max_count = int(params["max_count_per_image"])
    min_size = int(params["min_size"])
    model_kind = "person" if producer_type == "body" else producer_type
    candidates = []
    for detection in sorted(detections, key=lambda item: item.score, reverse=True)[:max_count]:
        if not _valid_float_box(detection.box, min_size):
            continue
        candidates.append(
            CropCandidate(
                crop_type,
                producer_type,
                detection.box,
                "detected" if not fallback_used else f"fallback_{producer_type}",
                model_path=f"imgutils:{model_kind}",
                conf=config["detection"].get("conf_threshold", ""),
                class_id=detection.label,
                score=round(detection.score, 4),
                padding_x=params["left"],
                padding_y=params["top"],
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
            )
        )
    return candidates


def _random_candidates(
    crop_type: str,
    width: int,
    height: int,
    config: dict[str, Any],
    rng: random.Random,
    body_detections: list[Detection],
    fallback_used: bool,
    fallback_reason: str,
) -> list[CropCandidate]:
    params = config["random_crop"]
    candidates = []
    attempts = max(20, int(params["count_per_image"]) * 10)
    forbidden = [item.box for item in body_detections] if params.get("avoid_body", False) else []
    while len(candidates) < int(params["count_per_image"]) and attempts > 0:
        attempts -= 1
        scale = rng.uniform(float(params["min_scale"]), float(params["max_scale"]))
        crop_w = max(1, int(width * scale * rng.uniform(0.75, 1.0)))
        crop_h = max(1, int(height * scale * rng.uniform(0.75, 1.0)))
        if crop_w > width or crop_h > height:
            continue
        x1 = rng.randint(0, width - crop_w)
        y1 = rng.randint(0, height - crop_h)
        box = (x1, y1, x1 + crop_w, y1 + crop_h)
        if forbidden and max(_overlap_ratio(box, item) for item in forbidden) > 0.05:
            continue
        candidates.append(
            CropCandidate(
                crop_type,
                "random_crop",
                box,
                "random" if not fallback_used else "fallback_random_crop",
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
            )
        )
    return candidates


def _prepare_crop(
    candidate: CropCandidate,
    image_size: tuple[int, int],
    config: dict[str, Any],
    rng: random.Random,
) -> PreparedCrop:
    image_width, image_height = image_size
    raw = _clamp_float_box(candidate.box, image_width, image_height)
    min_size = int(config["crop"].get("min_crop_size", 32))
    if not _valid_float_box(raw, min_size):
        raise ValueError("invalid_bbox")

    if candidate.producer_type == "full":
        output_size = _resize_full_by_area(image_width, image_height, config)
        return PreparedCrop(
            _round_box(raw),
            _round_box(raw),
            _round_box(raw),
            "original",
            output_size,
            image_width / image_height,
            "full_copy" if not candidate.fallback_used else "fallback_full",
        )

    semantic = _expand_bbox(raw, _semantic_params(config, candidate.producer_type))
    semantic = _clamp_float_box(semantic, image_width, image_height)
    if not _valid_float_box(semantic, min_size):
        raise ValueError("too_small")
    bbox_width, bbox_height = _box_width_height(semantic)
    bbox_ratio = bbox_width / bbox_height
    ratio_names = _rank_ratios_by_bbox(bbox_ratio, config)
    for ratio_name in _weighted_ratio_order(ratio_names, bbox_ratio, config, rng):
        target_ratio = RATIOS[ratio_name]
        fitted = _fit_bbox_to_ratio(semantic, target_ratio)
        adjusted = _shift_or_shrink_to_image(fitted, semantic, image_width, image_height, target_ratio)
        if adjusted is None:
            continue
        output_size = _choose_output_size(ratio_name, rng)
        return PreparedCrop(
            _round_box(raw),
            _round_box(semantic),
            _round_box(adjusted),
            ratio_name,
            output_size,
            bbox_ratio,
            candidate.reason,
        )
    raise ValueError("fit_failed")


def _select_candidates(candidates: list[CropCandidate], config: dict[str, Any], rng: random.Random) -> list[CropCandidate]:
    strategy = config["crop"].get("output_strategy", "fixed")
    if strategy == "fixed":
        return candidates
    target = int(config["crop"].get("target_crops_per_image", 3))
    weights = config.get("random_output_weights", {})
    pool = list(candidates)
    selected = []
    while pool and len(selected) < target:
        weighted = [max(0, int(weights.get(item.crop_type, 0))) for item in pool]
        if sum(weighted) <= 0:
            break
        choice = rng.choices(pool, weights=weighted, k=1)[0]
        selected.append(choice)
        pool.remove(choice)
    return selected


def _detect(image_path: Path, config: dict[str, Any], kind: str) -> list[Detection]:
    if kind == "body":
        return _imgutils_detect(image_path, config, "person")
    return _imgutils_detect(image_path, config, kind)


def _detect_halfbody(image: Image.Image, image_path: Path, config: dict[str, Any], body_detections: list[Detection]) -> list[Detection]:
    # imgutils halfbody is most reliable on single-person crops; when body
    # boxes exist, run halfbody inside each body and translate back.
    if not body_detections:
        return _imgutils_detect(image_path, config, "halfbody")

    width, height = image.size
    detections: list[Detection] = []
    for person in body_detections:
        x1, y1, x2, y2 = _clamp_box(_round_box(person.box), width, height)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = image.crop((x1, y1, x2, y2))
        for detection in _imgutils_detect(crop, config, "halfbody"):
            hx1, hy1, hx2, hy2 = detection.box
            detections.append(Detection((hx1 + x1, hy1 + y1, hx2 + x1, hy2 + y1), detection.score, detection.label))
    return detections


def _imgutils_detect(image: Any, config: dict[str, Any], kind: str) -> list[Detection]:
    params = config["detection"]
    level = str(params[f"{kind}_level"])
    version = str(params[f"{kind}_version"])
    conf_threshold = float(params.get("conf_threshold", 0.35))
    iou_threshold = float(params.get("iou_threshold", 0.7))
    raw_results = _call_imgutils_detector(image, kind, level, version, conf_threshold, iou_threshold)
    return [Detection(tuple(float(value) for value in box), float(score), label) for box, label, score in raw_results]


def _call_imgutils_detector(
    image: Any,
    kind: str,
    level: str,
    version: str,
    conf_threshold: float,
    iou_threshold: float,
) -> list[tuple[tuple[int, int, int, int], str, float]]:
    if kind == "face":
        from imgutils.detect import detect_faces

        return detect_faces(image, level=level, version=version, conf_threshold=conf_threshold, iou_threshold=iou_threshold)
    if kind == "person":
        from imgutils.detect import detect_person

        return detect_person(image, level=level, version=version, conf_threshold=conf_threshold, iou_threshold=iou_threshold)
    if kind == "halfbody":
        from imgutils.detect import detect_halfbody

        return detect_halfbody(image, level=level, version=version, conf_threshold=conf_threshold, iou_threshold=iou_threshold)
    raise ValueError(f"unsupported detection kind: {kind}")


def _needs_body_detection(config: dict[str, Any]) -> bool:
    enabled = config.get("crop_types", {})
    return bool(
        enabled.get("body", True)
        or enabled.get("halfbody", True)
        or enabled.get("face", True)
        or (enabled.get("random_crop", True) and config.get("random_crop", {}).get("avoid_body", False))
    )


def _needs_face_detection(config: dict[str, Any]) -> bool:
    enabled = config.get("crop_types", {})
    return bool(enabled.get("face", True) or enabled.get("halfbody", True) or enabled.get("body", True))


def _needs_halfbody_detection(config: dict[str, Any]) -> bool:
    enabled = config.get("crop_types", {})
    return bool(enabled.get("halfbody", True) or enabled.get("face", True) or enabled.get("body", True))


def _target_dir(output_dir: Path, crop_type: str) -> Path:
    return output_dir / crop_type


def _semantic_params(config: dict[str, Any], kind: str) -> dict[str, float | int]:
    if kind == "face":
        crop_config = config.get("face_crop", {})
        return {
            "top": float(crop_config.get("expand_top", 1.5)),
            "bottom": float(crop_config.get("expand_bottom", 2.0)),
            "left": float(crop_config.get("expand_left", 1.4)),
            "right": float(crop_config.get("expand_right", 1.4)),
            "min_size": int(crop_config.get("min_size", 128)),
            "max_count_per_image": int(crop_config.get("max_count_per_image", 5)),
        }
    if kind == "body":
        crop_config = config.get("body_crop", {})
        return {
            "top": float(crop_config.get("expand_top", 1.15)),
            "bottom": float(crop_config.get("expand_bottom", 1.15)),
            "left": float(crop_config.get("expand_left", 1.2)),
            "right": float(crop_config.get("expand_right", 1.2)),
            "min_size": int(crop_config.get("min_size", 256)),
            "max_count_per_image": int(crop_config.get("max_count_per_image", 3)),
        }
    if kind == "halfbody":
        crop_config = config.get("halfbody_crop", {})
        return {
            "top": float(crop_config.get("expand_top", 1.2)),
            "bottom": float(crop_config.get("expand_bottom", 1.25)),
            "left": float(crop_config.get("expand_left", 1.2)),
            "right": float(crop_config.get("expand_right", 1.2)),
            "min_size": int(crop_config.get("min_size", 192)),
            "max_count_per_image": int(crop_config.get("max_count_per_image", 3)),
        }
    return {
        "top": 1.0,
        "bottom": 1.0,
        "left": 1.0,
        "right": 1.0,
        "min_size": int(config["crop"].get("min_crop_size", 32)),
        "max_count_per_image": 1,
    }


def _expand_bbox(box: tuple[float, float, float, float], params: dict[str, float | int]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    left = float(params["left"])
    right = float(params["right"])
    top = float(params["top"])
    bottom = float(params["bottom"])
    return (x1 - width * (left - 1.0), y1 - height * (top - 1.0), x2 + width * (right - 1.0), y2 + height * (bottom - 1.0))


def _rank_ratios_by_bbox(bbox_ratio: float, config: dict[str, Any]) -> list[str]:
    ratio_config = config.get("ratio_selection", {})
    max_change = float(ratio_config.get("max_ratio_change", 2.2))
    allow_square = bool(ratio_config.get("always_allow_square", True))
    ranked = []
    for name, target_ratio in RATIOS.items():
        change = max(target_ratio / bbox_ratio, bbox_ratio / target_ratio)
        if name != "1:1" and change > max_change:
            continue
        if name == "1:1" and not allow_square and change > max_change:
            continue
        ranked.append(name)
    return ranked or ["1:1"]


def _weighted_ratio_order(
    names: list[str],
    bbox_ratio: float,
    config: dict[str, Any],
    rng: random.Random,
) -> list[str]:
    remaining = list(names)
    ordered = []
    while remaining:
        weights = [_ratio_weight(name, bbox_ratio, config) for name in remaining]
        choice = rng.choices(remaining, weights=weights, k=1)[0]
        ordered.append(choice)
        remaining.remove(choice)
    nearest = sorted(names, key=lambda name: abs(math.log(RATIOS[name] / bbox_ratio)))
    return ordered + [name for name in nearest if name not in ordered]


def _ratio_weight(name: str, bbox_ratio: float, config: dict[str, Any]) -> float:
    ratio_config = config.get("ratio_selection", {})
    base_weights = ratio_config.get("base_weights", BASE_RATIO_WEIGHTS)
    sigma = float(ratio_config.get("sigma", 0.45))
    target_ratio = RATIOS[name]
    distance = abs(math.log(target_ratio / bbox_ratio))
    gaussian = math.exp(-((distance**2) / (2 * sigma**2)))
    return max(float(base_weights.get(name, BASE_RATIO_WEIGHTS[name])) * gaussian, 0.01)


def _fit_bbox_to_ratio(box: tuple[float, float, float, float], target_ratio: float) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    current_ratio = width / height
    if current_ratio < target_ratio:
        new_width = height * target_ratio
        new_height = height
    else:
        new_width = width
        new_height = width / target_ratio
    return (cx - new_width / 2, cy - new_height / 2, cx + new_width / 2, cy + new_height / 2)


def _shift_or_shrink_to_image(
    box: tuple[float, float, float, float],
    semantic: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
    target_ratio: float,
) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    if width <= image_width and height <= image_height:
        x1 = min(max(0, x1), image_width - width)
        y1 = min(max(0, y1), image_height - height)
        shifted = (x1, y1, x1 + width, y1 + height)
        return shifted if _contains(shifted, semantic) else None

    candidate_width = min(image_width, image_height * target_ratio)
    candidate_height = candidate_width / target_ratio
    if candidate_height > image_height:
        candidate_height = image_height
        candidate_width = candidate_height * target_ratio

    sx1, sy1, sx2, sy2 = semantic
    semantic_width = sx2 - sx1
    semantic_height = sy2 - sy1
    if candidate_width + 1e-6 < semantic_width or candidate_height + 1e-6 < semantic_height:
        return None

    cx = (sx1 + sx2) / 2
    cy = (sy1 + sy2) / 2
    x1 = min(max(0, cx - candidate_width / 2), image_width - candidate_width)
    y1 = min(max(0, cy - candidate_height / 2), image_height - candidate_height)
    adjusted = (x1, y1, x1 + candidate_width, y1 + candidate_height)
    return adjusted if _contains(adjusted, semantic) else None


def _choose_output_size(ratio_name: str, rng: random.Random) -> tuple[int, int]:
    candidates = [(w, h) for w, h in OUTPUT_SIZE_PRESETS[ratio_name] if _area_is_valid(w * h)]
    if not candidates:
        raise ValueError(f"no_valid_output_size:{ratio_name}")
    return rng.choice(candidates)


def _resize_full_by_area(width: int, height: int, config: dict[str, Any]) -> tuple[int, int]:
    full_config = config.get("full_crop", {})
    min_area = int(full_config.get("min_area", MIN_AREA))
    max_area = int(full_config.get("max_area", MAX_AREA))
    max_upscale = float(full_config.get("max_upscale", 2.0))
    area = width * height
    if area > max_area:
        scale = math.sqrt(max_area / area)
    elif area < min_area:
        scale = min(max_upscale, math.sqrt(min_area / area))
    else:
        scale = 1.0
    output_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    if not (min_area <= output_size[0] * output_size[1] <= max_area):
        raise ValueError("full_output_area_out_of_range")
    return output_size


def _clamp_float_box(box: tuple[float, float, float, float], width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return (max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2))


def _round_box(box: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    return tuple(round(value) for value in box)  # type: ignore[return-value]


def _clamp_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (max(0, x1), max(0, y1), min(width, x2), min(height, y2))


def _valid_float_box(box: tuple[float, float, float, float], min_size: int) -> bool:
    if any(math.isnan(value) for value in box):
        return False
    width, height = _box_width_height(box)
    return width >= min_size and height >= min_size


def _box_width_height(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return box[2] - box[0], box[3] - box[1]


def _contains(outer: tuple[float, float, float, float], inner: tuple[float, float, float, float]) -> bool:
    return outer[0] <= inner[0] + 1e-6 and outer[1] <= inner[1] + 1e-6 and outer[2] >= inner[2] - 1e-6 and outer[3] >= inner[3] - 1e-6


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


def _area_is_valid(area: int) -> bool:
    return MIN_AREA <= area <= MAX_AREA


def _crop_log_row(
    work_dir: Path,
    source: Path,
    target: Path | str,
    candidate: CropCandidate,
    source_size: tuple[int, int],
    prepared: PreparedCrop | None,
    status: str,
    reason: str,
) -> dict[str, object]:
    raw = prepared.raw_box if prepared else _round_box(_clamp_float_box(candidate.box, source_size[0], source_size[1]))
    semantic = prepared.semantic_box if prepared else (0, 0, 0, 0)
    final = prepared.final_box if prepared else raw
    output_size = prepared.output_size if prepared else (0, 0)
    selected_ratio = prepared.selected_ratio if prepared else ""
    bbox_ratio = round(prepared.bbox_ratio, 4) if prepared else ""
    output_area = output_size[0] * output_size[1]
    return {
        "source_image": relative_to_or_absolute(source, work_dir),
        "output_image": relative_to_or_absolute(target, work_dir) if target else "",
        "episode_id": parse_episode_id(source),
        "crop_type": candidate.crop_type,
        "producer_type": candidate.producer_type,
        "x1": final[0],
        "y1": final[1],
        "x2": final[2],
        "y2": final[3],
        "raw_x1": raw[0],
        "raw_y1": raw[1],
        "raw_x2": raw[2],
        "raw_y2": raw[3],
        "semantic_x1": semantic[0],
        "semantic_y1": semantic[1],
        "semantic_x2": semantic[2],
        "semantic_y2": semantic[3],
        "source_width": source_size[0],
        "source_height": source_size[1],
        "output_width": output_size[0],
        "output_height": output_size[1],
        "output_area": output_area,
        "aspect_mode": selected_ratio,
        "bbox_ratio": bbox_ratio,
        "selected_ratio": selected_ratio,
        "padding_x": candidate.padding_x,
        "padding_y": candidate.padding_y,
        "model_path": candidate.model_path,
        "conf": candidate.conf,
        "class_id": candidate.class_id,
        "score": candidate.score,
        "random_seed": "",
        "fallback_used": candidate.fallback_used,
        "fallback_reason": candidate.fallback_reason,
        "reason": reason,
        "status": status,
        "error": "" if status != "skipped" else reason,
    }
