"""Gradio GUI wiring for the anime image dataset tool."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import gradio as gr

from .config import (
    deep_merge,
    initialize_work_dir,
    load_project_config,
    reset_params_from_default,
    resolve_work_path,
    save_params,
    write_yaml,
)
from .crop import CROP_LOG_FIELDS, crop_one_image
from .defaults import builtin_defaults
from .extract import EXTRACT_LOG_FIELDS, extract_frames_for_video
from .ignore_ranges import (
    export_csv,
    import_csv,
    load_ignore_state,
    normalize_ranges,
    save_ignore_state,
    state_to_rows,
    rows_to_state,
)
from .files import collect_images, relative_path_value
from .logging_utils import write_csv
from .progress import format_progress
from .stats import format_summary, recent_log_text, summarize_project
from .video import VideoInfo, probe_video, video_candidates, videos_as_dicts, videos_to_rows


IGNORE_HEADERS = ["episode_id", "video_name", "ignore_start", "ignore_end", "label", "enabled", "notes"]
VIDEO_HEADERS = ["episode_id", "video_path", "video_name", "duration_sec", "fps", "width", "height"]
PARAM_SCHEMA: dict[str, dict[str, Any]] = {
    "extract_output": {"component": "textbox", "label": "输出 frames_raw 文件夹", "info": "PNG 截帧保存目录（已包含去重输出）。"},
    "interval": {"component": "number", "label": "interval", "value": 0.25, "info": "采样间隔秒数；越小检查越频繁。"},
    "keyframe_only": {"component": "checkbox", "label": "截关键帧", "value": False, "info": "关键帧模式会忽略 interval / 去重分组。"},
    "png_compression": {"component": "slider", "label": "png_compression", "value": 3, "minimum": 0, "maximum": 9, "step": 1, "info": "PNG 压缩等级；越高越慢。"},
    "min_width": {"component": "number", "label": "min_width", "value": 0, "info": "输出最小宽度；0 表示不放大。"},
    "crop_bottom": {"component": "number", "label": "crop_bottom", "value": 0, "info": "从底部裁掉像素，可避开硬字幕。"},
    "phash_threshold": {"component": "number", "label": "phash_threshold", "value": 5, "info": "pHash 距离阈值；越大越激进。"},
    "phash_size": {"component": "number", "label": "phash_size", "value": 8, "info": "pHash 尺寸；越大越细。"},
    "phash_crop": {"component": "dropdown", "label": "phash_crop", "value": "center", "choices": ["center", "full"], "info": "center 更关注主体区域。"},
    "phash_resize_width": {"component": "number", "label": "phash_resize_width", "value": 256, "info": "hash 前缩放宽度；越小越快。"},
    "group_seconds_per_keep": {"component": "number", "label": "group_seconds_per_keep", "value": 5.0, "info": "相似组每多少秒保留一张。"},
    "group_max_duration": {"component": "number", "label": "group_max_duration", "value": 60.0, "info": "相似组强制切段时长上限（秒）。"},
    "extract_random_seed": {"component": "number", "label": "extract_random_seed", "value": 42, "info": "截帧分组随机抽取的随机种子。"},
    "crop_input": {"component": "textbox", "label": "输入图片文件夹", "info": "通常为 frames_raw。"},
    "crop_output": {"component": "textbox", "label": "输出 crops 文件夹", "info": "最终 PNG 裁剪输出目录。"},
    "crop_mode": {"component": "checkbox_group", "label": "输出类型", "choices": ["full", "face", "body", "halfbody", "random_crop"], "value": ["full", "face", "body", "halfbody", "random_crop"], "info": "未勾选的类型不会输出，也不会触发对应检测。"},
    "output_strategy": {"component": "dropdown", "label": "输出策略", "value": "fixed", "choices": ["fixed", "random_weighted"], "info": "fixed 全部尝试；random_weighted 按权重抽样。"},
    "weight_full": {"component": "number", "label": "full weight", "value": 0, "info": "随机策略下 full 权重。"},
    "weight_face": {"component": "number", "label": "face weight", "value": 30, "info": "随机策略下脸部权重。"},
    "weight_body": {"component": "number", "label": "body weight", "value": 30, "info": "随机策略下身体权重。"},
    "weight_halfbody": {"component": "number", "label": "halfbody weight", "value": 25, "info": "随机策略下上半身权重。"},
    "weight_random": {"component": "number", "label": "random_crop weight", "value": 20, "info": "随机策略下随机裁剪权重。"},
    "conf_threshold": {"component": "number", "label": "conf_threshold", "value": 0.35, "info": "imgutils 检测置信度阈值。"},
    "iou_threshold": {"component": "number", "label": "iou_threshold", "value": 0.7, "info": "imgutils 检测 NMS IoU 阈值。"},
    "face_level": {"component": "dropdown", "label": "face level", "value": "s", "choices": ["n", "s"], "info": "n 更快，s 更准。"},
    "face_version": {"component": "dropdown", "label": "face version", "value": "v1.4", "choices": ["v0", "v1", "v1.3", "v1.4"], "info": "imgutils face 检测模型版本。"},
    "person_level": {"component": "dropdown", "label": "person level", "value": "m", "choices": ["n", "s", "m", "x"], "info": "n 更快，x 更准。"},
    "person_version": {"component": "dropdown", "label": "person version", "value": "v1.1", "choices": ["v0", "v1", "v1.1"], "info": "imgutils person 检测模型版本。"},
    "halfbody_level": {"component": "dropdown", "label": "halfbody level", "value": "s", "choices": ["n", "s"], "info": "n 更快，s 更准。"},
    "halfbody_version": {"component": "dropdown", "label": "halfbody version", "value": "v1.0", "choices": ["v1.0"], "info": "imgutils halfbody 检测模型版本。"},
    "ratio_sigma": {"component": "number", "label": "ratio sigma", "value": 0.45, "info": "越小越贴近 bbox 原始比例。"},
    "max_ratio_change": {"component": "number", "label": "max ratio change", "value": 2.2, "info": "非 1:1 候选比例允许偏离 bbox 的最大倍数。"},
    "always_allow_square": {"component": "checkbox", "label": "always allow 1:1", "value": True, "info": "横竖 bbox 都允许少量输出正方形。"},
    "full_max_upscale": {"component": "number", "label": "full max upscale", "value": 2.0, "info": "full 模式小图最大放大倍数。"},
    "random_seed": {"component": "number", "label": "random seed", "value": 42, "info": "控制随机裁剪和随机输出复现。"},
    "face_expand_top": {"component": "number", "label": "face_expand_top", "value": 1.5, "info": "脸部 bbox 向上外扩倍数。"},
    "face_expand_bottom": {"component": "number", "label": "face_expand_bottom", "value": 2.0, "info": "脸部 bbox 向下外扩倍数。"},
    "face_expand_lr": {"component": "number", "label": "face_expand_lr", "value": 1.4, "info": "脸部 bbox 左右外扩倍数。"},
    "body_expand_tb": {"component": "number", "label": "body_expand_tb", "value": 1.15, "info": "身体 bbox 上下外扩倍数。"},
    "body_expand_lr": {"component": "number", "label": "body_expand_lr", "value": 1.2, "info": "身体 bbox 左右外扩倍数。"},
    "halfbody_expand_top": {"component": "number", "label": "halfbody_expand_top", "value": 1.2, "info": "半身 bbox 向上外扩倍数。"},
    "halfbody_expand_bottom": {"component": "number", "label": "halfbody_expand_bottom", "value": 1.25, "info": "半身 bbox 向下外扩倍数。"},
    "halfbody_expand_lr": {"component": "number", "label": "halfbody_expand_lr", "value": 1.2, "info": "半身 bbox 左右外扩倍数。"},
    "min_crop_size": {"component": "number", "label": "min_crop_size", "value": 128, "info": "小于该尺寸的 crop 跳过。"},
    "crop_png_compression": {"component": "slider", "label": "png_compression", "value": 3, "minimum": 0, "maximum": 9, "step": 1, "info": "裁剪 PNG 压缩等级。"},
    "target_crops_per_image": {"component": "number", "label": "target_crops_per_image", "value": 3, "info": "随机权重策略每图最多输出数。"},
}


def _param_controls(names: list[str], columns: int = 3) -> dict[str, Any]:
    controls: dict[str, Any] = {}
    for start in range(0, len(names), columns):
        row_names = names[start : start + columns]
        with gr.Row():
            for name in row_names:
                controls[name] = _param_component(name)
    return controls


def _param_component(name: str) -> Any:
    spec = PARAM_SCHEMA[name]
    kwargs = {
        "label": spec.get("label", name),
        "value": spec.get("value"),
        "info": spec.get("info"),
    }
    component = spec["component"]
    if component == "textbox":
        return gr.Textbox(**kwargs)
    if component == "number":
        return gr.Number(**kwargs)
    if component == "checkbox":
        return gr.Checkbox(**kwargs)
    if component == "dropdown":
        return gr.Dropdown(spec["choices"], **kwargs)
    if component == "checkbox_group":
        return gr.CheckboxGroup(spec["choices"], **kwargs)
    if component == "slider":
        return gr.Slider(
            minimum=spec["minimum"],
            maximum=spec["maximum"],
            step=spec["step"],
            **kwargs,
        )
    raise ValueError(f"unsupported parameter component: {component}")


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Anime Shot All") as app:
        config_state = gr.State({})
        video_state = gr.State([])
        extract_stop_state = gr.State({"stop": False})
        crop_stop_state = gr.State({"stop": False})
        gr.Markdown("# 动漫图像训练数据采集与裁剪工具")

        with gr.Tab("工作目录 / 视频 / 忽略区间 / 参数"):
            with gr.Row():
                work_dir = gr.Textbox(label="工作目录 work_dir", value="./work_dir", placeholder="./work_dir")
                open_work_dir = gr.Button("初始化 / 打开工作目录", variant="primary")
            video_dir = gr.Textbox(label="视频文件夹 video_dir", value="./video_dir", info="自动按配置中的视频扩展名过滤。")
            with gr.Row():
                scan_video_btn = gr.Button("扫描视频")
                load_params_btn = gr.Button("加载 params.yaml")
                save_params_btn = gr.Button("保存当前参数到 params.yaml")
                reset_params_btn = gr.Button("重置为 default.yaml")
            with gr.Accordion("YAML 导入 / 导出", open=False):
                yaml_import = gr.File(label="导入 YAML", file_types=[".yaml", ".yml"])
                import_yaml_btn = gr.Button("导入 YAML")
                export_yaml_path = gr.Textbox(label="导出 YAML 路径", value="configs/exported.yaml")
                export_yaml_btn = gr.Button("导出 YAML")
            video_table = gr.Dataframe(headers=VIDEO_HEADERS, label="视频列表", interactive=False, type="array")
            ignore_table = gr.Dataframe(headers=IGNORE_HEADERS, label="忽略区间配置", interactive=True, type="array")
            with gr.Row():
                load_ignore_btn = gr.Button("加载 ignore_ranges JSON")
                save_ignore_btn = gr.Button("保存 ignore_ranges JSON")
                validate_ignore_btn = gr.Button("配置校验")
            with gr.Row():
                ignore_csv_file = gr.File(label="导入 ignore_ranges CSV", file_types=[".csv"])
                import_ignore_csv_btn = gr.Button("导入 CSV")
                export_ignore_csv_path = gr.Textbox(label="导出 CSV 路径", value="states/ignore_ranges.csv")
                export_ignore_csv_btn = gr.Button("导出 CSV")
            setup_log = gr.Textbox(label="日志窗口", lines=8)

        with gr.Tab("截帧"):
            extract_output = _param_component("extract_output")
            video_selection = gr.CheckboxGroup(label="选择要截帧的视频", choices=[])
            with gr.Accordion("截帧参数", open=True):
                extract_params = _param_controls(
                    [
                        "interval",
                        "keyframe_only",
                        "png_compression",
                        "min_width",
                        "crop_bottom",
                    ],
                    columns=3,
                )
                interval = extract_params["interval"]
                keyframe_only = extract_params["keyframe_only"]
                png_compression = extract_params["png_compression"]
                min_width = extract_params["min_width"]
                crop_bottom = extract_params["crop_bottom"]
            with gr.Accordion("pHash 分组与随机保留", open=False):
                phash_params = _param_controls(
                    [
                        "phash_threshold",
                        "phash_size",
                        "phash_crop",
                        "phash_resize_width",
                        "group_seconds_per_keep",
                        "group_max_duration",
                        "extract_random_seed",
                    ],
                    columns=3,
                )
                phash_threshold = phash_params["phash_threshold"]
                phash_size = phash_params["phash_size"]
                phash_crop = phash_params["phash_crop"]
                phash_resize_width = phash_params["phash_resize_width"]
                group_seconds_per_keep = phash_params["group_seconds_per_keep"]
                group_max_duration = phash_params["group_max_duration"]
                extract_random_seed = phash_params["extract_random_seed"]
            with gr.Row():
                extract_btn = gr.Button("开始截帧", variant="primary")
                extract_stop_btn = gr.Button("停止截帧")
            extract_log = gr.Textbox(label="日志窗口", lines=8)

        with gr.Tab("裁剪"):
            with gr.Row():
                crop_input = _param_component("crop_input")
                crop_output = _param_component("crop_output")
            with gr.Accordion("输出类型与策略", open=True):
                crop_mode = _param_component("crop_mode")
                strategy_params = _param_controls(
                    [
                        "output_strategy",
                        "weight_full",
                        "weight_face",
                        "weight_body",
                        "weight_halfbody",
                        "weight_random",
                    ],
                    columns=3,
                )
                output_strategy = strategy_params["output_strategy"]
                weight_full = strategy_params["weight_full"]
                weight_face = strategy_params["weight_face"]
                weight_body = strategy_params["weight_body"]
                weight_halfbody = strategy_params["weight_halfbody"]
                weight_random = strategy_params["weight_random"]
            with gr.Accordion("imgutils 检测参数", open=False):
                detection_params = _param_controls(
                    [
                        "conf_threshold",
                        "iou_threshold",
                        "face_level",
                        "face_version",
                        "person_level",
                        "person_version",
                        "halfbody_level",
                        "halfbody_version",
                    ],
                    columns=4,
                )
                conf_threshold = detection_params["conf_threshold"]
                iou_threshold = detection_params["iou_threshold"]
                face_level = detection_params["face_level"]
                face_version = detection_params["face_version"]
                person_level = detection_params["person_level"]
                person_version = detection_params["person_version"]
                halfbody_level = detection_params["halfbody_level"]
                halfbody_version = detection_params["halfbody_version"]
            with gr.Accordion("自动比例与随机参数", open=False):
                aspect_params = _param_controls(
                    ["ratio_sigma", "max_ratio_change", "always_allow_square", "full_max_upscale", "random_seed"],
                    columns=3,
                )
                ratio_sigma = aspect_params["ratio_sigma"]
                max_ratio_change = aspect_params["max_ratio_change"]
                always_allow_square = aspect_params["always_allow_square"]
                full_max_upscale = aspect_params["full_max_upscale"]
                random_seed = aspect_params["random_seed"]
            with gr.Accordion("裁剪尺寸与边距", open=False):
                crop_size_params = _param_controls(
                    [
                        "face_expand_top",
                        "face_expand_bottom",
                        "face_expand_lr",
                        "body_expand_tb",
                        "body_expand_lr",
                        "halfbody_expand_top",
                        "halfbody_expand_bottom",
                        "halfbody_expand_lr",
                        "min_crop_size",
                        "crop_png_compression",
                        "target_crops_per_image",
                    ],
                    columns=3,
                )
                face_expand_top = crop_size_params["face_expand_top"]
                face_expand_bottom = crop_size_params["face_expand_bottom"]
                face_expand_lr = crop_size_params["face_expand_lr"]
                body_expand_tb = crop_size_params["body_expand_tb"]
                body_expand_lr = crop_size_params["body_expand_lr"]
                halfbody_expand_top = crop_size_params["halfbody_expand_top"]
                halfbody_expand_bottom = crop_size_params["halfbody_expand_bottom"]
                halfbody_expand_lr = crop_size_params["halfbody_expand_lr"]
                min_crop_size = crop_size_params["min_crop_size"]
                crop_png_compression = crop_size_params["crop_png_compression"]
                target_crops_per_image = crop_size_params["target_crops_per_image"]
            with gr.Row():
                crop_btn = gr.Button("开始裁剪", variant="primary")
                crop_stop_btn = gr.Button("停止裁剪")
            crop_log = gr.Textbox(label="日志窗口", lines=8)

        with gr.Tab("日志 / 输出"):
            refresh_stats_btn = gr.Button("刷新统计")
            stats_output = gr.Textbox(label="统计信息", lines=14)
            paths_output = gr.Textbox(label="当前路径", lines=8)
            recent_logs = gr.Textbox(label="最近一次任务日志", lines=12)

        param_inputs = [
            video_dir,
            extract_output,
            interval,
            keyframe_only,
            png_compression,
            min_width,
            crop_bottom,
            phash_threshold,
            phash_size,
            phash_crop,
            phash_resize_width,
            group_seconds_per_keep,
            group_max_duration,
            extract_random_seed,
            crop_input,
            crop_output,
            crop_mode,
            output_strategy,
            weight_full,
            weight_face,
            weight_body,
            weight_halfbody,
            weight_random,
            ratio_sigma,
            max_ratio_change,
            always_allow_square,
            full_max_upscale,
            random_seed,
            conf_threshold,
            iou_threshold,
            face_level,
            face_version,
            person_level,
            person_version,
            halfbody_level,
            halfbody_version,
            face_expand_top,
            face_expand_bottom,
            face_expand_lr,
            body_expand_tb,
            body_expand_lr,
            halfbody_expand_top,
            halfbody_expand_bottom,
            halfbody_expand_lr,
            min_crop_size,
            crop_png_compression,
            target_crops_per_image,
        ]

        open_work_dir.click(
            _open_project,
            inputs=[work_dir],
            outputs=[config_state, video_dir, extract_output, crop_input, crop_output, ignore_table, setup_log],
        )
        load_params_btn.click(_load_params, inputs=[work_dir], outputs=[config_state, setup_log, *param_inputs])
        save_params_btn.click(_save_params_from_gui, inputs=[work_dir, config_state, *param_inputs], outputs=[config_state, setup_log])
        reset_params_btn.click(_reset_params, inputs=[work_dir], outputs=[config_state, setup_log, *param_inputs])
        import_yaml_btn.click(_import_yaml, inputs=[work_dir, yaml_import], outputs=[config_state, setup_log, *param_inputs])
        export_yaml_btn.click(_export_yaml, inputs=[work_dir, config_state, export_yaml_path], outputs=[setup_log])

        scan_video_btn.click(
            _scan_videos,
            inputs=[work_dir, video_dir, config_state],
            outputs=[video_state, video_table, setup_log, video_selection],
        )
        load_ignore_btn.click(_load_ignore, inputs=[work_dir], outputs=[ignore_table, setup_log])
        save_ignore_btn.click(_save_ignore, inputs=[work_dir, ignore_table, video_state], outputs=[setup_log])
        validate_ignore_btn.click(_validate_ignore, inputs=[work_dir, ignore_table, video_state, config_state], outputs=[ignore_table, setup_log])
        import_ignore_csv_btn.click(_import_ignore_csv, inputs=[ignore_csv_file], outputs=[ignore_table, setup_log])
        export_ignore_csv_btn.click(_export_ignore_csv, inputs=[work_dir, ignore_table, export_ignore_csv_path], outputs=[setup_log])

        extract_btn.click(
            _run_extract,
            inputs=[work_dir, config_state, video_state, video_selection, extract_stop_state, *param_inputs],
            outputs=[config_state, extract_log, extract_stop_state],
        )
        extract_stop_btn.click(_request_stop, inputs=[extract_stop_state], outputs=[extract_stop_state, extract_log])
        crop_btn.click(_run_crop_gui, inputs=[work_dir, config_state, crop_stop_state, *param_inputs], outputs=[config_state, crop_log, crop_stop_state])
        crop_stop_btn.click(_request_stop, inputs=[crop_stop_state], outputs=[crop_stop_state, crop_log])
        refresh_stats_btn.click(_refresh_stats, inputs=[work_dir, config_state], outputs=[stats_output, paths_output, recent_logs])

    return app


def _open_project(work_dir: str):
    config, messages = initialize_work_dir(work_dir)
    root = Path(config["project"]["work_dir"])
    ignore_rows = state_to_rows(load_ignore_state(root))
    return (
        config,
        str(resolve_work_path(root, config["project"].get("video_dir", "videos"))),
        str(resolve_work_path(root, config["paths"]["frames_raw"])),
        str(resolve_work_path(root, config["crop"]["input_dir"])),
        str(resolve_work_path(root, config["crop"]["output_dir"])),
        ignore_rows,
        "\n".join(messages),
    )


def _load_params(work_dir: str):
    config = load_project_config(work_dir)
    return (config, f"loaded {Path(work_dir) / 'configs' / 'params.yaml'}", *_values_from_config(config))


def _save_params_from_gui(work_dir: str, config: dict[str, Any], *values: Any):
    updated = _apply_gui_values(config, values)
    path = save_params(work_dir, updated)
    return updated, f"saved {path}"


def _reset_params(work_dir: str):
    config = reset_params_from_default(work_dir)
    return (config, "reset params.yaml from default.yaml", *_values_from_config(config))


def _import_yaml(work_dir: str, file_obj: Any):
    if file_obj is None:
        return (load_project_config(work_dir), "no YAML selected", *_values_from_config(load_project_config(work_dir)))
    import yaml

    with open(file_obj.name, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    path = save_params(work_dir, data)
    config = load_project_config(work_dir)
    return (config, f"imported YAML to {path}", *_values_from_config(config))


def _export_yaml(work_dir: str, config: dict[str, Any], export_path: str):
    root = Path(work_dir).expanduser().resolve()
    path = resolve_work_path(root, export_path)
    write_yaml(path, config)
    return f"exported {path}"


def _scan_videos(work_dir: str, video_dir: str, config: dict[str, Any]):
    root = Path(work_dir).expanduser().resolve()
    config = _config_for_scan(root, config)
    video_path = resolve_work_path(root, video_dir)
    if not video_path.exists():
        yield _empty_video_scan(f"视频文件夹不存在: {video_path}")
        return
    if not video_path.is_dir():
        yield _empty_video_scan(f"视频路径不是文件夹: {video_path}")
        return
    candidates = video_candidates(video_path, config["project"]["supported_video_ext"])
    logs = [f"scan videos: {video_path}", f"matched files: {len(candidates)}"]
    videos: list[VideoInfo] = []
    yield [], [], "\n".join(logs), gr.update(choices=[], value=[])
    for index, path in enumerate(candidates, start=1):
        logs.append(format_progress("scan", index, len(candidates), path))
        yield videos_as_dicts(videos), videos_to_rows(videos), "\n".join(logs), gr.update(choices=[item.video_name for item in videos], value=[item.video_name for item in videos])
        try:
            video = probe_video(path, f"ep{len(videos) + 1:02d}", root)
        except RuntimeError as error:
            logs.append(str(error))
            continue
        videos.append(video)
        logs.append(f"{video.episode_id}: {video.video_name}")
    choices = [item.video_name for item in videos]
    logs.append(f"scanned {len(videos)} videos")
    yield videos_as_dicts(videos), videos_to_rows(videos), "\n".join(logs), gr.update(choices=choices, value=choices)


def _config_for_scan(root: Path, config: dict[str, Any] | None) -> dict[str, Any]:
    defaults = builtin_defaults()
    defaults["project"]["work_dir"] = str(root)
    try:
        base = load_project_config(root)
    except FileNotFoundError:
        base = defaults
    return deep_merge(base, config or {})


def _empty_video_scan(message: str):
    return [], [], message, gr.update(choices=[], value=[])


def _load_ignore(work_dir: str):
    root = Path(work_dir).expanduser().resolve()
    return state_to_rows(load_ignore_state(root)), f"loaded {root / 'states' / 'ignore_ranges.json'}"


def _save_ignore(work_dir: str, rows: list[list[Any]], videos: list[dict[str, Any]]):
    root = Path(work_dir).expanduser().resolve()
    lookup = {item["episode_id"]: item["video_path"] for item in videos}
    path = save_ignore_state(root, rows_to_state(rows or [], root, lookup))
    return f"saved {path}"


def _validate_ignore(work_dir: str, rows: list[list[Any]], videos: list[dict[str, Any]], config: dict[str, Any]):
    root = Path(work_dir).expanduser().resolve()
    lookup = {item["episode_id"]: item["video_path"] for item in videos}
    durations = {item["episode_id"]: item["duration_sec"] for item in videos}
    state = rows_to_state(rows or [], root, lookup)
    normalized, warnings, errors = normalize_ranges(state, root, durations, config["ignore_ranges"].get("auto_merge_overlaps", True))
    return state_to_rows(normalized), "\n".join([*warnings, *errors]) or "ignore_ranges valid"


def _import_ignore_csv(file_obj: Any):
    if file_obj is None:
        return [], "no CSV selected"
    rows = import_csv(Path(file_obj.name))
    return rows, f"imported {len(rows)} rows"


def _export_ignore_csv(work_dir: str, rows: list[list[Any]], export_path: str):
    path = export_csv(resolve_work_path(Path(work_dir).expanduser().resolve(), export_path), rows or [])
    return f"exported {path}"


def _run_extract(
    work_dir: str,
    config: dict[str, Any],
    videos: list[dict[str, Any]],
    selected: list[str],
    stop_state: dict[str, Any],
    *values: Any,
):
    updated = _apply_gui_values(config, values)
    if stop_state is None:
        stop_state = {}
    stop_state["stop"] = False
    root = Path(work_dir).expanduser().resolve()
    selected_set = set(selected or [])
    video_objs = [VideoInfo(**item) for item in videos if not selected_set or item.get("video_name") in selected_set]
    output_dir = resolve_work_path(root, updated["paths"]["frames_raw"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = resolve_work_path(root, updated["logging"]["extract_log"])
    ignore_state = load_ignore_state(root)
    rows: list[dict[str, object]] = []
    logs: list[str] = []

    def append_log(message: str) -> None:
        logs.append(message)
        print(message, flush=True)

    append_log(f"extract videos: {len(video_objs)}")
    append_log(f"output: {output_dir}")
    saved_total = 0
    yield updated, "\n".join(logs), stop_state
    for index, video in enumerate(video_objs, start=1):
        if stop_state and stop_state.get("stop"):
            append_log("stopped by user")
            break
        append_log(format_progress("extract", index, len(video_objs), video.video_name))
        yield updated, "\n".join(logs), stop_state
        saved, video_rows = extract_frames_for_video(
            root,
            updated,
            video,
            ignore_state,
            output_dir,
            stop_state=stop_state,
            progress=append_log,
        )
        rows.extend(video_rows)
        saved_total += saved
        append_log(f"{video.episode_id}: saved {saved} frames from {video.video_name}")
        yield updated, "\n".join(logs), stop_state
    write_csv(log_path, EXTRACT_LOG_FIELDS, rows)
    append_log(f"saved total: {saved_total}")
    append_log(f"log: {log_path}")
    yield updated, "\n".join(logs), stop_state




def _run_crop_gui(work_dir: str, config: dict[str, Any], stop_state: dict[str, Any], *values: Any):
    updated = _apply_gui_values(config, values)
    if stop_state is None:
        stop_state = {}
    stop_state["stop"] = False
    root = Path(work_dir).expanduser().resolve()
    params = updated["crop"]
    input_dir = resolve_work_path(root, params.get("input_dir", "frames_dedup"))
    output_dir = resolve_work_path(root, params.get("output_dir", "crops"))
    output_dir.mkdir(parents=True, exist_ok=True)
    images = collect_images(input_dir)
    rng = random.Random(int(params.get("random_seed", 42)))
    rows: list[dict[str, object]] = []
    saved_total = 0
    log_path = resolve_work_path(root, updated["logging"]["crop_log"])
    logs = [f"crop images: {len(images)}", f"input: {input_dir}", f"output: {output_dir}"]
    yield updated, "\n".join(logs), stop_state
    for index, image_path in enumerate(images, start=1):
        if stop_state and stop_state.get("stop"):
            logs.append("stopped by user")
            break
        logs.append(format_progress("crop", index, len(images), image_path))
        yield updated, "\n".join(logs), stop_state
        image_saved, image_rows = crop_one_image(root, updated, image_path, output_dir, rng)
        saved_total += image_saved
        rows.extend(image_rows)
        logs.append(f"{image_path.name}: saved {image_saved} crops, total {saved_total}")
        yield updated, "\n".join(logs), stop_state
    write_csv(log_path, CROP_LOG_FIELDS, rows)
    logs.append(f"saved crops: {saved_total}")
    logs.append(f"log: {log_path}")
    yield updated, "\n".join(logs), stop_state


def _request_stop(stop_state: dict[str, Any]):
    if stop_state is None:
        stop_state = {}
    stop_state["stop"] = True
    return stop_state, "stop requested"


def _refresh_stats(work_dir: str, config: dict[str, Any]):
    root = Path(work_dir).expanduser().resolve()
    summary = summarize_project(root, config)
    paths = [
        f"work_dir: {root}",
        f"params: {root / 'configs' / 'params.yaml'}",
        f"ignore_ranges: {root / 'states' / 'ignore_ranges.json'}",
    ]
    logs = []
    for key in ["extract_log", "crop_log"]:
        log_path = resolve_work_path(root, config["logging"][key])
        text = recent_log_text(log_path)
        if text:
            logs.append(f"[{key}]\n{text}")
    return format_summary(summary), "\n".join(paths), "\n\n".join(logs)


def _apply_gui_values(config: dict[str, Any], values: tuple[Any, ...]) -> dict[str, Any]:
    cfg = dict(config)
    cfg = _deep_copy_config(config)
    (
        video_dir,
        extract_output,
        interval,
        keyframe_only,
        png_compression,
        min_width,
        crop_bottom,
        phash_threshold,
        phash_size,
        phash_crop,
        phash_resize_width,
        group_seconds_per_keep,
        group_max_duration,
        extract_random_seed,
        crop_input,
        crop_output,
        crop_mode,
        output_strategy,
        weight_full,
        weight_face,
        weight_body,
        weight_halfbody,
        weight_random,
        ratio_sigma,
        max_ratio_change,
        always_allow_square,
        full_max_upscale,
        random_seed,
        conf_threshold,
        iou_threshold,
        face_level,
        face_version,
        person_level,
        person_version,
        halfbody_level,
        halfbody_version,
        face_expand_top,
        face_expand_bottom,
        face_expand_lr,
        body_expand_tb,
        body_expand_lr,
        halfbody_expand_top,
        halfbody_expand_bottom,
        halfbody_expand_lr,
        min_crop_size,
        crop_png_compression,
        target_crops_per_image,
    ) = values
    root = Path(cfg["project"]["work_dir"]).expanduser().resolve()
    cfg["project"]["video_dir"] = relative_path_value(video_dir, root)
    cfg["paths"]["frames_raw"] = relative_path_value(extract_output, root)
    cfg["extract"].update(
        {
            "interval": float(interval),
            "keyframe_only": bool(keyframe_only),
            "png_compression": int(png_compression),
            "min_width": int(min_width),
            "crop_bottom": int(crop_bottom),
            "phash_threshold": int(phash_threshold),
            "phash_size": int(phash_size),
            "phash_crop": phash_crop,
            "phash_resize_width": int(phash_resize_width),
            "group_seconds_per_keep": float(group_seconds_per_keep),
            "group_max_duration": float(group_max_duration),
            "extract_random_seed": int(extract_random_seed),
        }
    )
    cfg["crop"].update(
        {
            "input_dir": relative_path_value(crop_input, root),
            "output_dir": relative_path_value(crop_output, root),
            "output_strategy": output_strategy,
            "random_seed": int(random_seed),
            "min_crop_size": int(min_crop_size),
            "png_compression": int(crop_png_compression),
            "target_crops_per_image": int(target_crops_per_image),
        }
    )
    cfg.setdefault("full_crop", {})["max_upscale"] = float(full_max_upscale)
    cfg.setdefault("ratio_selection", {}).update(
        {
            "sigma": float(ratio_sigma),
            "max_ratio_change": float(max_ratio_change),
            "always_allow_square": bool(always_allow_square),
        }
    )
    cfg["crop_types"] = {key: key in (crop_mode or []) for key in ["full", "face", "body", "halfbody", "random_crop"]}
    cfg["random_output_weights"].update(
        {
            "full": int(weight_full),
            "face": int(weight_face),
            "body": int(weight_body),
            "halfbody": int(weight_halfbody),
            "random_crop": int(weight_random),
        }
    )
    cfg.setdefault("detection", {}).update(
        {
            "conf_threshold": float(conf_threshold),
            "iou_threshold": float(iou_threshold),
            "face_level": face_level,
            "face_version": face_version,
            "person_level": person_level,
            "person_version": person_version,
            "halfbody_level": halfbody_level,
            "halfbody_version": halfbody_version,
        }
    )
    cfg["face_crop"].update(
        {
            "expand_top": float(face_expand_top),
            "expand_bottom": float(face_expand_bottom),
            "expand_left": float(face_expand_lr),
            "expand_right": float(face_expand_lr),
        }
    )
    cfg["body_crop"].update(
        {
            "expand_top": float(body_expand_tb),
            "expand_bottom": float(body_expand_tb),
            "expand_left": float(body_expand_lr),
            "expand_right": float(body_expand_lr),
        }
    )
    cfg.setdefault("halfbody_crop", {})
    cfg["halfbody_crop"].update(
        {
            "expand_top": float(halfbody_expand_top),
            "expand_bottom": float(halfbody_expand_bottom),
            "expand_left": float(halfbody_expand_lr),
            "expand_right": float(halfbody_expand_lr),
        }
    )
    return cfg


def _values_from_config(config: dict[str, Any]) -> tuple[Any, ...]:
    enabled_types = [key for key in ["full", "face", "body", "halfbody", "random_crop"] if config["crop_types"].get(key)]
    ratio_selection = config.get("ratio_selection", {})
    full_crop = config.get("full_crop", {})
    return (
        config["project"].get("video_dir", "videos"),
        config["paths"]["frames_raw"],
        config["extract"]["interval"],
        config["extract"].get("keyframe_only", False),
        config["extract"]["png_compression"],
        config["extract"]["min_width"],
        config["extract"]["crop_bottom"],
        config["extract"].get("phash_threshold", 5),
        config["extract"].get("phash_size", 8),
        config["extract"].get("phash_crop", "center"),
        config["extract"].get("phash_resize_width", 256),
        config["extract"].get("group_seconds_per_keep", 5.0),
        config["extract"].get("group_max_duration", 60.0),
        config["extract"].get("extract_random_seed", 42),
        config["crop"]["input_dir"],
        config["crop"]["output_dir"],
        enabled_types,
        config["crop"]["output_strategy"],
        config["random_output_weights"]["full"],
        config["random_output_weights"]["face"],
        config["random_output_weights"]["body"],
        config["random_output_weights"].get("halfbody", 25),
        config["random_output_weights"]["random_crop"],
        ratio_selection.get("sigma", 0.45),
        ratio_selection.get("max_ratio_change", 2.2),
        ratio_selection.get("always_allow_square", True),
        full_crop.get("max_upscale", 2.0),
        config["crop"]["random_seed"],
        config.get("detection", {}).get("conf_threshold", 0.35),
        config.get("detection", {}).get("iou_threshold", 0.7),
        config.get("detection", {}).get("face_level", "s"),
        config.get("detection", {}).get("face_version", "v1.4"),
        config.get("detection", {}).get("person_level", "m"),
        config.get("detection", {}).get("person_version", "v1.1"),
        config.get("detection", {}).get("halfbody_level", "s"),
        config.get("detection", {}).get("halfbody_version", "v1.0"),
        config["face_crop"].get("expand_top", 1.5),
        config["face_crop"].get("expand_bottom", 2.0),
        config["face_crop"].get("expand_left", 1.4),
        config["body_crop"].get("expand_top", 1.15),
        config["body_crop"].get("expand_left", 1.2),
        config.get("halfbody_crop", {}).get("expand_top", 1.2),
        config.get("halfbody_crop", {}).get("expand_bottom", 1.25),
        config.get("halfbody_crop", {}).get("expand_left", 1.2),
        config["crop"]["min_crop_size"],
        config["crop"]["png_compression"],
        config["crop"]["target_crops_per_image"],
    )


def _deep_copy_config(config: dict[str, Any]) -> dict[str, Any]:
    import copy

    return copy.deepcopy(config)

