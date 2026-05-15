"""Gradio GUI wiring for the anime image dataset tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr

from .config import (
    initialize_work_dir,
    load_project_config,
    reset_params_from_default,
    resolve_work_path,
    save_params,
    write_yaml,
)
from .crop import run_crop
from .dedup import (
    analyze_duplicates,
    export_dedup_results,
    group_gallery_items,
    keep_all_in_group,
    load_dedup_state,
    save_dedup_state,
    update_group_decision,
)
from .extract import extract_frames_for_videos
from .ignore_ranges import (
    export_csv,
    import_csv,
    load_ignore_state,
    normalize_ranges,
    save_ignore_state,
    state_to_rows,
    rows_to_state,
)
from .files import relative_path_value
from .stats import format_summary, recent_log_text, summarize_project
from .video import VideoInfo, scan_videos, videos_as_dicts, videos_to_rows


IGNORE_HEADERS = ["episode_id", "video_name", "ignore_start", "ignore_end", "label", "enabled", "notes"]
VIDEO_HEADERS = ["episode_id", "video_path", "video_name", "duration_sec", "fps", "width", "height"]
ASPECT_OPTIONS = ["original", "square", "portrait_2_3", "portrait_3_4", "portrait_9_16", "landscape_16_9", "landscape_4_3", "random"]

PARAM_SCHEMA: dict[str, dict[str, Any]] = {
    "extract_output": {"component": "textbox", "label": "输出 frames_raw 文件夹", "info": "PNG 截帧保存目录。"},
    "interval": {"component": "number", "label": "interval", "value": 0.25, "info": "采样间隔秒数；越小检查越频繁。"},
    "diff_threshold": {"component": "number", "label": "diff_threshold", "value": 5, "info": "画面差异阈值；越低保存越多。"},
    "max_gap": {"component": "number", "label": "max_gap", "value": 2.0, "info": "最长强制保存间隔秒数。"},
    "png_compression": {"component": "slider", "label": "png_compression", "value": 3, "minimum": 0, "maximum": 9, "step": 1, "info": "PNG 压缩等级；越高越慢。"},
    "min_width": {"component": "number", "label": "min_width", "value": 0, "info": "输出最小宽度；0 表示不放大。"},
    "crop_bottom": {"component": "number", "label": "crop_bottom", "value": 0, "info": "从底部裁掉像素，可避开硬字幕。"},
    "resize_width_for_diff": {"component": "number", "label": "resize_width_for_diff", "value": 320, "info": "差异计算缩放宽度；越小越快。"},
    "scene_diff_method": {"component": "dropdown", "label": "scene_diff_method", "value": "gray_mean_absdiff", "choices": ["gray_mean_absdiff"], "info": "当前支持灰度平均绝对差。"},
    "reset_diff_after_ignore": {"component": "checkbox", "label": "reset_diff_after_ignore", "value": True, "info": "离开忽略区间后重置 diff 基准。"},
    "dedup_input": {"component": "textbox", "label": "输入图片目录", "info": "通常为 frames_raw。"},
    "dedup_output": {"component": "textbox", "label": "输出 frames_dedup 目录", "info": "人工确认后复制保留图片。"},
    "dedup_scope": {"component": "dropdown", "label": "dedup_scope", "value": "per_episode", "choices": ["per_episode", "global", "custom"], "info": "per_episode 更保守；global 更激进。"},
    "episode_filter": {"component": "textbox", "label": "episode_filter", "value": "all", "info": "all 或 ep01 等单集编号。"},
    "hash_threshold": {"component": "number", "label": "hash_threshold", "value": 5, "info": "pHash 距离阈值；越大越激进。"},
    "hash_size": {"component": "number", "label": "hash_size", "value": 8, "info": "pHash 尺寸；越大越细。"},
    "hash_crop": {"component": "dropdown", "label": "hash_crop", "value": "center", "choices": ["center", "full"], "info": "center 更关注主体区域。"},
    "hash_resize_width": {"component": "number", "label": "hash_resize_width", "value": 256, "info": "hash 前缩放宽度；越小越快。"},
    "num_workers": {"component": "number", "label": "num_workers", "value": -1, "info": "-1 表示自动选择。"},
    "export_rejected": {"component": "checkbox", "label": "export_rejected_duplicates", "value": True, "info": "复制被去重图片便于复查。"},
    "crop_input": {"component": "textbox", "label": "输入图片文件夹", "info": "通常为 frames_dedup。"},
    "crop_output": {"component": "textbox", "label": "输出 crops 文件夹", "info": "最终 PNG 裁剪输出目录。"},
    "crop_mode": {"component": "checkbox_group", "label": "输出类型", "choices": ["full", "hard_split", "face", "body", "background", "random_crop"], "value": ["full", "hard_split", "face", "body", "background", "random_crop"], "info": "未勾选的类型不会输出，也不会触发对应检测。"},
    "output_strategy": {"component": "dropdown", "label": "输出策略", "value": "fixed", "choices": ["fixed", "random_weighted"], "info": "fixed 全部尝试；random_weighted 按权重抽样。"},
    "weight_full": {"component": "number", "label": "full weight", "value": 0, "info": "随机策略下 full 权重。"},
    "weight_hard": {"component": "number", "label": "hard_split weight", "value": 0, "info": "随机策略下硬裁权重。"},
    "weight_face": {"component": "number", "label": "face weight", "value": 30, "info": "随机策略下脸部权重。"},
    "weight_body": {"component": "number", "label": "body weight", "value": 30, "info": "随机策略下身体权重。"},
    "weight_background": {"component": "number", "label": "background weight", "value": 20, "info": "随机策略下背景权重。"},
    "weight_random": {"component": "number", "label": "random_crop weight", "value": 20, "info": "随机策略下随机裁剪权重。"},
    "body_model_path": {"component": "textbox", "label": "body YOLO 权重", "info": "body/background/avoid_body 需要。"},
    "face_model_path": {"component": "textbox", "label": "face YOLO 权重", "info": "face 裁剪需要。"},
    "conf": {"component": "number", "label": "conf", "value": 0.35, "info": "检测置信度阈值。"},
    "imgsz": {"component": "number", "label": "imgsz", "value": 960, "info": "YOLO 推理输入尺寸。"},
    "body_class_id": {"component": "number", "label": "body class id", "value": 0, "info": "body/person 类别 ID。"},
    "face_class_id": {"component": "number", "label": "face class id", "value": 0, "info": "face_all_classes 关闭时生效。"},
    "face_all_classes": {"component": "checkbox", "label": "face all classes", "value": True, "info": "启用后 face 模型所有类别都接受。"},
    "face_aspect": {"component": "dropdown", "label": "face aspect", "value": "square", "choices": ASPECT_OPTIONS, "info": "face 输出比例。"},
    "body_aspect": {"component": "dropdown", "label": "body aspect", "value": "portrait_2_3", "choices": ASPECT_OPTIONS, "info": "body 输出比例。"},
    "background_aspect": {"component": "dropdown", "label": "background aspect", "value": "landscape_16_9", "choices": ASPECT_OPTIONS, "info": "background 输出比例。"},
    "random_seed": {"component": "number", "label": "random seed", "value": 42, "info": "控制随机裁剪和随机输出复现。"},
    "random_aspect_pool": {"component": "checkbox_group", "label": "random crop aspect pool", "choices": ["1:1", "2:3", "3:4", "9:16", "16:9", "4:3"], "value": ["1:1", "2:3", "16:9"], "info": "随机裁剪可选比例池。"},
    "face_padding": {"component": "number", "label": "face_padding", "value": 0.5, "info": "脸部框外扩比例，避免只裁五官。"},
    "body_padding_x": {"component": "number", "label": "body_padding_x", "value": 0.18, "info": "身体框水平外扩比例。"},
    "body_padding_y": {"component": "number", "label": "body_padding_y", "value": 0.25, "info": "身体框垂直外扩比例。"},
    "background_exclusion_padding": {"component": "number", "label": "background_exclusion_padding", "value": 0.15, "info": "人物禁区外扩比例。"},
    "background_max_overlap": {"component": "number", "label": "background_max_overlap", "value": 0.05, "info": "背景 crop 允许重叠上限。"},
    "min_crop_size": {"component": "number", "label": "min_crop_size", "value": 128, "info": "小于该尺寸的 crop 跳过。"},
    "max_side": {"component": "number", "label": "max_side", "value": 2048, "info": "输出最长边上限。"},
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
        dedup_state = gr.State({"config": {}, "groups": [], "unique_images": []})
        gr.Markdown("# 动漫图像训练数据采集与裁剪工具")

        with gr.Tab("工作目录 / 视频 / 忽略区间 / 参数"):
            with gr.Row():
                work_dir = gr.Textbox(label="工作目录 work_dir", placeholder="/data/anime_dataset/project_a")
                open_work_dir = gr.Button("初始化 / 打开工作目录", variant="primary")
            video_dir = gr.Textbox(label="视频文件夹 video_dir")
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
            with gr.Accordion("截帧参数", open=True):
                extract_params = _param_controls(
                    [
                        "interval",
                        "diff_threshold",
                        "max_gap",
                        "png_compression",
                        "min_width",
                        "crop_bottom",
                        "resize_width_for_diff",
                        "scene_diff_method",
                        "reset_diff_after_ignore",
                    ],
                    columns=3,
                )
                interval = extract_params["interval"]
                diff_threshold = extract_params["diff_threshold"]
                max_gap = extract_params["max_gap"]
                png_compression = extract_params["png_compression"]
                min_width = extract_params["min_width"]
                crop_bottom = extract_params["crop_bottom"]
                resize_width_for_diff = extract_params["resize_width_for_diff"]
                scene_diff_method = extract_params["scene_diff_method"]
                reset_diff_after_ignore = extract_params["reset_diff_after_ignore"]
            extract_btn = gr.Button("开始截帧", variant="primary")
            extract_log = gr.Textbox(label="日志窗口", lines=8)

        with gr.Tab("去重"):
            with gr.Row():
                dedup_input = _param_component("dedup_input")
                dedup_output = _param_component("dedup_output")
            with gr.Accordion("去重参数", open=True):
                dedup_params = _param_controls(
                    [
                        "dedup_scope",
                        "episode_filter",
                        "hash_threshold",
                        "hash_size",
                        "hash_crop",
                        "hash_resize_width",
                        "num_workers",
                        "export_rejected",
                    ],
                    columns=4,
                )
                dedup_scope = dedup_params["dedup_scope"]
                episode_filter = dedup_params["episode_filter"]
                hash_threshold = dedup_params["hash_threshold"]
                hash_size = dedup_params["hash_size"]
                hash_crop = dedup_params["hash_crop"]
                hash_resize_width = dedup_params["hash_resize_width"]
                num_workers = dedup_params["num_workers"]
                export_rejected = dedup_params["export_rejected"]
            with gr.Row():
                analyze_btn = gr.Button("分析重复", variant="primary")
                load_dedup_btn = gr.Button("加载 dedup_state")
                save_dedup_btn = gr.Button("保存 dedup_state")
            dedup_info = gr.Textbox(label="当前去重任务信息", lines=4)
            with gr.Row():
                group_dropdown = gr.Dropdown(label="group_id", choices=[])
                prev_group_btn = gr.Button("上一组")
                next_group_btn = gr.Button("下一组")
            group_gallery = gr.Gallery(label="当前重复组", columns=4, height=360)
            keep_choices = gr.CheckboxGroup(label="选择本组要保留的图片", choices=[])
            with gr.Row():
                save_group_btn = gr.Button("保存本组决策")
                keep_all_btn = gr.Button("本组全部保留")
                keep_selected_btn = gr.Button("本组只保留选中")
                export_dedup_btn = gr.Button("确认去重并导出", variant="primary")
            dedup_log = gr.Textbox(label="日志窗口", lines=8)

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
                        "weight_hard",
                        "weight_face",
                        "weight_body",
                        "weight_background",
                        "weight_random",
                    ],
                    columns=4,
                )
                output_strategy = strategy_params["output_strategy"]
                weight_full = strategy_params["weight_full"]
                weight_hard = strategy_params["weight_hard"]
                weight_face = strategy_params["weight_face"]
                weight_body = strategy_params["weight_body"]
                weight_background = strategy_params["weight_background"]
                weight_random = strategy_params["weight_random"]
            with gr.Accordion("YOLO 检测参数", open=False):
                yolo_params = _param_controls(
                    [
                        "body_model_path",
                        "face_model_path",
                        "conf",
                        "imgsz",
                        "body_class_id",
                        "face_class_id",
                        "face_all_classes",
                    ],
                    columns=3,
                )
                body_model_path = yolo_params["body_model_path"]
                face_model_path = yolo_params["face_model_path"]
                conf = yolo_params["conf"]
                imgsz = yolo_params["imgsz"]
                body_class_id = yolo_params["body_class_id"]
                face_class_id = yolo_params["face_class_id"]
                face_all_classes = yolo_params["face_all_classes"]
            with gr.Accordion("比例与随机参数", open=False):
                aspect_params = _param_controls(
                    ["face_aspect", "body_aspect", "background_aspect", "random_seed"],
                    columns=4,
                )
                face_aspect = aspect_params["face_aspect"]
                body_aspect = aspect_params["body_aspect"]
                background_aspect = aspect_params["background_aspect"]
                random_seed = aspect_params["random_seed"]
                random_aspect_pool = _param_component("random_aspect_pool")
            with gr.Accordion("裁剪尺寸与边距", open=False):
                crop_size_params = _param_controls(
                    [
                        "face_padding",
                        "body_padding_x",
                        "body_padding_y",
                        "background_exclusion_padding",
                        "background_max_overlap",
                        "min_crop_size",
                        "max_side",
                        "crop_png_compression",
                        "target_crops_per_image",
                    ],
                    columns=3,
                )
                face_padding = crop_size_params["face_padding"]
                body_padding_x = crop_size_params["body_padding_x"]
                body_padding_y = crop_size_params["body_padding_y"]
                background_exclusion_padding = crop_size_params["background_exclusion_padding"]
                background_max_overlap = crop_size_params["background_max_overlap"]
                min_crop_size = crop_size_params["min_crop_size"]
                max_side = crop_size_params["max_side"]
                crop_png_compression = crop_size_params["crop_png_compression"]
                target_crops_per_image = crop_size_params["target_crops_per_image"]
            crop_btn = gr.Button("开始裁剪", variant="primary")
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
            diff_threshold,
            max_gap,
            png_compression,
            min_width,
            crop_bottom,
            resize_width_for_diff,
            scene_diff_method,
            reset_diff_after_ignore,
            dedup_input,
            dedup_output,
            dedup_scope,
            episode_filter,
            hash_threshold,
            hash_size,
            hash_crop,
            hash_resize_width,
            num_workers,
            export_rejected,
            crop_input,
            crop_output,
            body_model_path,
            face_model_path,
            crop_mode,
            output_strategy,
            weight_full,
            weight_hard,
            weight_face,
            weight_body,
            weight_background,
            weight_random,
            face_aspect,
            body_aspect,
            background_aspect,
            random_seed,
            random_aspect_pool,
            conf,
            imgsz,
            body_class_id,
            face_class_id,
            face_all_classes,
            face_padding,
            body_padding_x,
            body_padding_y,
            background_exclusion_padding,
            background_max_overlap,
            min_crop_size,
            max_side,
            crop_png_compression,
            target_crops_per_image,
        ]

        open_work_dir.click(
            _open_project,
            inputs=[work_dir],
            outputs=[config_state, video_dir, extract_output, dedup_input, dedup_output, crop_input, crop_output, ignore_table, setup_log],
        )
        load_params_btn.click(_load_params, inputs=[work_dir], outputs=[config_state, setup_log, *param_inputs])
        save_params_btn.click(_save_params_from_gui, inputs=[work_dir, config_state, *param_inputs], outputs=[config_state, setup_log])
        reset_params_btn.click(_reset_params, inputs=[work_dir], outputs=[config_state, setup_log, *param_inputs])
        import_yaml_btn.click(_import_yaml, inputs=[work_dir, yaml_import], outputs=[config_state, setup_log, *param_inputs])
        export_yaml_btn.click(_export_yaml, inputs=[work_dir, config_state, export_yaml_path], outputs=[setup_log])

        scan_video_btn.click(_scan_videos, inputs=[work_dir, video_dir, config_state], outputs=[video_state, video_table, setup_log, episode_filter])
        load_ignore_btn.click(_load_ignore, inputs=[work_dir], outputs=[ignore_table, setup_log])
        save_ignore_btn.click(_save_ignore, inputs=[work_dir, ignore_table, video_state], outputs=[setup_log])
        validate_ignore_btn.click(_validate_ignore, inputs=[work_dir, ignore_table, video_state, config_state], outputs=[ignore_table, setup_log])
        import_ignore_csv_btn.click(_import_ignore_csv, inputs=[ignore_csv_file], outputs=[ignore_table, setup_log])
        export_ignore_csv_btn.click(_export_ignore_csv, inputs=[work_dir, ignore_table, export_ignore_csv_path], outputs=[setup_log])

        extract_btn.click(_run_extract, inputs=[work_dir, config_state, video_state, *param_inputs], outputs=[config_state, extract_log])
        analyze_btn.click(
            _analyze_dedup,
            inputs=[work_dir, config_state, *param_inputs],
            outputs=[config_state, dedup_state, group_dropdown, dedup_info, group_gallery, keep_choices, dedup_log],
        )
        load_dedup_btn.click(_load_dedup, inputs=[work_dir], outputs=[dedup_state, group_dropdown, dedup_info, group_gallery, keep_choices, dedup_log])
        save_dedup_btn.click(_save_dedup, inputs=[work_dir, dedup_state], outputs=[dedup_log])
        group_dropdown.change(_show_group, inputs=[work_dir, dedup_state, group_dropdown], outputs=[group_gallery, keep_choices])
        prev_group_btn.click(_move_group, inputs=[work_dir, dedup_state, group_dropdown, gr.State(-1)], outputs=[group_dropdown, group_gallery, keep_choices])
        next_group_btn.click(_move_group, inputs=[work_dir, dedup_state, group_dropdown, gr.State(1)], outputs=[group_dropdown, group_gallery, keep_choices])
        save_group_btn.click(_save_group, inputs=[work_dir, dedup_state, group_dropdown, keep_choices], outputs=[dedup_state, group_gallery, keep_choices, dedup_log])
        keep_selected_btn.click(_save_group, inputs=[work_dir, dedup_state, group_dropdown, keep_choices], outputs=[dedup_state, group_gallery, keep_choices, dedup_log])
        keep_all_btn.click(_keep_all, inputs=[work_dir, dedup_state, group_dropdown], outputs=[dedup_state, group_gallery, keep_choices, dedup_log])
        export_dedup_btn.click(_export_dedup, inputs=[work_dir, config_state, dedup_state, *param_inputs], outputs=[config_state, dedup_log])
        crop_btn.click(_run_crop_gui, inputs=[work_dir, config_state, *param_inputs], outputs=[config_state, crop_log])
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
        str(resolve_work_path(root, config["paths"]["frames_raw"])),
        str(resolve_work_path(root, config["paths"]["frames_dedup"])),
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
    videos = scan_videos(resolve_work_path(root, video_dir), root, config["project"]["supported_video_ext"])
    return videos_as_dicts(videos), videos_to_rows(videos), f"scanned {len(videos)} videos", "all"


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


def _run_extract(work_dir: str, config: dict[str, Any], videos: list[dict[str, Any]], *values: Any):
    updated = _apply_gui_values(config, values)
    root = Path(work_dir).expanduser().resolve()
    video_objs = [VideoInfo(**item) for item in videos]
    saved, log_path, messages = extract_frames_for_videos(root, updated, video_objs)
    return updated, "\n".join([*messages, f"saved total: {saved}", f"log: {log_path}"])


def _analyze_dedup(work_dir: str, config: dict[str, Any], *values: Any):
    updated = _apply_gui_values(config, values)
    root = Path(work_dir).expanduser().resolve()
    state, path = analyze_duplicates(root, updated, resolve_work_path(root, updated["dedup"]["input_dir"]))
    groups = [group["group_id"] for group in state.get("groups", [])]
    group_id = groups[0] if groups else None
    gallery, keep = group_gallery_items(root, state, group_id) if group_id else ([], [])
    info = f"groups: {len(groups)}\nunique: {len(state.get('unique_images', []))}\nstate: {path}"
    return updated, state, gr.update(choices=groups, value=group_id), info, gallery, gr.update(choices=keep or [], value=keep), f"analyzed {path}"


def _load_dedup(work_dir: str):
    root = Path(work_dir).expanduser().resolve()
    state = load_dedup_state(root)
    groups = [group["group_id"] for group in state.get("groups", [])]
    group_id = groups[0] if groups else None
    gallery, keep = group_gallery_items(root, state, group_id) if group_id else ([], [])
    info = f"groups: {len(groups)}\nunique: {len(state.get('unique_images', []))}"
    return state, gr.update(choices=groups, value=group_id), info, gallery, gr.update(choices=keep, value=keep), "loaded dedup_state.json"


def _save_dedup(work_dir: str, state: dict[str, Any]):
    path = save_dedup_state(Path(work_dir).expanduser().resolve(), state)
    return f"saved {path}"


def _show_group(work_dir: str, state: dict[str, Any], group_id: str):
    gallery, keep = group_gallery_items(Path(work_dir).expanduser().resolve(), state, group_id)
    choices = [item[1].split(" | ")[0] for item in gallery]
    path_choices = _group_paths(state, group_id)
    return gallery, gr.update(choices=path_choices, value=keep)


def _move_group(work_dir: str, state: dict[str, Any], current: str, delta: int):
    groups = [group["group_id"] for group in state.get("groups", [])]
    if not groups:
        return None, [], gr.update(choices=[], value=[])
    index = groups.index(current) if current in groups else 0
    next_id = groups[(index + delta) % len(groups)]
    gallery, keep = group_gallery_items(Path(work_dir).expanduser().resolve(), state, next_id)
    return next_id, gallery, gr.update(choices=_group_paths(state, next_id), value=keep)


def _save_group(work_dir: str, state: dict[str, Any], group_id: str, keep: list[str]):
    state = update_group_decision(state, group_id, keep or [])
    save_dedup_state(Path(work_dir).expanduser().resolve(), state)
    gallery, values = group_gallery_items(Path(work_dir).expanduser().resolve(), state, group_id)
    return state, gallery, gr.update(choices=_group_paths(state, group_id), value=values), f"saved decision for {group_id}"


def _keep_all(work_dir: str, state: dict[str, Any], group_id: str):
    state = keep_all_in_group(state, group_id)
    save_dedup_state(Path(work_dir).expanduser().resolve(), state)
    gallery, values = group_gallery_items(Path(work_dir).expanduser().resolve(), state, group_id)
    return state, gallery, gr.update(choices=_group_paths(state, group_id), value=values), f"kept all in {group_id}"


def _export_dedup(work_dir: str, config: dict[str, Any], state: dict[str, Any], *values: Any):
    updated = _apply_gui_values(config, values)
    kept, rejected, log_path = export_dedup_results(Path(work_dir).expanduser().resolve(), updated, state)
    return updated, f"kept: {kept}\nrejected copies: {rejected}\nlog: {log_path}"


def _run_crop_gui(work_dir: str, config: dict[str, Any], *values: Any):
    updated = _apply_gui_values(config, values)
    saved, log_path = run_crop(Path(work_dir).expanduser().resolve(), updated)
    return updated, f"saved crops: {saved}\nlog: {log_path}"


def _refresh_stats(work_dir: str, config: dict[str, Any]):
    root = Path(work_dir).expanduser().resolve()
    summary = summarize_project(root, config)
    paths = [
        f"work_dir: {root}",
        f"params: {root / 'configs' / 'params.yaml'}",
        f"ignore_ranges: {root / 'states' / 'ignore_ranges.json'}",
        f"dedup_state: {root / 'states' / 'dedup_state.json'}",
    ]
    logs = []
    for key in ["extract_log", "dedup_log", "crop_log"]:
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
        diff_threshold,
        max_gap,
        png_compression,
        min_width,
        crop_bottom,
        resize_width_for_diff,
        scene_diff_method,
        reset_diff_after_ignore,
        dedup_input,
        dedup_output,
        dedup_scope,
        episode_filter,
        hash_threshold,
        hash_size,
        hash_crop,
        hash_resize_width,
        num_workers,
        export_rejected,
        crop_input,
        crop_output,
        body_model_path,
        face_model_path,
        crop_mode,
        output_strategy,
        weight_full,
        weight_hard,
        weight_face,
        weight_body,
        weight_background,
        weight_random,
        face_aspect,
        body_aspect,
        background_aspect,
        random_seed,
        random_aspect_pool,
        conf,
        imgsz,
        body_class_id,
        face_class_id,
        face_all_classes,
        face_padding,
        body_padding_x,
        body_padding_y,
        background_exclusion_padding,
        background_max_overlap,
        min_crop_size,
        max_side,
        crop_png_compression,
        target_crops_per_image,
    ) = values
    root = Path(cfg["project"]["work_dir"]).expanduser().resolve()
    cfg["project"]["video_dir"] = relative_path_value(video_dir, root)
    cfg["paths"]["frames_raw"] = relative_path_value(extract_output, root)
    cfg["paths"]["frames_dedup"] = relative_path_value(dedup_output, root)
    cfg["extract"].update(
        {
            "interval": float(interval),
            "diff_threshold": float(diff_threshold),
            "max_gap": float(max_gap),
            "png_compression": int(png_compression),
            "min_width": int(min_width),
            "crop_bottom": int(crop_bottom),
            "resize_width_for_diff": int(resize_width_for_diff),
            "scene_diff_method": scene_diff_method,
            "reset_diff_after_ignore": bool(reset_diff_after_ignore),
        }
    )
    cfg["dedup"].update(
        {
            "input_dir": relative_path_value(dedup_input, root),
            "dedup_scope": dedup_scope,
            "episode_filter": episode_filter,
            "hash_threshold": int(hash_threshold),
            "hash_size": int(hash_size),
            "hash_crop": hash_crop,
            "hash_resize_width": int(hash_resize_width),
            "num_workers": int(num_workers),
            "export_rejected_duplicates": bool(export_rejected),
        }
    )
    cfg["crop"].update(
        {
            "input_dir": relative_path_value(crop_input, root),
            "output_dir": relative_path_value(crop_output, root),
            "output_strategy": output_strategy,
            "random_seed": int(random_seed),
            "min_crop_size": int(min_crop_size),
            "max_side": int(max_side),
            "png_compression": int(crop_png_compression),
            "target_crops_per_image": int(target_crops_per_image),
        }
    )
    cfg["crop_types"] = {key: key in (crop_mode or []) for key in ["full", "hard_split", "face", "body", "background", "random_crop"]}
    cfg["random_output_weights"].update(
        {
            "full": int(weight_full),
            "hard_split": int(weight_hard),
            "face": int(weight_face),
            "body": int(weight_body),
            "background": int(weight_background),
            "random_crop": int(weight_random),
        }
    )
    cfg["yolo"].update(
        {
            "body_model_path": body_model_path,
            "face_model_path": face_model_path,
            "conf": float(conf),
            "imgsz": int(imgsz),
            "body_class_id": int(body_class_id),
            "face_class_id": int(face_class_id),
            "face_all_classes": bool(face_all_classes),
        }
    )
    cfg["face_crop"]["padding"] = float(face_padding)
    cfg["face_crop"]["aspect_mode"] = face_aspect
    cfg["body_crop"]["padding_x"] = float(body_padding_x)
    cfg["body_crop"]["padding_y"] = float(body_padding_y)
    cfg["body_crop"]["aspect_mode"] = body_aspect
    cfg["background_crop"]["exclusion_padding"] = float(background_exclusion_padding)
    cfg["background_crop"]["max_overlap"] = float(background_max_overlap)
    cfg["background_crop"]["aspect_mode"] = background_aspect
    cfg["random_crop"]["aspect_pool"] = random_aspect_pool or ["1:1"]
    return cfg


def _values_from_config(config: dict[str, Any]) -> tuple[Any, ...]:
    enabled_types = [key for key, value in config["crop_types"].items() if value]
    return (
        config["project"].get("video_dir", "videos"),
        config["paths"]["frames_raw"],
        config["extract"]["interval"],
        config["extract"]["diff_threshold"],
        config["extract"]["max_gap"],
        config["extract"]["png_compression"],
        config["extract"]["min_width"],
        config["extract"]["crop_bottom"],
        config["extract"]["resize_width_for_diff"],
        config["extract"]["scene_diff_method"],
        config["extract"]["reset_diff_after_ignore"],
        config["dedup"].get("input_dir", config["paths"]["frames_raw"]),
        config["paths"]["frames_dedup"],
        config["dedup"]["dedup_scope"],
        config["dedup"]["episode_filter"],
        config["dedup"]["hash_threshold"],
        config["dedup"]["hash_size"],
        config["dedup"]["hash_crop"],
        config["dedup"]["hash_resize_width"],
        config["dedup"]["num_workers"],
        config["dedup"]["export_rejected_duplicates"],
        config["crop"]["input_dir"],
        config["crop"]["output_dir"],
        config["yolo"]["body_model_path"],
        config["yolo"]["face_model_path"],
        enabled_types,
        config["crop"]["output_strategy"],
        config["random_output_weights"]["full"],
        config["random_output_weights"]["hard_split"],
        config["random_output_weights"]["face"],
        config["random_output_weights"]["body"],
        config["random_output_weights"]["background"],
        config["random_output_weights"]["random_crop"],
        config["face_crop"]["aspect_mode"],
        config["body_crop"]["aspect_mode"],
        config["background_crop"]["aspect_mode"],
        config["crop"]["random_seed"],
        config["random_crop"]["aspect_pool"],
        config["yolo"]["conf"],
        config["yolo"]["imgsz"],
        config["yolo"]["body_class_id"],
        config["yolo"].get("face_class_id") or 0,
        config["yolo"]["face_all_classes"],
        config["face_crop"]["padding"],
        config["body_crop"]["padding_x"],
        config["body_crop"]["padding_y"],
        config["background_crop"]["exclusion_padding"],
        config["background_crop"]["max_overlap"],
        config["crop"]["min_crop_size"],
        config["crop"]["max_side"],
        config["crop"]["png_compression"],
        config["crop"]["target_crops_per_image"],
    )


def _group_paths(state: dict[str, Any], group_id: str) -> list[str]:
    for group in state.get("groups", []):
        if group.get("group_id") == group_id:
            return [item["path"] for item in group.get("images", [])]
    return []


def _deep_copy_config(config: dict[str, Any]) -> dict[str, Any]:
    import copy

    return copy.deepcopy(config)


def _aspect_options() -> list[str]:
    return list(ASPECT_OPTIONS)
