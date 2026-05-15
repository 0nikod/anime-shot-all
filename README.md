# Anime Shot All

基于 Gradio 的动漫训练图像数据采集、去重和裁剪工具。

## 启动

```bash
uv run anime-shot-all
```

默认地址：

```text
http://127.0.0.1:7860
```

如需指定端口：

```bash
uv run anime-shot-all --port 7861
```

## 基本流程

```text
选择工作目录 work_dir
→ 初始化 / 打开工作目录
→ 选择或填写 video_dir
→ 扫描视频
→ 配置并保存 ignore_ranges
→ 截帧
→ 分析重复并人工确认
→ 导出去重图片
→ 裁剪输出 PNG
```

默认情况下，项目状态和输出归档到源码根目录下的 `./work_dir/`，视频源目录使用源码根目录下的 `./video_dir/`，避免把运行态文件散落到项目根目录。也可以在 GUI 中改成任意外部路径。

## 输出结构

```text
work_dir/
  frames_raw/
  frames_dedup/
  rejected_duplicates/
  crops/
  models/
    yolo/
  configs/
    default.yaml
    params.yaml
  logs/
    extract_log.csv
    dedup_log.csv
    crop_log.csv
  states/
    ignore_ranges.json
    dedup_state.json

video_dir/
  *.mp4 / *.mkv / ...
```

## 当前边界

- `ignore_ranges` 只用于截帧阶段跳过时间段，不参与去重或裁剪分组。
- 不实现 OP / ED 自动识别、单独目录、单独去重或单独裁剪。
- 去重采用 pHash，算法只生成建议，最终结果以人工确认后的 `dedup_state.json` 为准。
- YOLO 裁剪兼容 Ultralytics `.pt` 权重。GUI 内置 `Bingsu/adetailer` 的 face/person 预设，可自动下载到 `work_dir/models/yolo/`；也可以填写本机 `.pt` 路径覆盖预设。
- 所有阶段都采用复制或生成新文件，不删除源视频或源图片。

## 验证

```bash
uv run pytest
uv run python -m compileall anime_shot_all tests
uv run python -c 'from anime_shot_all.gui import build_app; print(type(build_app()).__name__)'
```
