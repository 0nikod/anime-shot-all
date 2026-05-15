# Anime Shot All

基于 Gradio 的动漫训练图像数据采集、去重和裁剪工具。

## 启动

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run anime-shot-all
```

默认地址：

```text
http://127.0.0.1:7860
```

如需指定端口：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run anime-shot-all --port 7861
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

所有项目相关配置、状态、日志和输出默认归档在 `work_dir` 下。视频源可以放在 `work_dir/videos/`，也可以通过 `video_dir` 指向外部目录。

## 输出结构

```text
work_dir/
  videos/
  frames_raw/
  frames_dedup/
  rejected_duplicates/
  crops/
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
```

## 当前边界

- `ignore_ranges` 只用于截帧阶段跳过时间段，不参与去重或裁剪分组。
- 不实现 OP / ED 自动识别、单独目录、单独去重或单独裁剪。
- 去重采用 pHash，算法只生成建议，最终结果以人工确认后的 `dedup_state.json` 为准。
- YOLO 裁剪兼容 Ultralytics `.pt` 权重，但不内置模型文件，需要在 GUI 中填写本机模型路径。
- 所有阶段都采用复制或生成新文件，不删除源视频或源图片。

## 验证

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall anime_shot_all tests
UV_CACHE_DIR=/tmp/uv-cache uv run python -c 'from anime_shot_all.gui import build_app; print(type(build_app()).__name__)'
```
