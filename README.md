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
→ 截帧（内置 pHash 分组随机保留）
→ 裁剪输出 PNG
```

默认情况下，项目状态和输出归档到源码根目录下的 `./work_dir/`，视频源目录使用源码根目录下的 `./video_dir/`，避免把运行态文件散落到项目根目录。也可以在 GUI 中改成任意外部路径。

## 输出结构

```text
work_dir/
  frames_raw/
  crops/
  configs/
    default.yaml
    params.yaml
  logs/
    extract_log.csv
    crop_log.csv
  states/
    ignore_ranges.json

video_dir/
  *.mp4 / *.mkv / ...
```

## 当前边界

- `ignore_ranges` 只用于截帧阶段跳过时间段，不参与裁剪分组。
- 不实现 OP / ED 自动识别、单独目录、单独去重或单独裁剪。
- 截帧支持关键帧模式（此时忽略 interval / max_gap / 去重分组）。
- 语义裁剪使用 `dghs-imgutils` 的 face / person / halfbody 检测，不再由本工具直接管理检测模型权重。
- 裁剪采用 bbox-first 流程：`full` 保留完整画面并按面积缩放，`face` / `body` / `halfbody` / `random_crop` 先生成 bbox，再按 bbox 比例加权随机选择输出比例和内置尺寸预设。
- 非 `full` 裁剪输出面积固定限制在 `1024²` 到 `1536²` 之间，输出尺寸来自程序内置预设表；`1:1` 对横竖 bbox 都允许，但距离 bbox 比例越远权重越低。
- 所有阶段都采用复制或生成新文件，不删除源视频或源图片。

## 验证

```bash
uv run pytest
```
