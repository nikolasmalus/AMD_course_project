# 本地监控视频异常事件分析系统

## 6月15日更新摘要

- 精简硬件信息展示：网页端和报告中仅保留实际推理设备、PyTorch 版本、设备名称、CPU 线程数、系统内存。
- 取消本地大模型调用：移除 Ollama/qwen2.5:3b 客户端和测试脚本，告警说明统一由本地模板生成。
- 精简模型档位：取消 `YOLOv8s` 中间档，仅保留 `YOLOv8n` 快速模式和 `YOLO11m` 增强模式。
- 清理无用依赖和配置：移除 LLM 配置、`requests` 依赖、YOLOv8s 下载入口和相关命令行选项。
- 优化结果区布局：异常事件表移动到右侧信息区，删除轨迹编号列、独立风险/告警文本区，性能表删除一致率行和人员检测数量列。
- 统一事件命名：事件编号改为 `restricted_zone_event_1`、`loitering_event_1` 这类事件序号，不再以轨迹编号命名。
- 优化警戒区域触发：由“检测框中心点进入”改为“检测框少量关键点进入”即可触发，默认阈值为 `min_bbox_points_inside_ratio: 0.10`。

## 6月9日更新摘要

- 新增交互式警戒区域：上传视频后显示首帧，用户可点选多边形区域，并可撤销、清空、自动规整优化。
- 新增分析视频可视化：输出视频会叠加警戒区域，普通人员显示绿框，进入警戒区域后显示红框和 `ALERT ID`。
- 新增多模型模式：支持 `YOLOv8n` 快速模式、`YOLO11m` 增强模式，适配夜间、小目标等困难监控画面。
- 完善端侧运行记录：报告中记录本次警戒区域、标注视频路径、模型档位、模型路径和硬件加速信息。
- 保持 AMD 迁移友好：业务代码继续使用 PyTorch 统一设备接口，当前 NVIDIA CUDA 可运行，迁移到 AMD ROCm 后重建 PyTorch 环境即可。

`security_video_analyzer_py` 是一个本地端到端安防视频分析项目。用户上传 MP4 后，系统会读取视频信息、抽帧、调用当前选择的 YOLO 模型检测人员、基于 IoU 生成轨迹、识别禁区闯入和长时间徘徊，自动截取异常片段和缩略图，并通过本地模板生成中文告警说明。Gradio 页面会展示分析视频、硬件信息、事件表格、风险等级、告警文本、异常片段、缩略图、性能对比和报告文件。

## 当前完成状态

- 已实现网页端：`app.py`
- 已实现命令行分析：`scripts/run_demo.py`
- 已实现环境检查：`scripts/check_env.py`
- 已实现本地模型准备：`scripts/download_model.py`
- 已实现轻量 Agent：`VideoSecurityAgent`
- 已实现 YOLO FP32 与 FP16 benchmark
- 已实现 GPU 优先、CPU fallback 设备选择
- 已实现模板化中文告警生成
- 已实现 AMD ROCm 迁移说明

## 文档初稿

比赛报告的项目设计初稿位于：

```text
docs/project_design_draft.md
```

该文档主要罗列项目设计目标、功能模块、总体框架、处理流程、本地模型方案、告警设计和后续适配方向，可作为后续正式报告的基础材料。

## 项目结构

```text
security_video_analyzer_py/
├── app.py
├── config.yaml
├── requirements.txt
├── README.md
├── models/
│   ├── yolov8n.pt
│   └── yolo11m.pt
├── data/
│   ├── uploads/
│   ├── frames/
│   ├── clips/
│   ├── results/
│   └── eval/
├── scripts/
│   ├── check_env.py
│   ├── download_model.py
│   └── run_demo.py
├── src/
│   ├── alert_generator.py
│   ├── benchmark_profiler.py
│   ├── clip_extractor.py
│   ├── config.py
│   ├── event_detector.py
│   ├── frame_sampler.py
│   ├── hardware_manager.py
│   ├── media_compat.py
│   ├── pipeline.py
│   ├── simple_tracker.py
│   ├── video_io.py
│   ├── video_security_agent.py
│   ├── visualizer.py
│   └── yolo_person_detector.py
└── tests/
    ├── test_event_detector.py
    └── test_tracker.py
```

## 核心流程

1. `video_io.py` 保存上传 MP4，并读取分辨率、帧率、帧数和时长。
2. `frame_sampler.py` 按 `sample_fps` 抽帧，保存到 `data/frames/`。
3. `hardware_manager.py` 用 `torch.cuda.is_available()` 选择设备；可用则使用 `device="cuda"`，否则使用 CPU fallback。
4. `yolo_person_detector.py` 加载当前选择的 YOLOv8 模型，只检测 COCO person 类，即 `classes=[0]`。
5. `simple_tracker.py` 用 IoU 为检测框分配 `track_id`，形成简单人员轨迹。
6. `event_detector.py` 基于轨迹中心点判断禁区闯入，基于持续时间和移动半径判断长时间徘徊。
7. `clip_extractor.py` 截取异常片段，生成缩略图，并调用 `media_compat.py` 转成浏览器友好的 H.264 MP4。
8. `alert_generator.py` 根据事件类型、时间和风险等级生成模板中文告警。
9. `benchmark_profiler.py` 对同一批抽样帧运行 FP32 与 FP16 推理，生成性能和一致率对比。
10. `pipeline.py` 串起完整流程，`video_security_agent.py` 记录工具链执行顺序。

## 模型说明

当前 `models/` 目录中实际需要的检测模型是：

```text
models/yolov8n.pt
models/yolo11m.pt
```

网页端提供两种人员检测模式：

- `快速模式 YOLOv8n`：速度快、资源占用低，适合快速预览。
- `增强模式 YOLO11m`：模型更强，适合困难监控画面和赛题演示，但速度和显存占用更高。

性能对比在当前选择的 YOLOv8 权重上运行两种推理方式：

- `baseline_fp32`：FP32 推理，`half=False`
- `optimized_fp16`：FP16 GPU 推理，`half=True`，仅在 `device="cuda"` 时启用

告警文本不再调用本地大语言模型，统一由 `alert_generator.py` 的模板规则生成。

## 环境配置

推荐 Python 3.10-3.12。当前 WSL + NVIDIA 开发环境建议先单独安装 PyTorch CUDA 版，再安装项目其余依赖，避免普通 `pip install -r requirements.txt` 把 CUDA 版 PyTorch 覆盖成 CPU 版。

进入项目目录：

```bash
cd ~/note_wsl/homework/project/security_video_analyzer_py
```

重建虚拟环境：

```bash
deactivate 2>/dev/null
rm -rf .venv

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
```

先安装 PyTorch CUDA 版。WSL + NVIDIA 推荐使用 CUDA 12.1 wheel：

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --default-timeout=1000
```

安装后检查 GPU 是否可用：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

如果这里能输出 `True` 和你的 NVIDIA 显卡名，说明 PyTorch CUDA 环境已经装对。

`requirements.txt` 不应再包含 `torch` 或 `torchvision`，避免后续安装依赖时覆盖 CUDA 版 PyTorch。当前依赖文件应保持为：

```text
ultralytics
opencv-python
gradio
numpy
pyyaml
moviepy
imageio-ffmpeg
psutil
pytest
```

接着安装剩余依赖。为了避免 `ultralytics` 再触发 PyTorch 依赖升级，建议先加 `--no-deps` 单独安装它，然后安装其他包：

```bash
python -m pip install ultralytics --no-deps
python -m pip install opencv-python gradio numpy pyyaml moviepy imageio-ffmpeg psutil pytest matplotlib pillow scipy pandas tqdm pydantic fastapi uvicorn --default-timeout=1000
python -m pip install polars ultralytics-thop --default-timeout=1000
```

最后检查项目关键依赖：

```bash
python -c "import torch, gradio, ultralytics, cv2; print('env ok'); print(torch.cuda.is_available())"
```

如果输出：

```text
env ok
True
```

说明环境配置完成。项目业务代码不依赖 `nvidia-smi`、`CUDA_HOME` 或自定义 CUDA kernel。

## 准备 YOLO 模型

开发阶段可以在 `config.yaml` 中设置：

```yaml
model:
  allow_auto_download: true
```

然后运行：

```bash
python scripts/download_model.py
```

最终演示建议改成：

```yaml
model:
  allow_auto_download: false
```

并确认本地模型存在：

```bash
ls models/yolov8n.pt
```

## 检查是否使用 GPU

运行：

```bash
python scripts/check_env.py
```

如果当前机器能使用加速设备，应看到：

```text
actual_device: cuda
device_name: NVIDIA GeForce GTX 1660 Ti
```

注意：AMD ROCm 版 PyTorch 也使用 `torch.cuda` 这个接口名，底层实际是 ROCm/HIP。因此在 AMD GPU 主机上看到 `actual_device="cuda"` 是正常现象，不代表代码写死 NVIDIA。

如果只能使用 CPU，`actual_device` 会显示为 `cpu`，`device_name` 会显示为 `CPU`。

## 启动网页端

```bash
python app.py
```

默认地址：

```text
http://127.0.0.1:7860
```

网页端支持设置：

- 抽帧帧率
- 检测置信度阈值
- 徘徊最短持续时间
- 徘徊最大移动比例

网页端展示：

- 原始视频
- 视频基本信息
- 硬件信息
- 异常事件列表
- 风险等级
- 中文告警说明
- 异常视频片段
- 缩略图
- FP32 与 FP16 性能对比
- 报告文件和报告内容

## 命令行运行

```bash
python scripts/run_demo.py /path/to/your_monitor_video.mp4 --sample-fps 2 --conf 0.25
python scripts/run_demo.py /path/to/your_monitor_video.mp4 --sample-fps 5 --conf 0.2 --model-profile enhanced
```

输出目录：

```text
data/uploads/
data/frames/
data/clips/
data/results/*/result.json
data/results/*/benchmark_report.json
```

## 警戒区域是怎么画的

警戒区域配置在 `config.yaml`：

```yaml
events:
  restricted_zone:
    enabled: true
    min_bbox_points_inside_ratio: 0.10
    polygon:
      - [0.62, 0.35]
      - [0.98, 0.35]
      - [0.98, 0.98]
      - [0.62, 0.98]
```

这里的点是归一化坐标，不是像素坐标：

- `x=0.62` 表示画面宽度的 62%
- `y=0.35` 表示画面高度的 35%
- `[0.62, 0.35]` 是警戒区域左上角
- `[0.98, 0.98]` 接近右下角

程序运行时会根据视频实际宽高把这些点换算成像素坐标。事件检测逻辑在 `event_detector.py` 中：当人员检测框有少量关键点进入这个多边形，就生成禁区闯入事件并在标注视频中显示红框。

`min_bbox_points_inside_ratio` 控制触发灵敏度，默认 `0.10`。数值越低，人员框刚接触警戒区时越容易变红；数值越高，越要求人员框更深入警戒区。

如果后续你提供实际监控视频，可以按视频画面重新调整这组 polygon。真正判定逻辑使用 `config.yaml` 里的坐标；可视化画框可以在测试阶段单独叠加，便于确认区域是否对准。

## 模板告警

告警生成流程：

1. 规则模块先识别出禁区闯入、长时间徘徊等结构化事件。
2. `AlertGenerator` 根据事件类型、风险等级、开始时间、结束时间和持续时长拼接中文告警。
3. 事件级 `alert_text` 和汇总 `alert_text` 会写入 `result.json`，网页端直接展示这些模板文本。

## Benchmark 说明

系统会对同一批抽样帧分别运行：

- `baseline_fp32`：FP32 推理
- `optimized_fp16`：FP16 推理，仅 GPU 可用时运行

报告字段包括：

- `avg_latency_ms`
- `throughput_fps`
- `person_detections`
- `event_count`
- `detection_count_consistency`
- `event_consistency`
- `peak_memory_mb`

如果 GPU 不可用，`optimized_fp16` 会跳过，并写入：

```text
optimized_model_available=false
optimized_model_error="FP16 optimization requires GPU"
```

## AMD ROCm 迁移

拷贝项目代码到 AMD GPU 主机后，不建议直接拷贝虚拟环境。应在 AMD 主机重新创建环境，并安装 ROCm 版 PyTorch。

迁移步骤：

```bash
cd security_video_analyzer_py
python -m venv .venv
source .venv/bin/activate
# 先按 AMD/ROCm 对应版本安装 ROCm 版 PyTorch
# 再参考“环境配置”部分安装除 torch/torchvision 外的其余依赖
python scripts/check_env.py
```

业务代码通过 PyTorch 统一设备接口迁移：

- 当前 NVIDIA WSL：`torch.cuda` 底层是 NVIDIA CUDA
- 最终 AMD 工作站：`torch.cuda` 底层是 ROCm/HIP

## CPU/GPU 分工

- CPU：视频解码、抽帧、IoU 跟踪、事件规则判断、片段截取、Web UI
- GPU：YOLO 人员检测推理，FP32/FP16 benchmark

## 测试

```bash
pytest
```

没有安装 `pytest` 时，请先按“环境配置”部分完成虚拟环境和项目依赖安装。

## 最终演示前检查清单

1. `models/yolov8n.pt` 已存在。
2. `config.yaml` 中 `allow_auto_download=false`。
3. `python scripts/check_env.py` 显示 `actual_device=cuda` 或目标机器对应的 PyTorch 加速设备。
4. 用实际监控 MP4 跑通网页端或命令行。
5. `data/results/*/result.json` 中有硬件信息、事件、告警和 benchmark。
