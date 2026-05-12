# 本地监控视频异常事件分析系统

`security_video_analyzer_py` 是一个本地端到端安防视频分析项目。用户上传 MP4 后，系统会读取视频信息、抽帧、调用本地 YOLOv8n 检测人员、基于 IoU 生成轨迹、识别禁区闯入和长时间徘徊，自动截取异常片段和缩略图，并通过本地 Ollama 或模板生成中文告警说明。Gradio 页面会展示原始视频、硬件信息、事件表格、风险等级、告警文本、异常片段、缩略图、性能对比和报告文件。

## 当前完成状态

- 已实现网页端：`app.py`
- 已实现命令行分析：`scripts/run_demo.py`
- 已实现环境检查：`scripts/check_env.py`
- 已实现本地模型准备：`scripts/download_model.py`
- 已实现轻量 Agent：`VideoSecurityAgent`
- 已实现 YOLO FP32 与 FP16 benchmark
- 已实现 GPU 优先、CPU fallback 设备选择
- 已实现本地 Ollama 调用与模板兜底
- 已实现 AMD ROCm 迁移说明和 NPU 后续适配说明

## 文档初稿

比赛报告的项目设计初稿位于：

```text
docs/project_design_draft.md
```

该文档主要罗列项目设计目标、功能模块、总体框架、处理流程、本地模型方案、Ollama 告警设计和后续适配方向，可作为后续正式报告的基础材料。

## 项目结构

```text
security_video_analyzer_py/
├── app.py
├── config.yaml
├── requirements.txt
├── README.md
├── models/
│   └── yolov8n.pt
├── data/
│   ├── uploads/
│   ├── frames/
│   ├── clips/
│   ├── results/
│   └── eval/
├── scripts/
│   ├── check_env.py
│   ├── download_model.py
│   ├── test_ollama.py
│   └── run_demo.py
├── src/
│   ├── alert_generator.py
│   ├── benchmark_profiler.py
│   ├── clip_extractor.py
│   ├── config.py
│   ├── event_detector.py
│   ├── frame_sampler.py
│   ├── hardware_manager.py
│   ├── local_llm_client.py
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
4. `yolo_person_detector.py` 加载 `models/yolov8n.pt`，只检测 COCO person 类，即 `classes=[0]`。
5. `simple_tracker.py` 用 IoU 为检测框分配 `track_id`，形成简单人员轨迹。
6. `event_detector.py` 基于轨迹中心点判断禁区闯入，基于持续时间和移动半径判断长时间徘徊。
7. `clip_extractor.py` 截取异常片段，生成缩略图，并调用 `media_compat.py` 转成浏览器友好的 H.264 MP4。
8. `alert_generator.py` 优先请求本机 Ollama 生成中文告警；失败时自动使用模板告警。
9. `benchmark_profiler.py` 对同一批抽样帧运行 FP32 与 FP16 推理，生成性能和一致率对比。
10. `pipeline.py` 串起完整流程，`video_security_agent.py` 记录工具链执行顺序。

## 模型说明

当前 `models/` 目录中实际需要的检测模型是：

```text
models/yolov8n.pt
```

提示词中的“原始模型 vs 本地优化模型”在本项目中实现为同一个 YOLOv8n 权重的两种推理方式：

- `baseline_fp32`：YOLOv8n FP32 推理，`half=False`
- `optimized_fp16`：YOLOv8n FP16 GPU 推理，`half=True`，仅在 `device="cuda"` 时启用

本地大语言模型是 Ollama 中的模型，默认使用 `qwen2.5:3b`。它不放在 `models/` 目录里，由 Ollama 管理。

## 安装

推荐 Python 3.10-3.12。

```bash
cd /home/malus/note_wsl/homework/project/security_video_analyzer_py
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果是 NVIDIA WSL 开发环境，请确保安装的是匹配驱动的 CUDA 版 PyTorch。项目业务代码不依赖 `nvidia-smi`、`CUDA_HOME` 或自定义 CUDA kernel。

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
uses_accelerator: true
torch_cuda_available: true
```

注意：AMD ROCm 版 PyTorch 也使用 `torch.cuda` 这个接口名，底层实际是 ROCm/HIP。因此在 AMD GPU 主机上看到 `actual_device="cuda"` 是正常现象，不代表代码写死 NVIDIA。

如果只能使用 CPU，页面和 `result.json` 会提示：

```text
CPU fallback only, not valid for final AMD GPU requirement
```

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

程序运行时会根据视频实际宽高把这些点换算成像素坐标。事件检测逻辑在 `event_detector.py` 中：当某个轨迹的人员框中心点落入这个多边形，就生成禁区闯入事件。

如果后续你提供实际监控视频，可以按视频画面重新调整这组 polygon。真正判定逻辑使用 `config.yaml` 里的坐标；可视化画框可以在测试阶段单独叠加，便于确认区域是否对准。

## 本地 Ollama 告警

默认配置：

```yaml
llm:
  enabled: true
  provider: "ollama"
  api_url: "http://127.0.0.1:11434/api/generate"
  model_name: "qwen2.5:3b"
  timeout_seconds: 20
  allow_external_api: false
  fallback_to_template: true
```

安装 Ollama 后，下载模型：

```bash
ollama pull qwen2.5:3b
```

启动服务：

```bash
ollama serve
```

测试 API：

```bash
curl http://127.0.0.1:11434/api/generate -d '{
  "model": "qwen2.5:3b",
  "prompt": "请生成一句监控告警：有人进入禁区，持续3.5秒，风险等级高。",
  "stream": false
}'
```

也可以运行项目内测试脚本：

```bash
python scripts/test_ollama.py
```

Ollama 只用于生成告警文本。原始视频不会发送给 Ollama，发送给 Ollama 的只有事件类型、风险等级、开始时间、结束时间、持续时长、轨迹编号等结构化事件信息。程序只允许请求 `127.0.0.1`、`localhost` 或 `::1`，不调用云端 API，数据不出机。

告警生成流程：

1. 规则模块先识别出禁区闯入、长时间徘徊等结构化事件。
2. `AlertGenerator` 为每个事件构造短 prompt，只包含事件字段，不包含原始视频、图像帧或个人身份信息。
3. `LocalLLMClient` 调用本机 Ollama API，并固定设置 `stream=false`。
4. 如果 Ollama 返回有效文本，则写入事件级 `alert_text`。
5. 如果 Ollama 不可用或返回异常，则使用模板告警，主流程继续运行。

如果 Ollama 未安装、未启动、模型未下载、请求超时、接口异常或返回内容为空，系统不会崩溃，会自动使用模板告警，并在每个事件和 `result.json` 中记录：

- `llm_used`
- `llm_fallback_reason`

`result.json` 还会保留总体 LLM 信息：

```json
{
  "llm": {
    "enabled": true,
    "provider": "ollama",
    "model_name": "qwen2.5:3b",
    "local_only": true,
    "llm_used": true,
    "fallback_count": 0,
    "fallback_reasons": []
  }
}
```

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
pip install -r requirements.txt
python scripts/check_env.py
```

业务代码通过 PyTorch 统一设备接口迁移：

- 当前 NVIDIA WSL：`torch.cuda` 底层是 NVIDIA CUDA
- 最终 AMD 工作站：`torch.cuda` 底层是 ROCm/HIP

## CPU/GPU/NPU 分工

- CPU：视频解码、抽帧、IoU 跟踪、事件规则判断、片段截取、Web UI
- GPU：YOLO 人员检测推理，FP32/FP16 benchmark
- NPU：MVP 不实际使用，只在 README 和 `result.json` 中记录后续 Ryzen AI ONNX/Vitis AI 适配方向

## 测试

```bash
pytest
```

没有安装 `pytest` 时，可以先安装依赖：

```bash
pip install -r requirements.txt
```

## 最终演示前检查清单

1. `models/yolov8n.pt` 已存在。
2. `config.yaml` 中 `allow_auto_download=false`。
3. `python scripts/check_env.py` 显示 `uses_accelerator=true`。
4. `python scripts/test_ollama.py` 能连通 Ollama，或确认允许模板告警兜底。
5. 用实际监控 MP4 跑通网页端或命令行。
6. `data/results/*/result.json` 中有硬件信息、事件、告警、benchmark 和 NPU 后续适配说明。
