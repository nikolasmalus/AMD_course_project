# 本地监控视频异常事件分析系统项目设计初稿

## 1. 项目概述

本项目面向本地监控视频的异常事件分析场景，目标是在数据不出机的前提下，对用户上传的 MP4 视频进行人员检测、轨迹跟踪、规则识别、异常片段截取和中文告警生成。

系统采用 Python 实现，前端使用 Gradio，检测模型使用本地 YOLOv8n，告警文本由本地 Ollama 模型生成。项目整体设计强调本地部署、模块化、轻量化和 GPU 加速能力，便于后续从 NVIDIA WSL 开发环境迁移到 AMD ROCm GPU 工作站。

## 2. 设计目标

项目设计目标包括：

- 本地化处理：视频、抽帧图片、检测结果和告警生成均在本机完成，不调用外部云端 API。
- 端到端流程：覆盖视频上传、视频解析、抽帧、人员检测、轨迹跟踪、异常识别、片段截取、告警生成和报告输出。
- GPU 推理支持：人员检测通过 PyTorch/Ultralytics 统一设备接口运行，当前环境使用 CUDA，后续 AMD 环境使用 ROCm/HIP。
- 异构分工清晰：CPU 负责视频解码、抽帧、跟踪、规则判断、片段处理和 Web UI；GPU 负责 YOLO 推理；NPU 作为后续 Ryzen AI 适配方向。
- 模块化架构：各功能模块边界清晰，便于替换模型、调整规则或扩展新事件类型。
- 可靠兜底：Ollama 不可用时自动使用模板告警，保证主分析流程不中断。

## 3. 项目功能

系统主要功能包括：

- 上传本地 MP4 视频。
- 自动读取视频宽高、帧率、帧数和时长。
- 按配置的抽帧帧率保存关键帧。
- 使用本地 `models/yolov8n.pt` 检测人员，仅检测 COCO person 类。
- 基于 IoU 为人员检测框分配轨迹编号。
- 检测禁区闯入事件。
- 检测长时间徘徊事件。
- 自动截取异常视频片段。
- 自动生成事件缩略图。
- 调用本机 Ollama `qwen2.5:3b` 生成中文告警文本。
- Ollama 失败时使用模板告警。
- 输出 `result.json` 和 `benchmark_report.json`。
- 在 Gradio 页面展示视频、硬件信息、事件列表、告警文本、异常片段、缩略图和性能对比。

## 4. 总体架构

系统整体可分为五层：

```text
用户交互层
  └── app.py / Gradio 页面

流程调度层
  └── pipeline.py / video_security_agent.py

AI 推理与规则分析层
  ├── yolo_person_detector.py
  ├── simple_tracker.py
  ├── event_detector.py
  └── alert_generator.py

数据处理层
  ├── video_io.py
  ├── frame_sampler.py
  ├── clip_extractor.py
  └── media_compat.py

运行环境与配置层
  ├── config.py
  ├── hardware_manager.py
  ├── benchmark_profiler.py
  └── config.yaml
```

## 5. 处理流程

系统处理流程如下：

1. 用户通过 Gradio 页面上传 MP4，或通过命令行传入视频路径。
2. 系统保存上传视频到 `data/uploads/`。
3. 读取视频基础信息，包括分辨率、帧率、帧数和时长。
4. 按 `sample_fps` 抽帧，帧图片保存到 `data/frames/`。
5. 设备管理模块检测 PyTorch 加速设备，优先使用 `device="cuda"`。
6. YOLOv8n 对抽样帧进行人员检测。
7. IoU 跟踪模块将相邻帧人员检测结果关联成轨迹。
8. 事件模块基于轨迹检测禁区闯入和长时间徘徊。
9. 片段模块根据事件时间截取异常 MP4 片段，并生成缩略图。
10. 告警模块调用本地 Ollama 生成中文告警，失败时模板兜底。
11. Benchmark 模块对 FP32 和 FP16 推理进行性能对比。
12. 系统写出 `result.json` 和 `benchmark_report.json`。
13. Gradio 页面展示分析结果。

## 6. 核心模块说明

| 模块 | 作用 |
| --- | --- |
| `config.py` | 读取 `config.yaml`，解析路径和默认配置 |
| `hardware_manager.py` | 检测 CPU/GPU/NPU 状态，记录 PyTorch 设备信息 |
| `video_io.py` | 保存上传视频，读取视频基础信息 |
| `frame_sampler.py` | 按抽帧帧率保存视频帧 |
| `yolo_person_detector.py` | 加载本地 YOLO 模型并检测人员 |
| `simple_tracker.py` | 基于 IoU 生成人员轨迹 |
| `event_detector.py` | 检测禁区闯入和长时间徘徊 |
| `clip_extractor.py` | 截取异常片段并生成缩略图 |
| `media_compat.py` | 将片段转为浏览器兼容 MP4 |
| `local_llm_client.py` | 调用本机 Ollama API |
| `alert_generator.py` | 生成事件级中文告警，失败时模板兜底 |
| `benchmark_profiler.py` | 生成 FP32 与 FP16 性能对比 |
| `pipeline.py` | 串联完整分析流程 |
| `video_security_agent.py` | 以轻量 Agent 工具链形式组织流程 |
| `app.py` | Gradio 网页入口 |

## 7. 本地模型设计

项目包含两类本地 AI 能力：

1. 人员检测模型：`models/yolov8n.pt`
   - 用于检测监控画面中的人员。
   - 只使用 COCO person 类。
   - 通过 PyTorch/Ultralytics 统一设备接口运行。

2. 告警文本模型：Ollama `qwen2.5:3b`
   - 用于根据结构化事件生成一句中文告警。
   - 原始视频和图像帧不发送给 Ollama。
   - 发送内容仅包括事件类型、风险等级、时间范围、持续时长和轨迹编号。
   - API 地址限定为 `127.0.0.1`、`localhost` 或 `::1`。

## 8. 异常事件设计

当前规则主要包括：

- 禁区闯入：当某条人员轨迹的中心点进入配置的禁区多边形，生成禁区闯入事件。
- 长时间徘徊：当某条轨迹持续时间超过阈值，并且移动范围较小，生成徘徊事件。

禁区区域在 `config.yaml` 中使用归一化多边形配置，例如：

```yaml
polygon:
  - [0.62, 0.35]
  - [0.98, 0.35]
  - [0.98, 0.98]
  - [0.62, 0.98]
```

归一化坐标会根据视频实际宽高换算为像素坐标。后续接入真实监控视频时，可以根据实际画面重新调整该区域。

## 9. Ollama 告警设计

Ollama 配置如下：

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

告警生成流程：

1. 事件检测模块输出结构化事件。
2. `AlertGenerator` 为每个事件构造短提示词。
3. `LocalLLMClient` 调用本机 Ollama，并设置 `stream=false`。
4. 成功时将返回文本写入事件的 `alert_text`。
5. 失败时使用模板告警，并记录失败原因。

`result.json` 会记录事件级和总体 LLM 信息，便于说明是否使用了本地模型，以及是否发生模板兜底。

## 10. 性能对比设计

系统提供原始模型与本地优化模型的性能代理对比：

- `baseline_fp32`：YOLOv8n FP32 推理。
- `optimized_fp16`：YOLOv8n FP16 推理，仅 GPU 可用时启用。

统计指标包括：

- 平均延迟
- 吞吐率
- 人员检测数量
- 事件数量
- 检测数量一致率
- 事件一致率
- 显存峰值

## 11. 输出结果

分析结果主要输出到：

```text
data/uploads/   上传视频副本
data/frames/    抽样帧
data/clips/     异常片段和缩略图
data/results/   result.json 和 benchmark_report.json
```

`result.json` 包含：

- 视频信息
- 硬件信息
- 模型错误提示
- 轨迹数量
- 异常事件列表
- 事件级告警文本
- 总体 LLM 使用情况
- 风险等级
- 性能对比
- NPU 后续适配说明

## 12. 后续适配方向

后续可根据真实监控视频继续完善：

- 按实际画面调整禁区 polygon。
- 增加更多事件规则，例如越线、逆行、聚集等。
- 将检测模型导出 ONNX，为 Ryzen AI/NPU 适配做准备。
- 在 AMD ROCm 环境中重新验证 GPU 推理和 FP16 benchmark。
- 根据比赛演示视频优化页面展示和报告格式。
