请帮我创建一个全新的 Python 项目，项目名称为 security_video_analyzer_py。

项目目标：
实现一个本地监控视频异常事件分析系统。用户上传本地 MP4 视频后，系统自动读取视频信息、抽帧、使用本地 YOLO 模型检测人员、基于 IoU 生成人员轨迹、识别禁区闯入和长时间徘徊事件、自动截取异常视频片段、生成缩略图、调用本地 Ollama 或模板生成中文告警说明，并通过 Gradio 前端展示视频、事件列表、风险等级、告警文本、异常片段、缩略图和性能对比结果。

重要背景：
当前开发环境是 WSL + NVIDIA GPU，最终目标环境是 AMD Ryzen AI Max+ 工作站的 AMD GPU。项目必须使用 PyTorch/Ultralytics 的统一设备接口，避免写死 NVIDIA 专有逻辑。当前 NVIDIA 环境使用 PyTorch CUDA，后续 AMD 环境安装 PyTorch ROCm，同一套业务代码尽量保持不变。

必须满足的赛题要求：
1. 至少使用一个本地部署 AI 模型，数据不出机。人员检测使用本地 models/yolov8n.pt；告警文本优先调用本机 Ollama，地址只能是 127.0.0.1 或 localhost。
2. 体现端到端应用价值：上传 MP4、抽帧、人员检测、轨迹跟踪、异常事件识别、异常片段截取、告警生成、页面展示、报告输出。
3. 推理任务必须在 GPU 或 NPU 至少一个硬件上完成。YOLO 人员检测必须支持 GPU 推理。CPU 只能作为开发 fallback。
4. 合理利用 CPU + GPU + NPU 异构资源与统一内存。CPU 负责视频解码、抽帧、IoU 跟踪、事件规则判断、片段截取、Web UI；GPU 负责 YOLO 人员检测推理；NPU 在 MVP 中作为可选后端记录，不强制实现，但 README 和 result.json 中要说明后续 Ryzen AI 适配方向。
5. 构建轻量 AI Agent：实现 VideoSecurityAgent，调度 AnalyzeVideoTool、DetectPersonsTool、TrackPersonsTool、DetectEventsTool、GenerateClipsTool、GenerateAlertTool、GenerateReportTool。
6. 提供原始模型 vs 本地优化模型的性能和精度代理对比。baseline_fp32 使用 YOLOv8n FP32 推理，optimized_fp16 使用 YOLOv8n FP16 GPU 推理。输出延迟、吞吐、检测数量、事件数量、检测数量一致率、事件一致率、资源占用。

技术要求：
- Python 3.10-3.12
- PyTorch
- Ultralytics YOLO
- OpenCV
- Gradio
- NumPy
- PyYAML
- Requests
- MoviePy 或 OpenCV VideoWriter
- psutil
- 不写自定义 CUDA kernel
- 不使用 DirectML 作为主线
- 不依赖 nvidia-smi 作为程序运行前提
- 不依赖 CUDA_HOME 作为程序运行前提
- 不做人脸识别
- 不接实时摄像头，只处理上传的本地 MP4

设备选择要求：
1. 使用 torch.cuda.is_available() 判断是否有可用加速设备。
2. 如果可用，device="cuda"。
3. 如果不可用，device="cpu"。
4. 注意：在 AMD ROCm 版 PyTorch 中，也使用 torch.cuda 接口名，底层实际是 ROCm/HIP，不要因为名字是 cuda 就写成 NVIDIA 专用。
5. result.json 中记录：
   - actual_device
   - uses_accelerator
   - torch_version
   - torch_cuda_available
   - device_name
   - backend_note="PyTorch CUDA interface; NVIDIA CUDA in current WSL, AMD ROCm/HIP on final AMD workstation"
6. 如果 actual_device="cpu"，页面、日志、result.json 中必须提示：
   "CPU fallback only, not valid for final AMD GPU requirement"

模型要求：
1. 默认使用本地 models/yolov8n.pt。
2. 开发阶段可以设置 allow_auto_download=true，允许 Ultralytics 自动下载。
3. 最终演示要求 allow_auto_download=false，并要求模型已经在 models/yolov8n.pt。
4. 程序启动时检查模型是否存在。
5. 如果模型不存在且不允许自动下载，不要崩溃，要给出清晰提示。
6. 只检测 COCO person 类，classes=[0]。

核心模块：
1. config.py：读取 config.yaml。
2. video_io.py：保存上传 MP4，读取视频信息。
3. frame_sampler.py：按 sample_fps 抽帧，保存帧和时间戳。
4. hardware_manager.py：检测 PyTorch 设备，记录 CPU/GPU/NPU 状态。
5. yolo_person_detector.py：加载本地 YOLO 模型，检测 person，支持 FP32 和 FP16。
6. simple_tracker.py：基于 IoU 的人员轨迹跟踪。
7. event_detector.py：检测禁区闯入和长时间徘徊。
8. clip_extractor.py：截取异常视频片段和缩略图。
9. local_llm_client.py：只调用 localhost Ollama，不允许外部 API。
10. alert_generator.py：本地 LLM 生成一句中文告警，失败时模板兜底。
11. benchmark_profiler.py：FP32 vs FP16 性能对比。
12. visualizer.py：绘制检测框、track_id、禁区、轨迹、事件缩略图。
13. video_security_agent.py：轻量 Agent 工具链。
14. pipeline.py：完整流程调度。
15. app.py：Gradio 前端。

项目目录结构：
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
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── video_io.py
│   ├── frame_sampler.py
│   ├── yolo_person_detector.py
│   ├── simple_tracker.py
│   ├── event_detector.py
│   ├── clip_extractor.py
│   ├── alert_generator.py
│   ├── local_llm_client.py
│   ├── hardware_manager.py
│   ├── benchmark_profiler.py
│   ├── visualizer.py
│   ├── video_security_agent.py
│   └── pipeline.py
├── tests/
│   ├── test_event_detector.py
│   └── test_tracker.py
└── scripts/
    ├── check_env.py
    ├── download_model.py
    └── run_demo.py

Benchmark 要求：
1. 对同一批抽样帧分别运行：
   - baseline_fp32：half=False
   - optimized_fp16：half=True，仅在 device="cuda" 时启用
2. 统计：
   - avg_latency_ms
   - throughput_fps
   - person_detections
   - event_count
   - detection_count_consistency
   - event_consistency
   - peak_memory_mb，如无法准确获取可写 null，并说明原因
3. 如果 GPU 不可用：
   - baseline_fp32 用 CPU 跑通
   - optimized_fp16 跳过
   - benchmark_report.json 写入 optimized_model_available=false 和 optimized_model_error="FP16 optimization requires GPU"

本地 LLM 要求：
1. 默认使用 Ollama：
   http://127.0.0.1:11434/api/generate
2. 请求中设置 stream=false。
3. 只允许访问 127.0.0.1 或 localhost。
4. 如果配置为外部 URL，必须拒绝调用并回退模板告警。
5. 如果 Ollama 未安装、未启动、模型未下载、请求超时或返回异常，程序不能崩溃，AlertGenerator 自动回退模板告警。
6. result.json 中记录 llm_used 和 llm_fallback_reason。

Gradio 页面要求：
1. 支持上传 MP4。
2. 支持设置：
   - sample_fps
   - conf_threshold
   - loitering_min_duration
   - loitering_max_movement
3. 展示：
   - 原始视频
   - 视频基本信息
   - 硬件信息 actual_device / device_name / uses_accelerator
   - 异常事件表格
   - 风险等级
   - 告警文本
   - 异常视频片段
   - 缩略图
   - FP32 vs FP16 性能对比
4. 如果没有异常事件，显示“未检测到明显异常事件”。
5. 如果当前是 CPU fallback，页面醒目提示“不满足最终 GPU 推理要求，仅用于开发调试”。

README 要求：
1. 写清楚 NVIDIA WSL 当前开发环境安装方式。
2. 写清楚 AMD ROCm 最终环境迁移说明。
3. 说明 Python 代码通过 PyTorch 统一设备接口迁移，不写 NVIDIA 专用逻辑。
4. 说明本地模型 models/yolov8n.pt 的准备方式。
5. 说明 Ollama 可选安装方式。
6. 说明 FP32 vs FP16 benchmark 的意义。
7. 说明 CPU/GPU/NPU 异构分工。
8. 说明 NPU 未在 MVP 中实际使用，作为后续 Ryzen AI ONNX/Vitis AI 适配方向。

请生成完整、可运行、模块化的 Python 项目代码。

---

## 补充需求：接入本地 Ollama 中文告警模型

请把当前 Python 项目接入本地 Ollama，用于生成安防异常事件的中文告警文本。

背景：
我已经在本机安装好 Ollama，并且已经成功下载并测试 qwen2.5:3b。
Ollama 本地 API 地址为：
http://127.0.0.1:11434/api/generate

项目要求：
1. 不允许调用任何外部云端 API。
2. 只允许访问 127.0.0.1 或 localhost。
3. 默认模型使用 qwen2.5:3b。
4. 调用 Ollama 时必须设置 `"stream": false`，方便 Python 端一次性解析 JSON。
5. 如果 Ollama 未启动、模型不存在、请求超时、接口返回异常或返回内容为空，程序不能崩溃，必须自动回退到模板告警。
6. result.json 中必须记录本次是否使用了 LLM，以及失败原因。
7. Gradio 页面中可以展示告警文本，但不需要展示完整 prompt。

请实现或修改以下模块：

一、config.yaml 中增加或确认以下配置：

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

二、实现 src/local_llm_client.py：
1. 定义 LocalLLMClient 类。
2. `__init__` 参数包括 `api_url`、`model_name`、`timeout_seconds`、`allow_external_api`。
3. 初始化时校验 `api_url`，允许 host 只包括 `127.0.0.1`、`localhost`、`::1`；其他 host 抛出 ValueError 或返回不可用状态。
4. 实现 `is_available()` 方法，向 Ollama 发一个很短的测试请求；如果能正常返回 response 字段则返回 True，否则返回 False 和错误原因。
5. 实现 `generate(prompt: str) -> tuple[str, dict]`，使用 requests.post 调用 api_url，请求体必须包含 `model`、`prompt`、`stream=false`、`options.temperature=0.2`，timeout 使用 `timeout_seconds`。
6. 成功时返回生成文本和 metadata；失败时不要抛到主流程导致程序崩溃，而是返回空字符串和错误信息。
7. metadata 至少包含 `llm_used`、`provider`、`model_name`、`local_only`、`error`。

三、实现或修改 src/alert_generator.py：
1. 定义 AlertGenerator 类。
2. AlertGenerator 优先调用 LocalLLMClient 生成告警文本。
3. 如果 LocalLLMClient 不可用或生成失败，自动使用模板告警。
4. 实现 `build_prompt(event: dict)` 方法，提示词如下：

```text
你是一个本地安防视频告警助手。请根据结构化事件信息生成一句中文告警说明。

要求：
1. 不超过 60 个汉字。
2. 必须包含风险等级、时间范围、事件类型、持续时长和处理建议。
3. 不要编造身份信息、人名、地点名。
4. 不要输出多余解释。
5. 只输出一句话。

事件信息：
事件类型：{event_type}
风险等级：{risk_level}
开始时间：{start_time} 秒
结束时间：{end_time} 秒
持续时长：{duration} 秒
轨迹编号：{track_id}
```

5. 实现 `template_alert(event: dict)`：
   - `restricted_area_intrusion`：`高风险告警：{start_time}秒至{end_time}秒检测到人员进入禁区，持续约{duration}秒，建议立即查看片段。`
   - `loitering`：`中风险告警：{start_time}秒至{end_time}秒检测到人员长时间徘徊，持续约{duration}秒，建议关注现场情况。`
   - 其他事件：`告警：{start_time}秒至{end_time}秒检测到异常事件，建议查看视频片段。`
6. 实现 `generate(event: dict) -> dict`，返回更新后的 event，并写入 `alert_text`、`llm_used`、`llm_provider`、`llm_model`、`llm_fallback_reason`。
7. 实现 `generate_for_events(events: list[dict]) -> list[dict]`。

四、修改 pipeline.py 或 video_security_agent.py：
1. 在事件检测和片段截取之后，调用 `AlertGenerator.generate_for_events(events)`。
2. 每个 event 都要包含 `alert_text`。
3. result.json 中增加 llm 总体信息：

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

4. 如果部分事件用 LLM 成功、部分事件模板兜底，要正确统计 `fallback_count`。
5. 如果没有事件，也要在 result.json 中保留 `llm` 字段，`llm_used` 可以是 false。

五、修改 app.py / Gradio 页面：
1. 事件表格中展示 `alert_text`。
2. 事件详情中展示 `event_type`、`risk_level`、`start_time`、`end_time`、`duration`、`alert_text`、`llm_used`。
3. 如果 Ollama 不可用，页面不要报错，只显示模板告警。
4. 可在页面硬件/系统信息区域显示：`LLM：Ollama qwen2.5:3b，本地调用，数据不出机。`

六、增加测试脚本 scripts/test_ollama.py：
1. 从 config.yaml 读取 llm 配置。
2. 调用 LocalLLMClient。
3. 使用测试事件生成告警：

```json
{
  "event_type": "restricted_area_intrusion",
  "risk_level": "high",
  "start_time": 12.5,
  "end_time": 16.0,
  "duration": 3.5,
  "track_id": 2
}
```

4. 打印 Ollama API 是否可用、返回的告警文本、metadata。
5. 如果失败，打印失败原因，但脚本不要崩溃。

七、README 中补充 Ollama 使用说明：
1. 安装 Ollama。
2. 下载模型：`ollama pull qwen2.5:3b`
3. 启动服务：`ollama serve`
4. 测试 API：

```bash
curl http://127.0.0.1:11434/api/generate -d '{
  "model": "qwen2.5:3b",
  "prompt": "请生成一句监控告警：有人进入禁区，持续3.5秒，风险等级高。",
  "stream": false
}'
```

5. 说明：
   - Ollama 只用于生成告警文本。
   - 原始视频不会发送给 Ollama。
   - 发送给 Ollama 的只有结构化事件信息。
   - 如果 Ollama 不可用，系统自动使用模板告警。
   - 不调用云端 API，数据不出机。

请保持代码模块化、可读性强，并确保 Ollama 不可用时不会影响主流程运行。
