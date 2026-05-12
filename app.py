from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gradio as gr

from src.config import load_config
from src.hardware_manager import CPU_FALLBACK_MESSAGE
from src.pipeline import analyze_video


CONFIG = load_config()

KEY_LABELS = {
    "path": "文件路径",
    "filename": "文件名",
    "width": "宽度",
    "height": "高度",
    "fps": "视频帧率",
    "frame_count": "总帧数",
    "duration_seconds": "视频时长（秒）",
    "actual_device": "实际推理设备",
    "uses_accelerator": "是否使用加速器",
    "torch_version": "PyTorch 版本",
    "torch_cuda_available": "PyTorch 加速接口可用",
    "device_name": "设备名称",
    "backend_note": "后端说明",
    "cpu_count": "CPU 线程数",
    "memory_total_mb": "系统内存（MB）",
    "npu_available": "NPU 是否可用",
    "npu_note": "NPU 说明",
    "warning": "提示",
    "video_info": "视频基本信息",
    "hardware_info": "硬件信息",
    "model_error": "模型提示",
    "sample_count": "抽样帧数",
    "track_count": "轨迹数量",
    "events": "异常事件",
    "risk_level": "风险等级",
    "alert_text": "告警文本",
    "llm": "本地大模型",
    "llm_used": "是否使用本地大模型",
    "llm_provider": "大模型服务",
    "llm_model": "大模型名称",
    "llm_fallback_reason": "大模型回退原因",
    "provider": "服务提供方",
    "model_name": "模型名称",
    "local_only": "是否仅本地调用",
    "fallback_count": "模板兜底次数",
    "fallback_reasons": "模板兜底原因",
    "benchmark": "性能对比",
    "npu_adaptation_note": "NPU 适配说明",
    "benchmark_frame_count": "性能测试帧数",
    "event_count": "事件数量",
    "baseline_fp32": "基线 FP32",
    "optimized_fp16": "优化 FP16",
    "optimized_model_available": "优化模型是否可用",
    "optimized_model_error": "优化模型错误",
    "detection_count_consistency": "检测数量一致率",
    "event_consistency": "事件一致率",
    "peak_memory_note": "显存统计说明",
    "name": "方案名称",
    "available": "是否可用",
    "avg_latency_ms": "平均延迟（毫秒）",
    "throughput_fps": "吞吐率（帧/秒）",
    "person_detections": "人员检测数量",
    "peak_memory_mb": "显存峰值（MB）",
    "error": "错误说明",
    "event_id": "事件编号",
    "event_type": "事件类型",
    "type": "事件类型",
    "type_cn": "事件类型",
    "track_id": "轨迹编号",
    "start_time": "开始时间（秒）",
    "end_time": "结束时间（秒）",
    "severity": "风险",
    "duration": "持续时长（秒）",
    "description": "说明",
    "bbox": "检测框",
    "clip_path": "异常片段路径",
    "thumbnail_path": "缩略图路径",
    "clip_start": "片段开始（秒）",
    "clip_end": "片段结束（秒）",
}

VALUE_LABELS = {
    "high": "高",
    "medium": "中",
    "low": "低",
    "restricted_zone_intrusion": "禁区闯入",
    "restricted_area_intrusion": "禁区闯入",
    "loitering": "长时间徘徊",
    "baseline_fp32": "基线 FP32",
    "optimized_fp16": "优化 FP16",
    "consistency": "一致率",
    CPU_FALLBACK_MESSAGE: "仅 CPU 回退模式，不满足最终 AMD GPU 推理要求",
    "FP16 optimization requires GPU": "FP16 优化需要 GPU",
    "No sampled frames available for benchmark.": "没有可用于性能测试的抽样帧。",
    "GPU memory uses torch.cuda.max_memory_allocated when available; CPU memory peak is not measured.": "GPU 显存峰值使用 torch.cuda.max_memory_allocated 统计；CPU 内存峰值未统计。",
    "NPU is not used in this MVP; future Ryzen AI adaptation can export ONNX and run via Ryzen AI/Vitis AI stack.": "MVP 阶段未实际使用 NPU；后续可导出 ONNX，并通过 Ryzen AI/Vitis AI 工具链适配。",
    "NPU is recorded as future Ryzen AI ONNX/Vitis AI adaptation; MVP runs YOLO via PyTorch CPU/GPU.": "NPU 在当前版本中作为后续 Ryzen AI ONNX/Vitis AI 适配方向记录；MVP 使用 PyTorch CPU/GPU 运行 YOLO。",
    "PyTorch CUDA interface; NVIDIA CUDA in current WSL, AMD ROCm/HIP on final AMD workstation": "PyTorch 统一 CUDA 设备接口：当前 WSL 为 NVIDIA CUDA，最终 AMD 工作站为 ROCm/HIP。",
}


def _label_key(key: str) -> str:
    return KEY_LABELS.get(key, key)


def _format_value(value: Any) -> Any:
    if isinstance(value, bool):
        return "是" if value else "否"
    if value is None:
        return "无"
    if isinstance(value, str):
        return VALUE_LABELS.get(value, value)
    return value


def _markdown_dict(title: str, data: dict[str, Any]) -> str:
    lines = [f"### {title}"]
    for key, value in data.items():
        lines.append(f"- **{_label_key(key)}**: {_format_value(value)}")
    return "\n".join(lines)


def _events_table(events: list[dict[str, Any]]) -> list[list[Any]]:
    if not events:
        return [["-", "未检测到明显异常事件", "-", "-", "-", "-", "-", "未检测到明显异常事件", "否"]]
    return [
        [
            event.get("event_id"),
            _format_value(event.get("event_type", event.get("type_cn", event.get("type")))),
            _format_value(event.get("risk_level", event.get("severity"))),
            event.get("track_id"),
            event.get("start_time"),
            event.get("end_time"),
            event.get("duration"),
            event.get("alert_text"),
            _format_value(event.get("llm_used")),
        ]
        for event in events
    ]


def _benchmark_table(benchmark: dict[str, Any]) -> list[list[Any]]:
    rows = []
    for key in ("baseline_fp32", "optimized_fp16"):
        item = benchmark.get(key, {})
        rows.append(
            [
                _format_value(key),
                _format_value(item.get("available")),
                item.get("avg_latency_ms"),
                item.get("throughput_fps"),
                item.get("person_detections"),
                _format_value(item.get("peak_memory_mb")),
                _format_value(item.get("error")),
            ]
        )
    rows.append(["一致率", "是", "无", "无", benchmark.get("detection_count_consistency"), "无", f"事件一致率={benchmark.get('event_consistency')}"])
    return rows


def _localized_report(data: Any) -> Any:
    if isinstance(data, dict):
        return {_label_key(str(key)): _localized_report(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_localized_report(item) for item in data]
    return _format_value(data)


def run_ui(video_path: str, sample_fps: float, conf_threshold: float, loitering_min_duration: float, loitering_max_movement: float):
    if isinstance(video_path, dict):
        video_path = video_path.get("video") or video_path.get("path") or video_path.get("name")
    if not video_path:
        raise gr.Error("请先上传 MP4 视频。")
    result = analyze_video(
        video_path,
        sample_fps=sample_fps,
        conf_threshold=conf_threshold,
        loitering_min_duration=loitering_min_duration,
        loitering_max_movement=loitering_max_movement,
    )
    events = result.get("events", [])
    hardware = result.get("hardware_info", {})
    llm = result.get("llm", {})
    warning = ""
    if hardware.get("actual_device") == "cpu":
        warning = "\n\n> **仅 CPU 回退模式，不满足最终 AMD GPU 推理要求**"
    if result.get("model_error"):
        warning += f"\n\n> 模型提示：{result['model_error']}"

    clips = [event["clip_path"] for event in events if Path(event.get("clip_path", "")).exists()]
    thumbs = [event["thumbnail_path"] for event in events if Path(event.get("thumbnail_path", "")).exists()]
    result_json = json.dumps(_localized_report(result.get("result", {})), ensure_ascii=False, indent=2)

    return (
        result["video_path"],
        _markdown_dict("视频基本信息", result.get("video_info", {})),
        _markdown_dict("硬件信息", hardware)
        + "\n\n### 本地大模型\n"
        + f"- **说明**: Ollama {llm.get('model_name', CONFIG['llm'].get('model_name', 'qwen2.5:3b'))}，本地调用，数据不出机。\n"
        + f"- **是否使用 LLM**: {_format_value(llm.get('llm_used'))}\n"
        + f"- **模板兜底次数**: {llm.get('fallback_count', 0)}"
        + warning,
        _events_table(events),
        result.get("risk_level", "无"),
        result.get("alert_text", "未检测到明显异常事件"),
        clips,
        thumbs,
        _benchmark_table(result.get("benchmark", {})),
        result.get("result_json_path"),
        result_json,
    )


with gr.Blocks(title=CONFIG["app"]["title"]) as demo:
    gr.Markdown(f"# {CONFIG['app']['title']}")
    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="上传本地 MP4", sources=["upload"])
            sample_fps = gr.Slider(0.5, 10.0, value=float(CONFIG["video"]["sample_fps"]), step=0.5, label="抽帧帧率")
            conf_threshold = gr.Slider(0.1, 0.9, value=float(CONFIG["model"]["conf_threshold"]), step=0.05, label="检测置信度阈值")
            loitering_min_duration = gr.Slider(1.0, 60.0, value=float(CONFIG["events"]["loitering"]["min_duration_seconds"]), step=1.0, label="徘徊最短持续时间（秒）")
            loitering_max_movement = gr.Slider(0.01, 0.5, value=float(CONFIG["events"]["loitering"]["max_movement_ratio"]), step=0.01, label="徘徊最大移动比例")
            run_button = gr.Button("开始分析", variant="primary")
        with gr.Column(scale=2):
            original_video = gr.Video(label="原始视频")
            video_info = gr.Markdown()
            hardware_info = gr.Markdown()
    events_table = gr.Dataframe(headers=["事件编号", "事件类型", "风险等级", "轨迹编号", "开始秒", "结束秒", "持续时长（秒）", "告警文本", "是否使用 LLM"], label="异常事件详情")
    with gr.Row():
        risk = gr.Textbox(label="风险等级")
        alert = gr.Textbox(label="中文告警说明", lines=3)
    with gr.Row():
        clips = gr.File(label="异常视频片段", file_count="multiple")
        thumbnails = gr.Gallery(label="缩略图", columns=4, height=260)
    benchmark = gr.Dataframe(headers=["模型方案", "是否可用", "平均延迟（毫秒）", "吞吐率（帧/秒）", "人员检测数量", "显存峰值（MB）", "错误说明"], label="FP32 与 FP16 性能对比")
    result_file = gr.File(label="报告文件")
    result_json = gr.Code(label="报告内容", language="json")

    run_button.click(
        run_ui,
        inputs=[video_input, sample_fps, conf_threshold, loitering_min_duration, loitering_max_movement],
        outputs=[original_video, video_info, hardware_info, events_table, risk, alert, clips, thumbnails, benchmark, result_file, result_json],
    )


if __name__ == "__main__":
    app_cfg = CONFIG["app"]
    demo.launch(server_name=app_cfg.get("host", "127.0.0.1"), server_port=int(app_cfg.get("port", 7860)))
