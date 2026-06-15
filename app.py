from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gradio as gr

from src.config import load_config
from src.pipeline import analyze_video
from src.video_io import read_first_frame
from src.zone_optimizer import draw_zone_overlay, optimize_polygon, pixel_to_normalized, sanitize_normalized_polygon


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
    "torch_version": "PyTorch 版本",
    "device_name": "设备名称",
    "cpu_count": "CPU 线程数",
    "memory_total_mb": "系统内存（MB）",
    "warning": "提示",
    "video_info": "视频基本信息",
    "hardware_info": "硬件信息",
    "model_error": "模型提示",
    "model_profile": "模型模式",
    "model_label": "模型名称",
    "model_path": "模型路径",
    "sample_count": "抽样帧数",
    "track_count": "轨迹数量",
    "restricted_polygon": "警戒区域坐标",
    "annotated_video_path": "标注视频路径",
    "events": "异常事件",
    "risk_level": "风险等级",
    "alert_text": "告警文本",
    "benchmark": "性能对比",
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


def _video_path_from_input(video_path: Any) -> str | None:
    if isinstance(video_path, dict):
        video_path = video_path.get("video") or video_path.get("path") or video_path.get("name")
    return str(video_path) if video_path else None


MODEL_PROFILE_LABELS = {
    profile_name: profile.get("label", profile_name)
    for profile_name, profile in CONFIG["model"].get("profiles", {}).items()
}
MODEL_LABEL_TO_PROFILE = {label: name for name, label in MODEL_PROFILE_LABELS.items()}

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
    "FP16 optimization requires GPU": "FP16 优化需要 GPU",
    "No sampled frames available for benchmark.": "没有可用于性能测试的抽样帧。",
    "GPU memory uses torch.cuda.max_memory_allocated when available; CPU memory peak is not measured.": "GPU 显存峰值使用 torch.cuda.max_memory_allocated 统计；CPU 内存峰值未统计。",
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
        return [["-", "未检测到明显异常事件", "-", "-", "-", "-", "-", "未检测到明显异常事件"]]
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


def _zone_status(points: list[list[float]], message: str = "") -> str:
    polygon = sanitize_normalized_polygon(points)
    prefix = f"{message}\n\n" if message else ""
    if len(polygon) < 3:
        return prefix + f"已选择 {len(polygon)} 个点，至少需要 3 个点形成警戒区域。"
    return prefix + f"已选择 {len(polygon)} 个点，警戒区域已就绪。"


def load_zone_preview(video_path: Any):
    path = _video_path_from_input(video_path)
    if not path:
        return None, None, [], "请先上传 MP4 视频。"
    frame = read_first_frame(path)
    return frame, draw_zone_overlay(frame, []), [], _zone_status([], "已加载视频首帧。")


def add_zone_point(base_frame, points: list[list[float]] | None, evt: gr.SelectData):
    if base_frame is None:
        raise gr.Error("请先上传视频并加载首帧。")
    height, width = base_frame.shape[:2]
    index = evt.index
    if not isinstance(index, (tuple, list)) or len(index) < 2:
        raise gr.Error("没有获取到有效点击位置。")
    x, y = int(index[0]), int(index[1])
    polygon = sanitize_normalized_polygon(points)
    polygon.append(pixel_to_normalized((x, y), width, height))
    return draw_zone_overlay(base_frame, polygon), polygon, _zone_status(polygon)


def undo_zone_point(base_frame, points: list[list[float]] | None):
    polygon = sanitize_normalized_polygon(points)
    if polygon:
        polygon = polygon[:-1]
    return draw_zone_overlay(base_frame, polygon), polygon, _zone_status(polygon, "已撤销上一个点。")


def clear_zone_points(base_frame):
    return draw_zone_overlay(base_frame, []), [], _zone_status([], "已清空警戒区域。")


def optimize_zone_points(base_frame, points: list[list[float]] | None):
    if base_frame is None:
        raise gr.Error("请先上传视频并加载首帧。")
    polygon = sanitize_normalized_polygon(points)
    if len(polygon) < 3:
        raise gr.Error("请至少选择 3 个点后再优化区域。")
    height, width = base_frame.shape[:2]
    optimized = optimize_polygon(polygon, width, height)
    return draw_zone_overlay(base_frame, optimized, optimized=True), optimized, _zone_status(optimized, "已完成区域优化。")


def run_ui(
    video_path: str,
    sample_fps: float,
    conf_threshold: float,
    loitering_min_duration: float,
    loitering_max_movement: float,
    model_profile_label: str,
    zone_points: list[list[float]] | None,
):
    video_path = _video_path_from_input(video_path)
    if not video_path:
        raise gr.Error("请先上传 MP4 视频。")
    restricted_polygon = sanitize_normalized_polygon(zone_points)
    if len(restricted_polygon) < 3:
        raise gr.Error("请先在首帧图像上点选至少 3 个警戒区域点。")
    result = analyze_video(
        video_path,
        sample_fps=sample_fps,
        conf_threshold=conf_threshold,
        loitering_min_duration=loitering_min_duration,
        loitering_max_movement=loitering_max_movement,
        restricted_polygon=restricted_polygon,
        model_profile=MODEL_LABEL_TO_PROFILE.get(model_profile_label, CONFIG["model"].get("default_profile", "fast")),
    )
    events = result.get("events", [])
    hardware = result.get("hardware_info", {})

    clips = [event["clip_path"] for event in events if Path(event.get("clip_path", "")).exists()]
    thumbs = [event["thumbnail_path"] for event in events if Path(event.get("thumbnail_path", "")).exists()]
    result_json = json.dumps(_localized_report(result.get("result", {})), ensure_ascii=False, indent=2)

    return (
        result.get("annotated_video_path") or result["video_path"],
        _markdown_dict("视频基本信息", result.get("video_info", {})),
        _markdown_dict("硬件信息", hardware),
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
    zone_base_frame = gr.State(None)
    zone_points = gr.State([])
    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="上传本地 MP4", sources=["upload"])
            zone_preview = gr.Image(label="首帧警戒区域", type="numpy", interactive=False, height=360)
            with gr.Row():
                undo_zone = gr.Button("撤销点")
                clear_zone = gr.Button("清空区域")
                optimize_zone = gr.Button("优化区域")
            zone_status = gr.Markdown("上传视频后，可在首帧图像上点选警戒区域。")
            model_profile = gr.Dropdown(
                choices=list(MODEL_LABEL_TO_PROFILE.keys()),
                value=MODEL_PROFILE_LABELS.get(CONFIG["model"].get("default_profile", "fast"), "快速模式 YOLOv8n"),
                label="模型模式",
            )
            sample_fps = gr.Slider(0.5, 10.0, value=float(CONFIG["video"]["sample_fps"]), step=0.5, label="抽帧帧率")
            conf_threshold = gr.Slider(0.1, 0.9, value=float(CONFIG["model"]["conf_threshold"]), step=0.05, label="检测置信度阈值")
            loitering_min_duration = gr.Slider(1.0, 60.0, value=float(CONFIG["events"]["loitering"]["min_duration_seconds"]), step=1.0, label="徘徊最短持续时间（秒）")
            loitering_max_movement = gr.Slider(0.01, 0.5, value=float(CONFIG["events"]["loitering"]["max_movement_ratio"]), step=0.01, label="徘徊最大移动比例")
            run_button = gr.Button("开始分析", variant="primary")
        with gr.Column(scale=2):
            original_video = gr.Video(label="分析视频")
            video_info = gr.Markdown()
            hardware_info = gr.Markdown()
    events_table = gr.Dataframe(headers=["事件编号", "事件类型", "风险等级", "轨迹编号", "开始秒", "结束秒", "持续时长（秒）", "告警文本"], label="异常事件详情")
    with gr.Row():
        risk = gr.Textbox(label="风险等级")
        alert = gr.Textbox(label="中文告警说明", lines=3)
    with gr.Row():
        clips = gr.File(label="异常视频片段", file_count="multiple")
        thumbnails = gr.Gallery(label="缩略图", columns=4, height=260)
    benchmark = gr.Dataframe(headers=["模型方案", "是否可用", "平均延迟（毫秒）", "吞吐率（帧/秒）", "人员检测数量", "显存峰值（MB）", "错误说明"], label="FP32 与 FP16 性能对比")
    result_file = gr.File(label="报告文件")
    result_json = gr.Code(label="报告内容", language="json")

    video_input.change(load_zone_preview, inputs=[video_input], outputs=[zone_base_frame, zone_preview, zone_points, zone_status])
    zone_preview.select(add_zone_point, inputs=[zone_base_frame, zone_points], outputs=[zone_preview, zone_points, zone_status])
    undo_zone.click(undo_zone_point, inputs=[zone_base_frame, zone_points], outputs=[zone_preview, zone_points, zone_status])
    clear_zone.click(clear_zone_points, inputs=[zone_base_frame], outputs=[zone_preview, zone_points, zone_status])
    optimize_zone.click(optimize_zone_points, inputs=[zone_base_frame, zone_points], outputs=[zone_preview, zone_points, zone_status])
    run_button.click(
        run_ui,
        inputs=[video_input, sample_fps, conf_threshold, loitering_min_duration, loitering_max_movement, model_profile, zone_points],
        outputs=[original_video, video_info, hardware_info, events_table, risk, alert, clips, thumbnails, benchmark, result_file, result_json],
    )


if __name__ == "__main__":
    app_cfg = CONFIG["app"]
    demo.launch(server_name=app_cfg.get("host", "127.0.0.1"), server_port=int(app_cfg.get("port", 7860)))
