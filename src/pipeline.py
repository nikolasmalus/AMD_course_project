from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .alert_generator import AlertGenerator, risk_level, summarize_alert_text
from .benchmark_profiler import run_benchmark
from .clip_extractor import extract_event_clips
from .config import ensure_data_dirs, load_config, resolve_path
from .event_detector import EventDetector
from .frame_sampler import sample_frames
from .hardware_manager import detect_hardware
from .simple_tracker import track_frame_results
from .video_io import read_video_info, save_uploaded_video
from .video_security_agent import FunctionTool, VideoSecurityAgent
from .yolo_person_detector import ModelUnavailableError, YoloPersonDetector


def _run_dir(base_dir: Path, prefix: str) -> Path:
    path = base_dir / f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: str | Path, data: dict[str, Any]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(target)


def build_agent(config: dict[str, Any]) -> VideoSecurityAgent:
    def analyze_video_tool(context: dict[str, Any]) -> dict[str, Any]:
        ensure_data_dirs(config)
        uploads_dir = resolve_path(config, config["paths"]["uploads_dir"])
        saved_video = save_uploaded_video(context["video_path"], uploads_dir)
        video_info = read_video_info(saved_video)
        results_dir = _run_dir(resolve_path(config, config["paths"]["results_dir"]), saved_video.stem)
        frames_dir = _run_dir(resolve_path(config, config["paths"]["frames_dir"]), saved_video.stem)
        clips_dir = _run_dir(resolve_path(config, config["paths"]["clips_dir"]), saved_video.stem)
        context.update(
            {
                "video_path": str(saved_video),
                "video_info": video_info.to_dict(),
                "results_dir": str(results_dir),
                "frames_dir": str(frames_dir),
                "clips_dir": str(clips_dir),
                "hardware_info": detect_hardware().to_dict(),
            }
        )
        return context

    def detect_persons_tool(context: dict[str, Any]) -> dict[str, Any]:
        sample_fps = float(context.get("sample_fps", config["video"]["sample_fps"]))
        samples = [sample.to_dict() for sample in sample_frames(context["video_path"], context["frames_dir"], sample_fps)]
        context["samples"] = samples

        hardware = context["hardware_info"]
        model_path = resolve_path(config, config["paths"]["model_path"])
        try:
            detector = YoloPersonDetector(
                model_path=model_path,
                device=hardware["actual_device"],
                allow_auto_download=bool(config["model"].get("allow_auto_download", False)),
                conf_threshold=float(context.get("conf_threshold", config["model"]["conf_threshold"])),
                iou_threshold=float(config["model"]["iou_threshold"]),
                image_size=int(config["model"]["image_size"]),
                half=False,
            )
            context["frame_results"] = detector.detect_samples(samples)
            context["model_error"] = None
        except ModelUnavailableError as exc:
            context["frame_results"] = [{**sample, "detections": [], "latency_ms": 0.0} for sample in samples]
            context["model_error"] = str(exc)
        except Exception as exc:
            context["frame_results"] = [{**sample, "detections": [], "latency_ms": 0.0} for sample in samples]
            context["model_error"] = f"Person detector failed: {exc}"
        return context

    def track_persons_tool(context: dict[str, Any]) -> dict[str, Any]:
        tracked_frames, tracks = track_frame_results(
            context.get("frame_results", []),
            iou_threshold=float(config["tracking"]["iou_threshold"]),
            max_missing_frames=int(config["tracking"]["max_missing_frames"]),
        )
        context["tracked_frames"] = tracked_frames
        context["tracks"] = tracks
        return context

    def detect_events_tool(context: dict[str, Any]) -> dict[str, Any]:
        video_info = context["video_info"]
        restricted = config["events"]["restricted_zone"]
        loitering = config["events"]["loitering"]
        detector = EventDetector(
            frame_width=int(video_info["width"]),
            frame_height=int(video_info["height"]),
            restricted_polygon=restricted.get("polygon", []),
            restricted_enabled=bool(restricted.get("enabled", True)),
            loitering_enabled=bool(loitering.get("enabled", True)),
            loitering_min_duration=float(context.get("loitering_min_duration", loitering["min_duration_seconds"])),
            loitering_max_movement_ratio=float(context.get("loitering_max_movement", loitering["max_movement_ratio"])),
        )
        context["events"] = detector.detect(context.get("tracks", []))
        return context

    def generate_clips_tool(context: dict[str, Any]) -> dict[str, Any]:
        context["events"] = extract_event_clips(
            context["video_path"],
            context.get("events", []),
            context["clips_dir"],
            padding_seconds=float(config["video"]["clip_padding_seconds"]),
        )
        return context

    def generate_alert_tool(context: dict[str, Any]) -> dict[str, Any]:
        generator = AlertGenerator(config["llm"])
        events = generator.generate_for_events(context.get("events", []))
        context["events"] = events
        context["llm"] = generator.summarize_llm(events)
        context["alert_text"] = summarize_alert_text(events)
        context["llm_used"] = context["llm"]["llm_used"]
        context["llm_fallback_reason"] = "; ".join(context["llm"]["fallback_reasons"]) if context["llm"]["fallback_reasons"] else None
        context["risk_level"] = risk_level(context.get("events", []))
        return context

    def generate_report_tool(context: dict[str, Any]) -> dict[str, Any]:
        benchmark = run_benchmark(
            samples=context.get("samples", []),
            model_path=resolve_path(config, config["paths"]["model_path"]),
            device=context["hardware_info"]["actual_device"],
            allow_auto_download=bool(config["model"].get("allow_auto_download", False)),
            conf_threshold=float(context.get("conf_threshold", config["model"]["conf_threshold"])),
            iou_threshold=float(config["model"]["iou_threshold"]),
            image_size=int(config["model"]["image_size"]),
            event_count=len(context.get("events", [])),
        )
        context["benchmark"] = benchmark
        result = {
            "video_info": context.get("video_info"),
            "hardware_info": context.get("hardware_info"),
            "model_error": context.get("model_error"),
            "sample_count": len(context.get("samples", [])),
            "track_count": len(context.get("tracks", [])),
            "events": context.get("events", []),
            "risk_level": context.get("risk_level"),
            "alert_text": context.get("alert_text"),
            "llm": context.get("llm"),
            "llm_used": context.get("llm_used"),
            "llm_fallback_reason": context.get("llm_fallback_reason"),
            "benchmark": benchmark,
            "npu_adaptation_note": "NPU is recorded as future Ryzen AI ONNX/Vitis AI adaptation; MVP runs YOLO via PyTorch CPU/GPU.",
        }
        if context["hardware_info"].get("warning"):
            result["warning"] = context["hardware_info"]["warning"]
        context["result_json_path"] = _write_json(Path(context["results_dir"]) / "result.json", result)
        context["benchmark_json_path"] = _write_json(Path(context["results_dir"]) / "benchmark_report.json", benchmark)
        context["result"] = result
        return context

    return VideoSecurityAgent(
        [
            FunctionTool("AnalyzeVideoTool", analyze_video_tool),
            FunctionTool("DetectPersonsTool", detect_persons_tool),
            FunctionTool("TrackPersonsTool", track_persons_tool),
            FunctionTool("DetectEventsTool", detect_events_tool),
            FunctionTool("GenerateClipsTool", generate_clips_tool),
            FunctionTool("GenerateAlertTool", generate_alert_tool),
            FunctionTool("GenerateReportTool", generate_report_tool),
        ]
    )


def analyze_video(
    video_path: str | Path,
    config_path: str | Path | None = None,
    sample_fps: float | None = None,
    conf_threshold: float | None = None,
    loitering_min_duration: float | None = None,
    loitering_max_movement: float | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    context: dict[str, Any] = {"video_path": str(video_path)}
    if sample_fps is not None:
        context["sample_fps"] = sample_fps
    if conf_threshold is not None:
        context["conf_threshold"] = conf_threshold
    if loitering_min_duration is not None:
        context["loitering_min_duration"] = loitering_min_duration
    if loitering_max_movement is not None:
        context["loitering_max_movement"] = loitering_max_movement
    return build_agent(config).run(context)
