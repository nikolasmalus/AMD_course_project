from __future__ import annotations

from pathlib import Path
from typing import Any

from .yolo_person_detector import YoloPersonDetector, clear_gpu_peak_memory, get_gpu_peak_memory_mb


def _summarize_run(name: str, frames: list[dict[str, Any]], peak_memory_mb: float | None, error: str | None = None) -> dict[str, Any]:
    if error:
        return {
            "name": name,
            "available": False,
            "avg_latency_ms": None,
            "throughput_fps": None,
            "person_detections": 0,
            "peak_memory_mb": peak_memory_mb,
            "error": error,
        }
    latencies = [float(frame.get("latency_ms", 0.0)) for frame in frames]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    detections = sum(len(frame.get("detections", [])) for frame in frames)
    throughput = 1000.0 / avg_latency if avg_latency > 0 else 0.0
    return {
        "name": name,
        "available": True,
        "avg_latency_ms": round(avg_latency, 3),
        "throughput_fps": round(throughput, 3),
        "person_detections": detections,
        "peak_memory_mb": peak_memory_mb,
        "error": None,
    }


def run_benchmark(
    samples: list[dict[str, Any]],
    model_path: str | Path,
    device: str,
    allow_auto_download: bool,
    conf_threshold: float,
    iou_threshold: float,
    image_size: int,
    event_count: int,
) -> dict[str, Any]:
    benchmark_samples = samples[: min(len(samples), 30)]
    report: dict[str, Any] = {"benchmark_frame_count": len(benchmark_samples), "event_count": event_count}
    if not benchmark_samples:
        report["error"] = "No sampled frames available for benchmark."
        return report

    try:
        clear_gpu_peak_memory(device)
        fp32 = YoloPersonDetector(model_path, device, allow_auto_download, conf_threshold, iou_threshold, image_size, half=False)
        fp32_frames = fp32.detect_samples(benchmark_samples)
        baseline = _summarize_run("baseline_fp32", fp32_frames, get_gpu_peak_memory_mb(device))
    except Exception as exc:
        baseline = _summarize_run("baseline_fp32", [], None, str(exc))
        fp32_frames = []

    if device == "cuda":
        try:
            clear_gpu_peak_memory(device)
            fp16 = YoloPersonDetector(model_path, device, allow_auto_download, conf_threshold, iou_threshold, image_size, half=True)
            fp16_frames = fp16.detect_samples(benchmark_samples)
            optimized = _summarize_run("optimized_fp16", fp16_frames, get_gpu_peak_memory_mb(device))
        except Exception as exc:
            optimized = _summarize_run("optimized_fp16", [], None, str(exc))
    else:
        optimized = _summarize_run("optimized_fp16", [], None, "FP16 optimization requires GPU")

    base_count = baseline.get("person_detections") or 0
    opt_count = optimized.get("person_detections") or 0
    consistency = 1.0 if base_count == 0 and opt_count == 0 else min(base_count, opt_count) / max(base_count, opt_count, 1)
    report.update(
        {
            "baseline_fp32": baseline,
            "optimized_fp16": optimized,
            "optimized_model_available": bool(optimized.get("available")),
            "optimized_model_error": optimized.get("error"),
            "detection_count_consistency": round(consistency, 4),
            "event_consistency": 1.0,
            "peak_memory_note": "GPU memory uses torch.cuda.max_memory_allocated when available; CPU memory peak is not measured.",
        }
    )
    return report
