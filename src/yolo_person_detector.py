from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - handled at runtime for clearer UI feedback
    YOLO = None


class ModelUnavailableError(RuntimeError):
    pass


class YoloPersonDetector:
    def __init__(
        self,
        model_path: str | Path,
        device: str,
        allow_auto_download: bool = False,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        image_size: int = 640,
        half: bool = False,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = device
        self.allow_auto_download = allow_auto_download
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.half = bool(half and device == "cuda")
        self.model = self._load_model()

    def _load_model(self) -> Any:
        if YOLO is None:
            raise ModelUnavailableError("Ultralytics is not installed. Run: pip install -r requirements.txt")
        if self.model_path.exists():
            model_ref = str(self.model_path)
        elif self.allow_auto_download:
            model_ref = "yolov8n.pt"
        else:
            raise ModelUnavailableError(
                f"Model file not found: {self.model_path}. Put yolov8n.pt under models/ or set allow_auto_download=true for development."
            )
        return YOLO(model_ref)

    def detect_frame(self, frame_path: str | Path) -> tuple[list[dict[str, Any]], float]:
        start = time.perf_counter()
        results = self.model.predict(
            source=str(frame_path),
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            classes=[0],
            device=self.device,
            half=self.half,
            verbose=False,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        detections: list[dict[str, Any]] = []
        if not results:
            return detections, latency_ms
        boxes = results[0].boxes
        if boxes is None:
            return detections, latency_ms

        xyxy = boxes.xyxy.detach().cpu().numpy() if boxes.xyxy is not None else np.empty((0, 4))
        confs = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.empty((0,))
        for bbox, conf in zip(xyxy, confs):
            detections.append(
                {
                    "bbox": [float(v) for v in bbox.tolist()],
                    "confidence": float(conf),
                    "class_id": 0,
                    "label": "person",
                }
            )
        return detections, latency_ms

    def detect_samples(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        output = []
        for sample in samples:
            detections, latency_ms = self.detect_frame(sample["path"])
            output.append({**sample, "detections": detections, "latency_ms": latency_ms})
        return output


def clear_gpu_peak_memory(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def get_gpu_peak_memory_mb(device: str) -> float | None:
    if device == "cuda" and torch.cuda.is_available():
        return round(float(torch.cuda.max_memory_allocated()) / 1024 / 1024, 2)
    return None
