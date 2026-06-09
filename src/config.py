from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "app": {"title": "本地监控视频异常事件分析系统", "host": "127.0.0.1", "port": 7860},
    "paths": {
        "model_path": "models/yolov8n.pt",
        "yolo8n_model_path": "models/yolov8n.pt",
        "yolo8s_model_path": "models/yolov8s.pt",
        "yolo11m_model_path": "models/yolo11m.pt",
        "uploads_dir": "data/uploads",
        "frames_dir": "data/frames",
        "clips_dir": "data/clips",
        "results_dir": "data/results",
        "eval_dir": "data/eval",
    },
    "model": {
        "allow_auto_download": True,
        "default_profile": "fast",
        "profiles": {
            "fast": {"label": "快速模式 YOLOv8n", "model_path": "models/yolov8n.pt"},
            "accurate": {"label": "精准模式 YOLOv8s", "model_path": "models/yolov8s.pt"},
            "enhanced": {"label": "增强模式 YOLO11m", "model_path": "models/yolo11m.pt"},
        },
        "conf_threshold": 0.35,
        "iou_threshold": 0.45,
        "image_size": 640,
    },
    "video": {"sample_fps": 2.0, "clip_padding_seconds": 3.0},
    "tracking": {"iou_threshold": 0.3, "max_missing_frames": 8},
    "events": {
        "restricted_zone": {
            "enabled": True,
            "polygon": [[0.62, 0.35], [0.98, 0.35], [0.98, 0.98], [0.62, 0.98]],
        },
        "loitering": {"enabled": True, "min_duration_seconds": 8.0, "max_movement_ratio": 0.10},
    },
    "llm": {
        "enabled": True,
        "provider": "ollama",
        "api_url": "http://127.0.0.1:11434/api/generate",
        "model_name": "qwen2.5:3b",
        "timeout_seconds": 20,
        "allow_external_api": False,
        "fallback_to_template": True,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else project_root() / "config.yaml"
    loaded: dict[str, Any] = {}
    if path.exists():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    config = _deep_merge(DEFAULT_CONFIG, loaded)
    config["_project_root"] = str(project_root())
    return config


def resolve_path(config: dict[str, Any], path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path(config.get("_project_root", project_root())) / path


def ensure_data_dirs(config: dict[str, Any]) -> None:
    for key in ("uploads_dir", "frames_dir", "clips_dir", "results_dir", "eval_dir"):
        resolve_path(config, config["paths"][key]).mkdir(parents=True, exist_ok=True)
