from __future__ import annotations

import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: str
    filename: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


def save_uploaded_video(uploaded_path: str | Path, uploads_dir: str | Path) -> Path:
    source = Path(uploaded_path)
    if not source.exists():
        raise FileNotFoundError(f"Uploaded video not found: {source}")
    if source.suffix.lower() != ".mp4":
        raise ValueError("Only local MP4 files are supported.")

    uploads = Path(uploads_dir)
    uploads.mkdir(parents=True, exist_ok=True)
    safe_name = f"{int(time.time())}_{source.name}"
    target = uploads / safe_name
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    return target


def read_video_info(video_path: str | Path) -> VideoInfo:
    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    duration = frame_count / fps if fps > 0 else 0.0
    return VideoInfo(
        path=str(path),
        filename=path.name,
        width=width,
        height=height,
        fps=round(fps, 3),
        frame_count=frame_count,
        duration_seconds=round(duration, 3),
    )


def read_first_frame(video_path: str | Path) -> np.ndarray:
    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {path}")
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise ValueError(f"Unable to read first frame: {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
