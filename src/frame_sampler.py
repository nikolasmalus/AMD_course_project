from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2


@dataclass
class FrameSample:
    frame_index: int
    timestamp: float
    path: str

    def to_dict(self) -> dict:
        return asdict(self)


def sample_frames(video_path: str | Path, output_dir: str | Path, sample_fps: float) -> list[FrameSample]:
    if sample_fps <= 0:
        raise ValueError("sample_fps must be greater than 0.")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if video_fps <= 0:
        video_fps = sample_fps
    frame_interval = max(1, round(video_fps / sample_fps))

    samples: list[FrameSample] = []
    frame_index = 0
    saved_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % frame_interval == 0:
            timestamp = frame_index / video_fps
            frame_path = output / f"frame_{saved_index:06d}_{timestamp:.3f}.jpg"
            cv2.imwrite(str(frame_path), frame)
            samples.append(FrameSample(frame_index=frame_index, timestamp=round(timestamp, 3), path=str(frame_path)))
            saved_index += 1
        frame_index += 1
    cap.release()
    return samples
