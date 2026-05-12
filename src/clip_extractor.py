from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from .media_compat import transcode_to_browser_mp4


def _safe_event_name(event: dict[str, Any]) -> str:
    return f"{event.get('start_time', 0):07.3f}_{event.get('event_id', 'event')}".replace(".", "_").replace("/", "_")


def extract_event_clips(
    video_path: str | Path,
    events: list[dict[str, Any]],
    clips_dir: str | Path,
    padding_seconds: float = 3.0,
) -> list[dict[str, Any]]:
    clips_root = Path(clips_dir)
    clips_root.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frame_count / fps if fps > 0 else 0.0

    enriched: list[dict[str, Any]] = []
    for event in events:
        start_time = max(0.0, float(event["start_time"]) - padding_seconds)
        end_time = min(duration, float(event["end_time"]) + padding_seconds)
        if end_time <= start_time:
            end_time = min(duration, start_time + 1.0)

        name = _safe_event_name(event)
        clip_path = clips_root / f"{name}.mp4"
        thumb_path = clips_root / f"{name}.jpg"

        start_frame = max(0, int(start_time * fps))
        end_frame = min(frame_count, max(start_frame + 1, int(end_time * fps)))
        writer = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current = start_frame
        thumb_written = False
        mid_frame = (start_frame + end_frame) // 2
        while current < end_frame:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            if not thumb_written and current >= mid_frame:
                cv2.imwrite(str(thumb_path), frame)
                thumb_written = True
            current += 1
        writer.release()
        clip_path = Path(transcode_to_browser_mp4(clip_path, clip_path.with_name(f"{clip_path.stem}_h264.mp4")))
        if not thumb_written:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            ok, frame = cap.read()
            if ok:
                cv2.imwrite(str(thumb_path), frame)

        enriched.append({**event, "clip_path": str(clip_path), "thumbnail_path": str(thumb_path), "clip_start": start_time, "clip_end": end_time})

    cap.release()
    return enriched
