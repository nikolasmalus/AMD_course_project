from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .event_detector import bbox_overlaps_polygon
from .media_compat import transcode_to_browser_mp4


def draw_frame(
    frame_path: str | Path,
    detections: list[dict[str, Any]],
    restricted_polygon: list[list[float]] | None,
    output_path: str | Path,
) -> str:
    image = cv2.imread(str(frame_path))
    if image is None:
        raise ValueError(f"Unable to read frame: {frame_path}")
    height, width = image.shape[:2]

    if restricted_polygon:
        points = np.array([[int(x * width), int(y * height)] for x, y in restricted_polygon], dtype=np.int32)
        cv2.polylines(image, [points], isClosed=True, color=(0, 0, 255), thickness=2)

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        track_id = det.get("track_id", "?")
        conf = det.get("confidence", 0.0)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 180, 0), 2)
        cv2.putText(image, f"ID {track_id} {conf:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 0), 2)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), image)
    return str(output)


def _scaled_polygon(restricted_polygon: list[list[float]] | None, width: int, height: int) -> list[list[float]]:
    return [[float(x) * width, float(y) * height] for x, y in (restricted_polygon or [])]


def _track_histories(tracked_frames: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    histories: dict[int, list[dict[str, Any]]] = {}
    for frame in tracked_frames:
        for detection in frame.get("detections", []):
            track_id = detection.get("track_id")
            if track_id is None:
                continue
            histories.setdefault(int(track_id), []).append(
                {
                    "timestamp": float(frame.get("timestamp", 0.0)),
                    "bbox": [float(value) for value in detection.get("bbox", [])],
                    "confidence": float(detection.get("confidence", 0.0)),
                }
            )
    return {track_id: sorted(items, key=lambda item: item["timestamp"]) for track_id, items in histories.items()}


def _interpolate_bbox(history: list[dict[str, Any]], timestamp: float, hold_seconds: float) -> tuple[list[float], float] | None:
    if not history:
        return None
    if len(history) == 1:
        item = history[0]
        if abs(timestamp - item["timestamp"]) <= hold_seconds:
            return item["bbox"], item["confidence"]
        return None

    for first, second in zip(history, history[1:]):
        start = first["timestamp"]
        end = second["timestamp"]
        if start <= timestamp <= end:
            span = max(end - start, 1e-6)
            ratio = (timestamp - start) / span
            bbox = [float(a) + (float(b) - float(a)) * ratio for a, b in zip(first["bbox"], second["bbox"])]
            confidence = float(first["confidence"]) + (float(second["confidence"]) - float(first["confidence"])) * ratio
            return bbox, confidence

    nearest = min(history, key=lambda item: abs(timestamp - item["timestamp"]))
    if abs(timestamp - nearest["timestamp"]) <= hold_seconds:
        return nearest["bbox"], nearest["confidence"]
    return None


def render_annotated_video(
    video_path: str | Path,
    tracked_frames: list[dict[str, Any]],
    restricted_polygon: list[list[float]] | None,
    output_path: str | Path,
    restricted_min_bbox_points_ratio: float = 0.10,
) -> str:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    histories = _track_histories(tracked_frames)
    sample_times = sorted({float(frame.get("timestamp", 0.0)) for frame in tracked_frames})
    if len(sample_times) >= 2:
        intervals = [second - first for first, second in zip(sample_times, sample_times[1:]) if second > first]
        hold_seconds = max(0.15, min(intervals) / 2.0) if intervals else 0.25
    else:
        hold_seconds = 0.25

    scaled_polygon = _scaled_polygon(restricted_polygon, width, height)
    polygon_points = np.array(scaled_polygon, dtype=np.int32) if len(scaled_polygon) >= 3 else None

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        timestamp = frame_index / fps if fps > 0 else 0.0

        if polygon_points is not None:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [polygon_points], color=(0, 0, 255))
            frame = cv2.addWeighted(overlay, 0.12, frame, 0.88, 0)
            cv2.polylines(frame, [polygon_points], isClosed=True, color=(0, 0, 255), thickness=2)

        for track_id, history in histories.items():
            interpolated = _interpolate_bbox(history, timestamp, hold_seconds)
            if interpolated is None:
                continue
            bbox, confidence = interpolated
            in_zone = bool(scaled_polygon) and bbox_overlaps_polygon(bbox, scaled_polygon, restricted_min_bbox_points_ratio)
            color = (0, 0, 255) if in_zone else (0, 180, 0)
            x1, y1, x2, y2 = [int(value) for value in bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ALERT ID {track_id}" if in_zone else f"ID {track_id}"
            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        writer.write(frame)
        frame_index += 1
        if frame_count and frame_index >= frame_count:
            break

    writer.release()
    cap.release()
    return transcode_to_browser_mp4(output, output.with_name(f"{output.stem}_h264.mp4"))
