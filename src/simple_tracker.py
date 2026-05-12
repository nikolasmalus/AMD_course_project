from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def bbox_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def bbox_center(bbox: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


@dataclass
class Track:
    track_id: int
    bbox: list[float]
    last_seen_frame: int
    missing_frames: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    def update(self, bbox: list[float], frame_index: int, timestamp: float, confidence: float) -> None:
        self.bbox = bbox
        self.last_seen_frame = frame_index
        self.missing_frames = 0
        cx, cy = bbox_center(bbox)
        self.history.append(
            {
                "frame_index": frame_index,
                "timestamp": float(timestamp),
                "bbox": [float(v) for v in bbox],
                "center": [float(cx), float(cy)],
                "confidence": float(confidence),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "bbox": self.bbox,
            "last_seen_frame": self.last_seen_frame,
            "missing_frames": self.missing_frames,
            "history": self.history,
        }


class SimpleTracker:
    def __init__(self, iou_threshold: float = 0.3, max_missing_frames: int = 8) -> None:
        self.iou_threshold = iou_threshold
        self.max_missing_frames = max_missing_frames
        self.next_track_id = 1
        self.tracks: dict[int, Track] = {}
        self.finished_tracks: list[Track] = []

    def update(self, detections: list[dict[str, Any]], frame_index: int, timestamp: float) -> list[dict[str, Any]]:
        unmatched_track_ids = set(self.tracks.keys())
        tracked_detections: list[dict[str, Any]] = []

        for detection in sorted(detections, key=lambda item: item.get("confidence", 0.0), reverse=True):
            bbox = detection["bbox"]
            best_track_id = None
            best_iou = 0.0
            for track_id in unmatched_track_ids:
                score = bbox_iou(bbox, self.tracks[track_id].bbox)
                if score > best_iou:
                    best_iou = score
                    best_track_id = track_id

            if best_track_id is not None and best_iou >= self.iou_threshold:
                track = self.tracks[best_track_id]
                unmatched_track_ids.remove(best_track_id)
            else:
                track = Track(track_id=self.next_track_id, bbox=bbox, last_seen_frame=frame_index)
                self.tracks[track.track_id] = track
                self.next_track_id += 1

            track.update(bbox, frame_index, timestamp, detection.get("confidence", 0.0))
            tracked_detections.append({**detection, "track_id": track.track_id, "track_iou": best_iou})

        for track_id in list(unmatched_track_ids):
            self.tracks[track_id].missing_frames += 1
            if self.tracks[track_id].missing_frames > self.max_missing_frames:
                self.finished_tracks.append(self.tracks.pop(track_id))

        return tracked_detections

    def all_tracks(self) -> list[dict[str, Any]]:
        all_items = [*self.finished_tracks, *self.tracks.values()]
        return [track.to_dict() for track in sorted(all_items, key=lambda item: item.track_id)]


def track_frame_results(
    frame_results: list[dict[str, Any]],
    iou_threshold: float = 0.3,
    max_missing_frames: int = 8,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tracker = SimpleTracker(iou_threshold=iou_threshold, max_missing_frames=max_missing_frames)
    tracked_frames = []
    for frame in frame_results:
        tracked = tracker.update(frame.get("detections", []), frame["frame_index"], frame["timestamp"])
        tracked_frames.append({**frame, "detections": tracked})
    return tracked_frames, tracker.all_tracks()
