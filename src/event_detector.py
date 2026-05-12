from __future__ import annotations

import math
from typing import Any


def point_in_polygon(point: tuple[float, float] | list[float], polygon: list[list[float]]) -> bool:
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi)
        if intersects:
            inside = not inside
        j = i
    return inside


def _scale_polygon(normalized_polygon: list[list[float]], width: int, height: int) -> list[list[float]]:
    return [[float(x) * width, float(y) * height] for x, y in normalized_polygon]


def _movement_ratio(history: list[dict[str, Any]], width: int, height: int) -> float:
    if len(history) < 2:
        return 0.0
    centers = [item["center"] for item in history]
    max_distance = 0.0
    for i, first in enumerate(centers):
        for second in centers[i + 1 :]:
            max_distance = max(max_distance, math.dist(first, second))
    return max_distance / max(1.0, min(width, height))


class EventDetector:
    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        restricted_polygon: list[list[float]] | None = None,
        restricted_enabled: bool = True,
        loitering_enabled: bool = True,
        loitering_min_duration: float = 8.0,
        loitering_max_movement_ratio: float = 0.10,
    ) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.restricted_polygon = _scale_polygon(restricted_polygon or [], frame_width, frame_height)
        self.restricted_enabled = restricted_enabled and bool(self.restricted_polygon)
        self.loitering_enabled = loitering_enabled
        self.loitering_min_duration = loitering_min_duration
        self.loitering_max_movement_ratio = loitering_max_movement_ratio

    def detect(self, tracks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for track in tracks:
            history = track.get("history", [])
            if not history:
                continue
            track_id = track["track_id"]

            if self.restricted_enabled:
                zone_hits = [item for item in history if point_in_polygon(item["center"], self.restricted_polygon)]
                if zone_hits:
                    first, last = zone_hits[0], zone_hits[-1]
                    events.append(
                        {
                            "event_id": f"restricted_zone_track_{track_id}",
                            "type": "restricted_zone_intrusion",
                            "type_cn": "禁区闯入",
                            "track_id": track_id,
                            "start_time": round(first["timestamp"], 3),
                            "end_time": round(last["timestamp"], 3),
                            "severity": "high",
                            "description": f"人员 track_id={track_id} 进入预设禁区",
                            "bbox": first["bbox"],
                        }
                    )

            duration = history[-1]["timestamp"] - history[0]["timestamp"]
            movement = _movement_ratio(history, self.frame_width, self.frame_height)
            if self.loitering_enabled and duration >= self.loitering_min_duration and movement <= self.loitering_max_movement_ratio:
                events.append(
                    {
                        "event_id": f"loitering_track_{track_id}",
                        "type": "loitering",
                        "type_cn": "长时间徘徊",
                        "track_id": track_id,
                        "start_time": round(history[0]["timestamp"], 3),
                        "end_time": round(history[-1]["timestamp"], 3),
                        "severity": "medium",
                        "description": f"人员 track_id={track_id} 在局部区域停留 {duration:.1f} 秒，移动幅度较小",
                        "bbox": history[-1]["bbox"],
                    }
                )
        return sorted(events, key=lambda item: (item["start_time"], item["track_id"], item["type"]))
