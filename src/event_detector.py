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


def _bbox_keypoints(bbox: list[float]) -> list[list[float]]:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return [
        [x1, y1],
        [cx, y1],
        [x2, y1],
        [x1, cy],
        [cx, cy],
        [x2, cy],
        [x1, y2],
        [cx, y2],
        [x2, y2],
    ]


def _point_in_bbox(point: list[float], bbox: list[float]) -> bool:
    x, y = point
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2)


def bbox_overlaps_polygon(
    bbox: list[float],
    polygon: list[list[float]],
    min_keypoint_ratio: float = 0.10,
) -> bool:
    if not bbox or len(bbox) < 4 or not polygon:
        return False
    keypoints = _bbox_keypoints(bbox)
    inside_count = sum(1 for point in keypoints if point_in_polygon(point, polygon))
    keypoint_ratio = inside_count / len(keypoints)
    polygon_point_inside_bbox = any(_point_in_bbox(point, bbox) for point in polygon)
    return keypoint_ratio >= min_keypoint_ratio or polygon_point_inside_bbox


class EventDetector:
    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        restricted_polygon: list[list[float]] | None = None,
        restricted_enabled: bool = True,
        restricted_min_bbox_points_ratio: float = 0.10,
        loitering_enabled: bool = True,
        loitering_min_duration: float = 8.0,
        loitering_max_movement_ratio: float = 0.10,
    ) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.restricted_polygon = _scale_polygon(restricted_polygon or [], frame_width, frame_height)
        self.restricted_enabled = restricted_enabled and bool(self.restricted_polygon)
        self.restricted_min_bbox_points_ratio = restricted_min_bbox_points_ratio
        self.loitering_enabled = loitering_enabled
        self.loitering_min_duration = loitering_min_duration
        self.loitering_max_movement_ratio = loitering_max_movement_ratio

    def detect(self, tracks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        pending_events: list[dict[str, Any]] = []
        for track in tracks:
            history = track.get("history", [])
            if not history:
                continue

            if self.restricted_enabled:
                zone_hits = [
                    item
                    for item in history
                    if bbox_overlaps_polygon(
                        item.get("bbox", []),
                        self.restricted_polygon,
                        self.restricted_min_bbox_points_ratio,
                    )
                ]
                if zone_hits:
                    first, last = zone_hits[0], zone_hits[-1]
                    pending_events.append(
                        {
                            "type": "restricted_zone_intrusion",
                            "type_cn": "禁区闯入",
                            "start_time": round(first["timestamp"], 3),
                            "end_time": round(last["timestamp"], 3),
                            "severity": "high",
                            "description": "检测到人员进入预设禁区",
                            "bbox": first["bbox"],
                        }
                    )

            duration = history[-1]["timestamp"] - history[0]["timestamp"]
            movement = _movement_ratio(history, self.frame_width, self.frame_height)
            if self.loitering_enabled and duration >= self.loitering_min_duration and movement <= self.loitering_max_movement_ratio:
                pending_events.append(
                    {
                        "type": "loitering",
                        "type_cn": "长时间徘徊",
                        "start_time": round(history[0]["timestamp"], 3),
                        "end_time": round(history[-1]["timestamp"], 3),
                        "severity": "medium",
                        "description": f"检测到人员在局部区域停留 {duration:.1f} 秒，移动幅度较小",
                        "bbox": history[-1]["bbox"],
                    }
                )
        events = sorted(pending_events, key=lambda item: (item["start_time"], item["type"]))
        counters: dict[str, int] = {}
        for event in events:
            event_type = str(event["type"])
            prefix = "restricted_zone_event" if event_type == "restricted_zone_intrusion" else f"{event_type}_event"
            counters[prefix] = counters.get(prefix, 0) + 1
            event["event_id"] = f"{prefix}_{counters[prefix]}"
        return events
