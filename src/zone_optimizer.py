from __future__ import annotations

import cv2
import numpy as np


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def sanitize_normalized_polygon(points: list[list[float]] | None) -> list[list[float]]:
    if not points:
        return []
    cleaned: list[list[float]] = []
    for point in points:
        if len(point) < 2:
            continue
        x = round(_clamp(float(point[0]), 0.0, 1.0), 6)
        y = round(_clamp(float(point[1]), 0.0, 1.0), 6)
        if not cleaned or cleaned[-1] != [x, y]:
            cleaned.append([x, y])
    if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
        cleaned.pop()
    return cleaned


def pixel_to_normalized(point: tuple[int, int] | list[int], width: int, height: int) -> list[float]:
    x = _clamp(float(point[0]), 0.0, max(float(width - 1), 0.0))
    y = _clamp(float(point[1]), 0.0, max(float(height - 1), 0.0))
    return [round(x / max(width, 1), 6), round(y / max(height, 1), 6)]


def normalized_to_pixel(points: list[list[float]] | None, width: int, height: int) -> np.ndarray:
    polygon = sanitize_normalized_polygon(points)
    return np.array([[int(x * width), int(y * height)] for x, y in polygon], dtype=np.int32)


def sort_polygon_points(points: list[list[float]] | None) -> list[list[float]]:
    polygon = sanitize_normalized_polygon(points)
    if len(polygon) < 3:
        return polygon
    arr = np.array(polygon, dtype=np.float32)
    center = arr.mean(axis=0)
    angles = np.arctan2(arr[:, 1] - center[1], arr[:, 0] - center[0])
    ordered = arr[np.argsort(angles)]
    return [[round(float(x), 6), round(float(y), 6)] for x, y in ordered.tolist()]


def optimize_polygon(points: list[list[float]] | None, width: int, height: int) -> list[list[float]]:
    polygon = sort_polygon_points(points)
    if len(polygon) < 3:
        return polygon

    pixel_points = normalized_to_pixel(polygon, width, height)
    hull = cv2.convexHull(pixel_points)
    perimeter = cv2.arcLength(hull, True)
    epsilon = max(2.0, 0.015 * perimeter)
    approx = cv2.approxPolyDP(hull, epsilon, True).reshape(-1, 2)

    if len(approx) < 3:
        approx = hull.reshape(-1, 2)

    normalized = [pixel_to_normalized([int(x), int(y)], width, height) for x, y in approx.tolist()]
    return sort_polygon_points(normalized)


def draw_zone_overlay(
    image: np.ndarray | None,
    points: list[list[float]] | None,
    *,
    optimized: bool = False,
) -> np.ndarray | None:
    if image is None:
        return None
    output = image.copy()
    height, width = output.shape[:2]
    polygon = sanitize_normalized_polygon(points)
    color = (36, 99, 235) if optimized else (239, 68, 68)

    if len(polygon) >= 2:
        pixel_points = normalized_to_pixel(polygon, width, height)
        cv2.polylines(output, [pixel_points], isClosed=len(polygon) >= 3, color=color, thickness=3)
        if len(polygon) >= 3:
            mask = output.copy()
            cv2.fillPoly(mask, [pixel_points], color=color)
            output = cv2.addWeighted(mask, 0.18, output, 0.82, 0)

    for index, point in enumerate(polygon, start=1):
        x, y = normalized_to_pixel([point], width, height)[0]
        cv2.circle(output, (int(x), int(y)), 6, color, -1)
        cv2.putText(output, str(index), (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    return output
