from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


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
