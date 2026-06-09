from pathlib import Path

import cv2
import numpy as np

from src.visualizer import render_annotated_video


def test_render_annotated_video_writes_output(tmp_path: Path):
    video_path = tmp_path / "input.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (80, 60))
    for _ in range(5):
        writer.write(np.zeros((60, 80, 3), dtype=np.uint8))
    writer.release()

    output = render_annotated_video(
        video_path=video_path,
        tracked_frames=[
            {
                "frame_index": 0,
                "timestamp": 0.0,
                "detections": [{"bbox": [45, 20, 65, 50], "confidence": 0.9, "track_id": 1}],
            }
        ],
        restricted_polygon=[[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]],
        output_path=tmp_path / "annotated.mp4",
    )

    assert Path(output).exists()
