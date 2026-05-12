from src.simple_tracker import SimpleTracker, bbox_iou


def test_bbox_iou_overlap():
    assert round(bbox_iou([0, 0, 10, 10], [5, 5, 15, 15]), 4) == 0.1429


def test_tracker_keeps_id_for_overlapping_person():
    tracker = SimpleTracker(iou_threshold=0.2)
    first = tracker.update([{"bbox": [0, 0, 10, 10], "confidence": 0.9}], frame_index=0, timestamp=0.0)
    second = tracker.update([{"bbox": [1, 1, 11, 11], "confidence": 0.8}], frame_index=1, timestamp=0.5)
    assert first[0]["track_id"] == second[0]["track_id"]
