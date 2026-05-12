from src.event_detector import EventDetector, point_in_polygon


def test_point_in_polygon():
    polygon = [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert point_in_polygon([5, 5], polygon)
    assert not point_in_polygon([15, 5], polygon)


def test_detects_restricted_zone_intrusion():
    detector = EventDetector(
        frame_width=100,
        frame_height=100,
        restricted_polygon=[[0.5, 0.5], [1.0, 0.5], [1.0, 1.0], [0.5, 1.0]],
        loitering_enabled=False,
    )
    tracks = [
        {
            "track_id": 1,
            "history": [
                {"timestamp": 0.0, "bbox": [10, 10, 20, 20], "center": [15, 15]},
                {"timestamp": 1.0, "bbox": [60, 60, 70, 70], "center": [65, 65]},
            ],
        }
    ]
    events = detector.detect(tracks)
    assert len(events) == 1
    assert events[0]["type"] == "restricted_zone_intrusion"


def test_detects_loitering():
    detector = EventDetector(
        frame_width=100,
        frame_height=100,
        restricted_enabled=False,
        loitering_min_duration=5.0,
        loitering_max_movement_ratio=0.2,
    )
    tracks = [
        {
            "track_id": 7,
            "history": [
                {"timestamp": 0.0, "bbox": [10, 10, 20, 20], "center": [15, 15]},
                {"timestamp": 6.0, "bbox": [12, 12, 22, 22], "center": [17, 17]},
            ],
        }
    ]
    assert detector.detect(tracks)[0]["type"] == "loitering"
