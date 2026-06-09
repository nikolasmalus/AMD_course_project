from src.zone_optimizer import optimize_polygon, pixel_to_normalized, sanitize_normalized_polygon


def test_pixel_to_normalized_clamps_to_frame_bounds():
    assert pixel_to_normalized((-10, 250), width=100, height=200) == [0.0, 0.995]


def test_sanitize_normalized_polygon_removes_closing_duplicate():
    polygon = sanitize_normalized_polygon([[0.1, 0.2], [1.5, -0.5], [0.1, 0.2]])
    assert polygon == [[0.1, 0.2], [1.0, 0.0]]


def test_optimize_polygon_returns_valid_ordered_polygon():
    optimized = optimize_polygon([[0.8, 0.8], [0.2, 0.2], [0.8, 0.2], [0.2, 0.8]], width=100, height=100)
    assert len(optimized) >= 3
    assert all(0.0 <= coordinate <= 1.0 for point in optimized for coordinate in point)
