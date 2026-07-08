import pytest

from sentinel_edge.rules import geometry as g

SQUARE = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]


def test_point_in_polygon_basics():
    assert g.point_in_polygon((5, 5), SQUARE)
    assert not g.point_in_polygon((15, 5), SQUARE)
    assert g.point_in_polygon((0, 5), SQUARE)  # boundary counts as inside


def test_denormalize():
    assert g.denormalize([[0.5, 0.5]], 1920, 1080) == [(960.0, 540.0)]


def test_crossing_direction_signs():
    # vertical segment pointing down-screen: A=(0,0) B=(0,10)
    a, b = (0.0, 0.0), (0.0, 10.0)
    assert g.crossing_direction((-5, 5), (5, 5), a, b) == "rl"
    assert g.crossing_direction((5, 5), (-5, 5), a, b) == "lr"


def test_crossing_only_on_segment_not_extension():
    a, b = (0.0, 0.0), (0.0, 10.0)
    assert g.crossing_direction((-5, 50), (5, 50), a, b) is None


def test_crossing_overshoot_tolerance():
    a, b = (0.0, 0.0), (0.0, 10.0)
    assert g.crossing_direction((-5, 10.4), (5, 10.4), a, b, overshoot=0.05) == "rl"
    assert g.crossing_direction((-5, 10.4), (5, 10.4), a, b, overshoot=0.0) is None


def test_landing_exactly_on_line_still_detected():
    a, b = (0.0, 0.0), (0.0, 10.0)
    # step ends exactly on the line, next step leaves it — at least one fires
    first = g.crossing_direction((-5, 5), (0, 5), a, b)
    second = g.crossing_direction((0, 5), (5, 5), a, b)
    assert "rl" in {first, second}


def test_slow_and_fast_movement_both_detected():
    a, b = (0.0, 0.0), (0.0, 10.0)
    assert g.crossing_direction((-0.5, 5), (0.5, 5), a, b) == "rl"   # 1px step
    assert g.crossing_direction((-500, 5), (500, 5), a, b) == "rl"  # 1000px jump


def test_parallel_movement_no_cross():
    a, b = (0.0, 0.0), (0.0, 10.0)
    assert g.crossing_direction((5, 1), (5, 9), a, b) is None


def test_degenerate_polygon_rejected():
    with pytest.raises(ValueError):
        g.validate_zone_polygon([[0.1, 0.1], [0.1, 0.1]])
    with pytest.raises(ValueError):
        g.validate_zone_polygon([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])  # collinear
    g.validate_zone_polygon([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])  # fine
