import pytest

from sentinel_edge.rules import SpeedEstimator


def test_px_per_m_scalar():
    est = SpeedEstimator({"px_per_m": 10.0}, window_s=1.0)
    # 100 px/s = 10 m/s = 36 km/h
    speeds = [est.update(1, t * 0.25, (t * 0.25 * 100.0, 0.0)) for t in range(9)]
    final = [s for s in speeds if s is not None][-1]
    assert final == pytest.approx(36.0, rel=0.05)


def test_homography_scaling():
    h = [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 1]]  # = 10 px per metre
    est = SpeedEstimator({"homography": h}, window_s=1.0)
    speeds = [est.update(1, t * 0.25, (t * 0.25 * 100.0, 0.0)) for t in range(9)]
    final = [s for s in speeds if s is not None][-1]
    assert final == pytest.approx(36.0, rel=0.05)


def test_uncalibrated_camera_returns_none():
    est = SpeedEstimator(None)
    assert not est.calibrated
    assert est.update(1, 0.0, (0, 0)) is None
    assert est.update(1, 1.0, (500, 0)) is None


def test_bad_homography_rejected():
    with pytest.raises(ValueError):
        SpeedEstimator({"homography": [[1, 0], [0, 1]]})


def test_prune_drops_dead_tracks():
    est = SpeedEstimator({"px_per_m": 10.0})
    est.update(1, 0.0, (0, 0))
    est.update(2, 0.0, (0, 0))
    est.prune({2})
    assert 1 not in est._history and 2 in est._history
