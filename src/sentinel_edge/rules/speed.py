"""Speed estimation: ByteTrack displacement x per-camera calibration -> km/h.

Monocular speed is approximate (+-20%), deterrent-grade, not court-grade —
that caveat must surface wherever a speed value is shown (CANONICAL-FACTS §6).

Calibration (cameras.calibration jsonb, pushed in config):
- {"px_per_m": 11.4}                       flat scalar (entry-level)
- {"homography": [[..],[..],[..]]}         3x3 image->ground-plane (metres)
"""
from __future__ import annotations

from collections import deque
from typing import Any, Optional

import numpy as np


class SpeedEstimator:
    def __init__(
        self,
        calibration: Optional[dict[str, Any]],
        window_s: float = 1.0,
        smooth_alpha: float = 0.4,
    ):
        self.window_s = window_s
        self.smooth_alpha = smooth_alpha
        self._h: Optional[np.ndarray] = None
        self._px_per_m: Optional[float] = None
        if calibration:
            if calibration.get("homography") is not None:
                h = np.asarray(calibration["homography"], dtype=float)
                if h.shape != (3, 3):
                    raise ValueError("homography must be 3x3")
                self._h = h
            elif calibration.get("px_per_m"):
                self._px_per_m = float(calibration["px_per_m"])
        self._history: dict[int, deque[tuple[float, np.ndarray]]] = {}
        self._smoothed: dict[int, float] = {}

    @property
    def calibrated(self) -> bool:
        return self._h is not None or self._px_per_m is not None

    def _to_ground(self, point_px: tuple[float, float]) -> np.ndarray:
        if self._h is not None:
            v = self._h @ np.array([point_px[0], point_px[1], 1.0])
            return v[:2] / v[2]
        assert self._px_per_m is not None
        return np.array(point_px, dtype=float) / self._px_per_m

    def update(self, track_id: int, ts: float, point_px: tuple[float, float]) -> Optional[float]:
        """Feed one anchor sample; returns the smoothed speed in km/h once the
        window has enough span, else None. Uncalibrated cameras return None."""
        if not self.calibrated:
            return None
        ground = self._to_ground(point_px)
        hist = self._history.setdefault(track_id, deque())
        hist.append((ts, ground))
        while hist and ts - hist[0][0] > self.window_s * 2:
            hist.popleft()
        t0, p0 = hist[0]
        span = ts - t0
        if span < self.window_s * 0.5:
            return None
        metres = float(np.hypot(*(ground - p0)))
        kmh = metres / span * 3.6
        prev = self._smoothed.get(track_id)
        smoothed = kmh if prev is None else self.smooth_alpha * kmh + (1 - self.smooth_alpha) * prev
        self._smoothed[track_id] = smoothed
        return smoothed

    def prune(self, live_track_ids: set[int]) -> None:
        for tid in list(self._history):
            if tid not in live_track_ids:
                self._history.pop(tid, None)
                self._smoothed.pop(tid, None)
