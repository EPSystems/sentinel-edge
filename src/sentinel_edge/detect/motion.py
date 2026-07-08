"""Motion gate — the single biggest capacity lever (buildspec §7.1).

Runs before every inference: a frame reaches the detector only if the moving-
pixel fraction on a downscaled grey frame exceeds the threshold. Expected to
suppress >80% of inference calls on an idle scene; every camera-per-device
figure in the plan assumes it.

Pure-numpy running-average background subtraction so the gate works on every
build; if OpenCV is installed the MOG2 variant can be selected per profile.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class MotionGate:
    def __init__(
        self,
        downscale: int = 4,
        pixel_thresh: int = 25,
        min_fraction: float = 0.003,
        learn_rate: float = 0.05,
    ):
        self.downscale = max(1, downscale)
        self.pixel_thresh = pixel_thresh
        self.min_fraction = min_fraction
        self.learn_rate = learn_rate
        self._background: Optional[np.ndarray] = None
        self.last_fraction: float = 0.0
        self.frames_seen = 0
        self.frames_passed = 0

    def _grey_small(self, frame_bgr: np.ndarray) -> np.ndarray:
        small = frame_bgr[:: self.downscale, :: self.downscale]
        return small.mean(axis=2).astype(np.float32) if small.ndim == 3 else small.astype(np.float32)

    def process(self, frame_bgr: np.ndarray) -> bool:
        """True -> frame has motion, run detection. The first frame always
        passes (no background yet; better one extra inference than a missed
        event on startup)."""
        self.frames_seen += 1
        grey = self._grey_small(frame_bgr)
        if self._background is None or self._background.shape != grey.shape:
            self._background = grey.copy()
            self.last_fraction = 1.0
            self.frames_passed += 1
            return True
        diff = np.abs(grey - self._background)
        moving = diff > self.pixel_thresh
        self.last_fraction = float(moving.mean())
        # learn slower where motion is happening so a stopped object is not
        # instantly absorbed into the background
        alpha = np.where(moving, self.learn_rate * 0.2, self.learn_rate)
        self._background += alpha * (grey - self._background)
        passed = self.last_fraction >= self.min_fraction
        if passed:
            self.frames_passed += 1
        return passed

    @property
    def suppression_rate(self) -> float:
        if self.frames_seen == 0:
            return 0.0
        return 1.0 - self.frames_passed / self.frames_seen
