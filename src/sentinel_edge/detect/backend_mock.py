"""Mock detector for tests and the `simulate` mode: finds the brightest
moving blob in the frame — which is exactly what the synthetic source draws —
so the full pipeline (gate -> detect -> track -> rules -> record -> publish)
runs end-to-end with zero model weights and zero accelerator.
"""
from __future__ import annotations

import numpy as np

from ..contracts import ObjectClass
from ..track import Detection
from .backend_base import Detector


class MockDetector(Detector):
    def __init__(self, defaults, emit_class: ObjectClass = ObjectClass.person,
                 brightness_thresh: int = 200):
        super().__init__(defaults)
        self.emit_class = emit_class
        self.brightness_thresh = brightness_thresh

    def load(self) -> None:  # nothing to load
        pass

    def _infer_raw(self, frame_bgr: np.ndarray) -> list[Detection]:
        grey = frame_bgr.mean(axis=2) if frame_bgr.ndim == 3 else frame_bgr
        mask = grey > self.brightness_thresh
        if not mask.any():
            return []
        ys, xs = np.nonzero(mask)
        return [Detection(
            object_class=self.emit_class,
            confidence=0.9,
            bbox=(float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)),
        )]
