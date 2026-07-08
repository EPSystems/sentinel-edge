"""Detector interface. The same pipeline code runs on Hailo, Jetson and ONNX
by swapping only the profile (buildspec §4.2 DoD); accelerator runtimes are
lazy-imported so this package imports cleanly everywhere.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from ..contracts import ObjectClass
from ..track import Detection

if TYPE_CHECKING:
    from ..config import PipelineDefaults


class Detector(ABC):
    """load() once, warmup() once, then infer() per motion-gated frame."""

    def __init__(self, defaults: "PipelineDefaults"):
        self.defaults = defaults
        self.class_map = {k: ObjectClass.coerce(v) for k, v in defaults.class_map.items()}
        self.last_latency_ms: float = 0.0

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def _infer_raw(self, frame_bgr: np.ndarray) -> list[Detection]: ...

    def warmup(self) -> None:
        self._infer_raw(np.zeros((360, 640, 3), dtype=np.uint8))

    def infer(self, frame_bgr: np.ndarray) -> list[Detection]:
        t0 = time.perf_counter()
        dets = self._infer_raw(frame_bgr)
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
        return self._filter(dets)

    def _filter(self, dets: list[Detection]) -> list[Detection]:
        """conf floor + class filter + min-object-size (far-field noise).
        The tracker gets detections down to its low threshold — ByteTrack
        needs low-confidence boxes for association — so the floor here is
        the tracker's low threshold, not the alerting conf_threshold."""
        floor = min(self.defaults.track_low_thresh, self.defaults.conf_threshold)
        out = []
        for d in dets:
            if d.confidence < floor:
                continue
            if d.object_class == ObjectClass.unknown:
                continue
            x1, y1, x2, y2 = d.bbox
            if min(x2 - x1, y2 - y1) < self.defaults.min_object_px:
                continue
            out.append(d)
        return out


class DetectorError(RuntimeError):
    pass


def get_backend(defaults: "PipelineDefaults") -> Detector:
    name = defaults.backend
    if name == "onnx":
        from .backend_onnx import OnnxDetector
        return OnnxDetector(defaults)
    if name == "hailo":
        from .backend_hailo import HailoDetector
        return HailoDetector(defaults)
    if name == "tensorrt":
        from .backend_tensorrt import TensorRTDetector
        return TensorRTDetector(defaults)
    if name == "mock":
        from .backend_mock import MockDetector
        return MockDetector(defaults)
    raise DetectorError(f"unknown detector backend {name!r}")
