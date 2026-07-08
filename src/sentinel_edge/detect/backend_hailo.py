"""Hailo-8/8L backend (HailoRT, compiled .hef artifact).

The standard <=6-camera profile (buildspec §6.1). HailoRT is installed
system-side on the device image, never via pip — hence the lazy import and
the explicit error message when it is absent.

The pre/post-processing mirrors backend_onnx (same YOLOX export geometry,
INT8-quantized at compile time per §5.2); only the runtime differs.
"""
from __future__ import annotations

import numpy as np

from ..contracts import ObjectClass
from ..track import Detection
from .backend_base import Detector, DetectorError


class HailoDetector(Detector):
    def __init__(self, defaults):
        super().__init__(defaults)
        self._device = None
        self._infer_model = None

    def load(self) -> None:
        try:
            from hailo_platform import VDevice  # type: ignore[import-not-found]
        except ImportError as e:
            raise DetectorError(
                "HailoRT not present — this backend runs only on a provisioned "
                "Hailo device image (accel-hailo8 profile)") from e
        self._device = VDevice()
        self._infer_model = self._device.create_infer_model(self.defaults.model_path)
        self._infer_model.set_batch_size(1)

    def _infer_raw(self, frame_bgr: np.ndarray) -> list[Detection]:
        if self._infer_model is None:
            raise DetectorError("load() not called")
        # HailoRT NMS-on-chip postprocess: output is per-class [ymin, xmin,
        # ymax, xmax, score] in normalized coords when the HEF is compiled
        # with the NMS layer (our §5.2 pipeline compiles it in).
        h, w = frame_bgr.shape[:2]
        with self._infer_model.configure() as configured:
            bindings = configured.create_bindings()
            bindings.input().set_buffer(self._preprocess(frame_bgr))
            configured.run([bindings], timeout=2000)
            raw = bindings.output().get_buffer()
        dets: list[Detection] = []
        for cls_idx, rows in enumerate(raw):
            mapped = self.class_map.get(cls_idx, ObjectClass.unknown)
            for ymin, xmin, ymax, xmax, score in rows:
                dets.append(Detection(
                    object_class=mapped,
                    confidence=float(score),
                    bbox=(float(xmin) * w, float(ymin) * h,
                          float(xmax) * w, float(ymax) * h),
                ))
        return dets

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        import cv2
        size = self.defaults.model_input_size
        return cv2.resize(frame_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
