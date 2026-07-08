"""Jetson Orin backend (TensorRT serialized .engine).

Engines are device/JetPack-specific and built ON the target Jetson
(buildspec §5.2) — this module only loads and runs them. TensorRT and
PyCUDA come from JetPack, never pip; lazy-imported with a clear error.

Decode path is shared with backend_onnx: the engine is built from the same
YOLOX ONNX export, so the raw head output has identical geometry.
"""
from __future__ import annotations

import numpy as np

from ..track import Detection
from .backend_base import Detector, DetectorError
from .backend_onnx import OnnxDetector


class TensorRTDetector(Detector):
    def __init__(self, defaults):
        super().__init__(defaults)
        self._engine = None
        self._context = None
        self._io = None
        # reuse the ONNX decode/NMS by delegating the numpy postprocess
        self._decoder = OnnxDetector(defaults)
        self._decoder._build_grids(defaults.model_input_size)

    def load(self) -> None:
        try:
            import tensorrt as trt  # type: ignore[import-not-found]
            import pycuda.autoinit  # noqa: F401  type: ignore[import-not-found]
            import pycuda.driver as cuda  # type: ignore[import-not-found]
        except ImportError as e:
            raise DetectorError(
                "TensorRT/PyCUDA not present — this backend runs only on a "
                "provisioned Jetson image (accel-jetson-orin profile)") from e
        logger = trt.Logger(trt.Logger.WARNING)
        with open(self.defaults.model_path, "rb") as f, trt.Runtime(logger) as runtime:
            self._engine = runtime.deserialize_cuda_engine(f.read())
        self._context = self._engine.create_execution_context()
        size = self.defaults.model_input_size
        in_shape = (1, 3, size, size)
        out_shape = tuple(self._engine.get_tensor_shape(self._engine.get_tensor_name(1)))
        self._io = {
            "cuda": cuda,
            "d_in": cuda.mem_alloc(int(np.prod(in_shape)) * 4),
            "d_out": cuda.mem_alloc(int(np.prod(out_shape)) * 4),
            "h_out": np.empty(out_shape, dtype=np.float32),
            "stream": cuda.Stream(),
        }

    def _infer_raw(self, frame_bgr: np.ndarray) -> list[Detection]:
        if self._context is None or self._io is None:
            raise DetectorError("load() not called")
        img, ratio = self._decoder._preprocess(frame_bgr)
        io, cuda = self._io, self._io["cuda"]
        cuda.memcpy_htod_async(io["d_in"], np.ascontiguousarray(img), io["stream"])
        self._context.execute_v2([int(io["d_in"]), int(io["d_out"])])
        cuda.memcpy_dtoh_async(io["h_out"], io["d_out"], io["stream"])
        io["stream"].synchronize()
        return self._decoder._decode(io["h_out"][0], ratio)
