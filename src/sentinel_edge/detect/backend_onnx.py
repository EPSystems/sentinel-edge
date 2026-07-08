"""ONNXRuntime backend running an exported YOLOX model.

Bench/dev fallback — NOT sold (buildspec §4.2). Requires the [onnx] extra
(onnxruntime + opencv-headless); both are lazy-imported.

YOLOX export convention: raw head output [1, N, 5 + num_classes] with
grid-relative xywh — decoded here against strides 8/16/32, then class-aware
NMS in numpy.
"""
from __future__ import annotations

import numpy as np

from ..contracts import ObjectClass
from ..track import Detection
from .backend_base import Detector, DetectorError


class OnnxDetector(Detector):
    STRIDES = (8, 16, 32)

    def __init__(self, defaults):
        super().__init__(defaults)
        self._session = None
        self._input_name = ""
        self._grids: np.ndarray | None = None
        self._expanded_strides: np.ndarray | None = None

    def load(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise DetectorError(
                "onnxruntime not installed — pip install 'sentinel-edge[onnx]'") from e
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        available = ort.get_available_providers()
        self._session = ort.InferenceSession(
            self.defaults.model_path,
            providers=[p for p in providers if p in available] or ["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        self._build_grids(self.defaults.model_input_size)

    def _build_grids(self, size: int) -> None:
        grids, strides = [], []
        for s in self.STRIDES:
            n = size // s
            xv, yv = np.meshgrid(np.arange(n), np.arange(n))
            grid = np.stack((xv, yv), 2).reshape(-1, 2)
            grids.append(grid)
            strides.append(np.full((grid.shape[0], 1), s))
        self._grids = np.concatenate(grids, axis=0).astype(np.float32)
        self._expanded_strides = np.concatenate(strides, axis=0).astype(np.float32)

    def _preprocess(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        import cv2
        size = self.defaults.model_input_size
        h, w = frame_bgr.shape[:2]
        ratio = min(size / h, size / w)
        resized = cv2.resize(frame_bgr, (int(w * ratio), int(h * ratio)),
                             interpolation=cv2.INTER_LINEAR)
        padded = np.full((size, size, 3), 114, dtype=np.uint8)
        padded[: resized.shape[0], : resized.shape[1]] = resized
        img = padded.transpose(2, 0, 1)[None].astype(np.float32)
        return img, ratio

    def _infer_raw(self, frame_bgr: np.ndarray) -> list[Detection]:
        if self._session is None:
            raise DetectorError("load() not called")
        img, ratio = self._preprocess(frame_bgr)
        out = self._session.run(None, {self._input_name: img})[0][0]  # [N, 5+C]
        return self._decode(out, ratio)

    def _decode(self, pred: np.ndarray, ratio: float) -> list[Detection]:
        assert self._grids is not None and self._expanded_strides is not None
        xy = (pred[:, :2] + self._grids) * self._expanded_strides
        wh = np.exp(pred[:, 2:4]) * self._expanded_strides
        obj = pred[:, 4:5]
        cls_scores = pred[:, 5:]
        scores = obj * cls_scores
        cls_ids = scores.argmax(axis=1)
        confs = scores[np.arange(len(cls_ids)), cls_ids]
        keep = confs >= self.defaults.track_low_thresh
        if not keep.any():
            return []
        xy, wh, cls_ids, confs = xy[keep], wh[keep], cls_ids[keep], confs[keep]
        boxes = np.concatenate([xy - wh / 2, xy + wh / 2], axis=1) / ratio
        keep_idx = _nms_per_class(boxes, confs, cls_ids, iou_thresh=0.45)
        dets = []
        for i in keep_idx:
            mapped = self.class_map.get(int(cls_ids[i]), ObjectClass.unknown)
            dets.append(Detection(
                object_class=mapped,
                confidence=float(confs[i]),
                bbox=tuple(float(v) for v in boxes[i]),
            ))
        return dets


def _nms_per_class(boxes: np.ndarray, scores: np.ndarray, cls_ids: np.ndarray,
                   iou_thresh: float) -> list[int]:
    keep: list[int] = []
    for c in np.unique(cls_ids):
        idx = np.where(cls_ids == c)[0]
        keep.extend(idx[i] for i in _nms(boxes[idx], scores[idx], iou_thresh))
    return keep


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
    order = scores.argsort()[::-1]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thresh]
    return keep
