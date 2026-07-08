"""ByteTrack-style multi-object tracker behind our own Detection/Track types.

Implements the BYTE association strategy (Zhang et al., MIT): two-stage
greedy-IoU matching — confident detections first, then low-score detections
rescue unmatched tracks — with a TTL for lost tracks. This pure-numpy
implementation has no Kalman filter (constant-position prediction), which is
sufficient at 6-8 detect fps for the rule types we run; the vendored upstream
ByteTrack can be slotted in behind the same interface on accelerator builds.

Nothing outside this module may depend on tracker internals (buildspec §4.3).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from ..contracts import ObjectClass

BBox = tuple[float, float, float, float]  # x1, y1, x2, y2


@dataclass
class Detection:
    object_class: ObjectClass
    confidence: float
    bbox: BBox


@dataclass
class Track:
    track_id: int
    object_class: ObjectClass
    confidence: float
    bbox: BBox
    age_frames: int = 1
    history: deque = field(default_factory=lambda: deque(maxlen=90))  # (ts, cx, cy)
    _misses: int = 0
    _last_ts: float = 0.0

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def bottom_center(self) -> tuple[float, float]:
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)

    def anchor(self, kind: str) -> tuple[float, float]:
        return self.bottom_center if kind == "bottom" else self.centroid


def iou_matrix(a: Sequence[BBox], b: Sequence[BBox]) -> np.ndarray:
    if not a or not b:
        return np.zeros((len(a), len(b)))
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    x1 = np.maximum(a_arr[:, None, 0], b_arr[None, :, 0])
    y1 = np.maximum(a_arr[:, None, 1], b_arr[None, :, 1])
    x2 = np.minimum(a_arr[:, None, 2], b_arr[None, :, 2])
    y2 = np.minimum(a_arr[:, None, 3], b_arr[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a_arr[:, 2] - a_arr[:, 0]) * (a_arr[:, 3] - a_arr[:, 1])
    area_b = (b_arr[:, 2] - b_arr[:, 0]) * (b_arr[:, 3] - b_arr[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def _greedy_match(iou: np.ndarray, thresh: float) -> list[tuple[int, int]]:
    """Highest-IoU-first greedy assignment; simple and deterministic."""
    matches: list[tuple[int, int]] = []
    if iou.size == 0:
        return matches
    iou = iou.copy()
    while True:
        i, j = np.unravel_index(np.argmax(iou), iou.shape)
        if iou[i, j] < thresh:
            break
        matches.append((int(i), int(j)))
        iou[i, :] = -1
        iou[:, j] = -1
    return matches


class ByteTracker:
    def __init__(
        self,
        high_thresh: float = 0.5,
        low_thresh: float = 0.1,
        iou_thresh: float = 0.3,
        track_ttl_s: float = 2.0,
        history_len: int = 90,
    ):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.iou_thresh = iou_thresh
        self.track_ttl_s = track_ttl_s
        self.history_len = history_len
        self._tracks: list[Track] = []
        self._next_id = 1

    def update(self, detections: Sequence[Detection], ts: float) -> list[Track]:
        """Feed one frame's detections; returns currently-active tracks
        (matched this frame). Lost tracks are kept internally for ttl_s so a
        brief occlusion doesn't change the id."""
        # expire lost tracks BEFORE association — a track past its TTL must
        # not be resurrected with its old id by a new detection
        self._tracks = [t for t in self._tracks if ts - t._last_ts <= self.track_ttl_s]

        high = [d for d in detections if d.confidence >= self.high_thresh]
        low = [d for d in detections if self.low_thresh <= d.confidence < self.high_thresh]

        unmatched_tracks = list(self._tracks)
        active: list[Track] = []

        # stage 1: confident detections vs all tracks
        remaining_high = self._associate(unmatched_tracks, high, ts, active)
        # stage 2 (BYTE): low-score detections rescue still-unmatched tracks
        self._associate(unmatched_tracks, low, ts, active, spawn_new=False)

        # unmatched confident detections start new tracks
        for det in remaining_high:
            track = Track(
                track_id=self._next_id,
                object_class=det.object_class,
                confidence=det.confidence,
                bbox=det.bbox,
                history=deque(maxlen=self.history_len),
                _last_ts=ts,
            )
            cx, cy = track.centroid
            track.history.append((ts, cx, cy))
            self._next_id += 1
            self._tracks.append(track)
            active.append(track)

        for tr in unmatched_tracks:
            tr._misses += 1
        return active

    def _associate(
        self,
        unmatched_tracks: list[Track],
        dets: list[Detection],
        ts: float,
        active: list[Track],
        spawn_new: bool = True,
    ) -> list[Detection]:
        if not unmatched_tracks or not dets:
            return dets if spawn_new else []
        iou = iou_matrix([t.bbox for t in unmatched_tracks], [d.bbox for d in dets])
        matched_det_idx: set[int] = set()
        matched_track_idx: set[int] = set()
        for ti, di in _greedy_match(iou, self.iou_thresh):
            tr, det = unmatched_tracks[ti], dets[di]
            tr.bbox = det.bbox
            tr.confidence = det.confidence
            if det.object_class != ObjectClass.unknown:
                tr.object_class = det.object_class
            tr.age_frames += 1
            tr._misses = 0
            tr._last_ts = ts
            cx, cy = tr.centroid
            tr.history.append((ts, cx, cy))
            matched_det_idx.add(di)
            matched_track_idx.add(ti)
            active.append(tr)
        for ti in sorted(matched_track_idx, reverse=True):
            unmatched_tracks.pop(ti)
        return [d for i, d in enumerate(dets) if i not in matched_det_idx]
