"""Per-camera rolling pre-event frame buffer (buildspec §4.6).

Holds ~clip_pre_s + 1s of decoded frames, memory-bounded by time window.
Frames are stored decoded (BGR ndarray) — at detect-fps sampling (6-8 fps)
a 1080p pre-buffer is ~6 MB per camera, cheaper than a GOP-aligned encoded
buffer and much simpler to mux correctly.
"""
from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np


class RingBuffer:
    def __init__(self, window_s: float):
        self.window_s = window_s
        self._frames: deque[tuple[float, np.ndarray]] = deque()

    def push(self, mono_ts: float, frame: np.ndarray) -> None:
        self._frames.append((mono_ts, frame))
        cutoff = mono_ts - self.window_s
        while self._frames and self._frames[0][0] < cutoff:
            self._frames.popleft()

    def snapshot(self, since_ts: float) -> list[tuple[float, np.ndarray]]:
        """Frames with ts >= since_ts, oldest first. Copies the deque (not
        the frames) so a concurrent event on another task can't tear it."""
        return [(ts, f) for ts, f in list(self._frames) if ts >= since_ts]

    def __len__(self) -> int:
        return len(self._frames)

    def frames(self) -> Iterable[tuple[float, np.ndarray]]:
        return list(self._frames)
