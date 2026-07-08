"""Backoff policy + stream health, shared by streams and cloudlink."""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Literal

StreamState = Literal["connecting", "streaming", "reconnecting", "down"]


class Backoff:
    """Exponential 1s -> 2s -> 4s ... capped at 30s, jittered ±20%
    (buildspec §4.1 step 4)."""

    def __init__(self, base: float = 1.0, cap: float = 30.0, jitter: float = 0.2):
        self.base = base
        self.cap = cap
        self.jitter = jitter
        self._attempt = 0

    def next_delay(self) -> float:
        delay = min(self.base * (2 ** self._attempt), self.cap)
        self._attempt += 1
        spread = delay * self.jitter
        return max(0.05, delay + random.uniform(-spread, spread))

    def reset(self) -> None:
        self._attempt = 0


@dataclass
class StreamHealth:
    camera_id: str
    state: StreamState = "connecting"
    fps: float = 0.0
    last_frame_ts: float = 0.0
    last_event_ts: float = 0.0
    reconnects: int = 0
    frames_total: int = 0

    def frame_received(self, mono_ts: float) -> None:
        if self.frames_total > 0 and self.last_frame_ts:
            dt = mono_ts - self.last_frame_ts
            if dt > 0:
                inst = 1.0 / dt
                self.fps = inst if self.fps == 0 else 0.9 * self.fps + 0.1 * inst
        self.last_frame_ts = mono_ts
        self.frames_total += 1
        self.state = "streaming"

    def last_frame_age_s(self, now_mono: float | None = None) -> float:
        now = time.monotonic() if now_mono is None else now_mono
        return now - self.last_frame_ts if self.last_frame_ts else float("inf")
