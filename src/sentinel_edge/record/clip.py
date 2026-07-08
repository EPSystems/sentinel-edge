"""Clip recorder: on a rule firing, cut a ~10s MP4 (5s pre-roll from the ring
buffer + 5s post-roll) plus a JPEG thumbnail at the trigger frame, into a
size-capped local spool. Files leave the spool only after the publisher
confirms upload (or the offline queue takes ownership).

Encoding is FFmpeg via stdin rawvideo; the encoder is injectable so the
recorder logic is unit-testable without ffmpeg on the box.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Optional

import numpy as np

from ..contracts import uuid7
from .ringbuffer import RingBuffer

log = logging.getLogger(__name__)

# (frames, fps, out_path) -> None
Encoder = Callable[[list[np.ndarray], float, Path], Awaitable[None]]
# (frame, out_path) -> None
ThumbEncoder = Callable[[np.ndarray, Path], Awaitable[None]]


@dataclass
class ClipResult:
    clip_path: Path
    thumb_path: Optional[Path]
    duration_s: float


async def ffmpeg_encode(frames: list[np.ndarray], fps: float, out_path: Path) -> None:
    h, w = frames[0].shape[:2]
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
        "-r", f"{fps:.3f}", "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_path),
        stdin=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    assert proc.stdin is not None
    for f in frames:
        proc.stdin.write(np.ascontiguousarray(f).tobytes())
        await proc.stdin.drain()
    proc.stdin.close()
    rc = await proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg clip encode failed rc={rc}")


async def ffmpeg_thumb(frame: np.ndarray, out_path: Path) -> None:
    h, w = frame.shape[:2]
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
        "-i", "-", "-frames:v", "1", "-q:v", "4",
        str(out_path),
        stdin=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    assert proc.stdin is not None
    proc.stdin.write(np.ascontiguousarray(frame).tobytes())
    await proc.stdin.drain()
    proc.stdin.close()
    rc = await proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg thumb encode failed rc={rc}")


class ClipRecorder:
    """One per camera. Feed every decoded frame via push(); on an event call
    record(trigger_ts) — the pre-roll comes from the ring buffer, the
    post-roll from frames pushed during the await."""

    def __init__(
        self,
        camera_id: str,
        spool: "SpoolManager",
        fps_hint: float,
        pre_s: float = 5.0,
        post_s: float = 5.0,
        encoder: Encoder = ffmpeg_encode,
        thumb_encoder: ThumbEncoder = ffmpeg_thumb,
    ):
        self.camera_id = camera_id
        self.spool = spool
        self.fps_hint = max(1.0, fps_hint)
        self.pre_s = pre_s
        self.post_s = post_s
        self.encoder = encoder
        self.thumb_encoder = thumb_encoder
        self.ring = RingBuffer(window_s=pre_s + 1.0)

    def push(self, mono_ts: float, frame: np.ndarray) -> None:
        self.ring.push(mono_ts, frame)

    async def record(self, trigger_ts: float, micro: bool = False) -> ClipResult:
        """micro=True encodes a single-frame clip — the queue-backpressure
        degrade mode that keeps events contract-valid at minimal disk cost."""
        event_ref = uuid7()
        clip_path = self.spool.path_for(f"{event_ref}.mp4")
        thumb_path = self.spool.path_for(f"{event_ref}.jpg")

        pre = self.ring.snapshot(trigger_ts - self.pre_s)
        trigger_frame = pre[-1][1] if pre else None

        if not micro:
            deadline = trigger_ts + self.post_s
            while time.monotonic() < deadline:
                await asyncio.sleep(0.1)
            frames = [f for _, f in self.ring.snapshot(trigger_ts - self.pre_s)]
        else:
            frames = [trigger_frame] if trigger_frame is not None else []

        if not frames:
            raise RuntimeError("no frames available for clip")
        if trigger_frame is None:
            trigger_frame = frames[0]

        await self.encoder(frames, self.fps_hint, clip_path)
        try:
            await self.thumb_encoder(trigger_frame, thumb_path)
        except Exception:
            log.exception("thumbnail encode failed — event ships without one")
            thumb_path = None
        self.spool.register(clip_path)
        if thumb_path:
            self.spool.register(thumb_path)
        return ClipResult(clip_path=clip_path, thumb_path=thumb_path,
                          duration_s=len(frames) / self.fps_hint)


class SpoolManager:
    """Size-capped local clip spool. Eviction: oldest already-uploaded files
    first; files still owned by the offline queue are never evicted
    (buildspec §4.6 step 4 — disk never fills to failure)."""

    def __init__(self, spool_dir: Path, max_bytes: int):
        self.dir = Path(spool_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self._uploaded: set[Path] = set()
        self._known: list[Path] = []

    def path_for(self, name: str) -> Path:
        return self.dir / name

    def register(self, path: Path) -> None:
        self._known.append(path)
        self._enforce_cap()

    def mark_uploaded(self, path: Path) -> None:
        self._uploaded.add(path)
        try:
            path.unlink(missing_ok=True)
        except OSError:
            log.warning("could not delete uploaded clip %s", path)

    def total_bytes(self) -> int:
        return sum(p.stat().st_size for p in self._known if p.exists())

    def _enforce_cap(self) -> None:
        if self.total_bytes() <= self.max_bytes:
            return
        for path in list(self._known):
            if self.total_bytes() <= self.max_bytes:
                break
            if path in self._uploaded or not path.exists():
                self._known.remove(path)
                continue
        # still over cap with only queue-owned files left: alarm via health,
        # never delete un-uploaded evidence silently
        if self.total_bytes() > self.max_bytes:
            log.error("spool over cap (%.0f MB) with only pending clips — "
                      "publisher backlog; degrade mode should engage",
                      self.total_bytes() / 1e6)
