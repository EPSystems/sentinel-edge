"""RTSP ingest: one StreamDecoder per camera yielding timestamped BGR frames
at the configured detect_fps, surviving network blips and camera reboots.

PyAV is the primary path ([rtsp] extra); an FFmpeg-subprocess fallback sits
behind the same interface for cameras PyAV chokes on (buildspec §4.1) and
for images where PyAV isn't installed. RTSP-over-TCP is forced — UDP drops
too much on contended SMB LANs.

Credentials exist only in memory; every log line goes through redact_rtsp.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncIterator, Optional

import numpy as np

from ..config import redact_rtsp
from .reconnect import Backoff, StreamHealth

log = logging.getLogger(__name__)

Frame = tuple[float, float, np.ndarray]  # (monotonic_ts, wall_ts, frame_bgr)

OPEN_TIMEOUT_S = 10.0
READ_TIMEOUT_S = 5.0


class StreamDecoder:
    def __init__(self, camera_id: str, rtsp_url: str, detect_fps: int,
                 prefer_pyav: bool = True):
        self.camera_id = camera_id
        self._rtsp_url = rtsp_url
        self.detect_fps = max(1, detect_fps)
        self.prefer_pyav = prefer_pyav
        self.health = StreamHealth(camera_id=camera_id)
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    async def frames(self) -> AsyncIterator[Frame]:
        """Yields at most detect_fps frames/s; reconnects forever until
        stop() — a dead camera degrades health, it never kills the pipeline."""
        backoff = Backoff()
        redacted = redact_rtsp(self._rtsp_url)
        while not self._stop.is_set():
            self.health.state = "connecting" if self.health.reconnects == 0 else "reconnecting"
            try:
                async for frame in self._read_once():
                    backoff.reset()
                    yield frame
                    if self._stop.is_set():
                        return
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("stream %s (%s) dropped: %s", self.camera_id, redacted, e)
            if self._stop.is_set():
                return
            self.health.state = "reconnecting"
            self.health.reconnects += 1
            delay = backoff.next_delay()
            log.info("stream %s reconnecting in %.1fs", self.camera_id, delay)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass
        self.health.state = "down"

    async def _read_once(self) -> AsyncIterator[Frame]:
        if self.prefer_pyav and _pyav_available():
            async for f in self._read_pyav():
                yield f
        else:
            async for f in self._read_ffmpeg():
                yield f

    # ------------------------------------------------------------ PyAV path

    async def _read_pyav(self) -> AsyncIterator[Frame]:
        import av

        loop = asyncio.get_running_loop()
        min_interval = 1.0 / self.detect_fps
        watchdog_s = max(5.0, 3.0 * min_interval)

        def _open():
            return av.open(self._rtsp_url, options={
                "rtsp_transport": "tcp",
                "stimeout": str(int(OPEN_TIMEOUT_S * 1_000_000)),
                "max_delay": "500000",
            }, timeout=(OPEN_TIMEOUT_S, READ_TIMEOUT_S))

        container = await loop.run_in_executor(None, _open)
        try:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            decoder = container.decode(stream)
            last_emit = 0.0
            while not self._stop.is_set():
                frame = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: next(decoder, None)),
                    timeout=watchdog_s,
                )
                if frame is None:
                    raise EOFError("stream ended")
                mono = time.monotonic()
                if mono - last_emit < min_interval:
                    continue  # sample down to detect_fps; decode stays cheap
                last_emit = mono
                img = frame.to_ndarray(format="bgr24")
                self.health.frame_received(mono)
                yield mono, time.time(), img
        finally:
            await loop.run_in_executor(None, container.close)

    # ------------------------------------------------------------ FFmpeg path

    async def _read_ffmpeg(self) -> AsyncIterator[Frame]:
        """rawvideo over a pipe; resolution probed from the first frame via
        ffprobe. Fallback path — used when PyAV is unavailable."""
        w, h = await self._probe_resolution()
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-nostdin", "-loglevel", "error",
            "-rtsp_transport", "tcp",
            "-i", self._rtsp_url,
            "-vf", f"fps={self.detect_fps}",
            "-pix_fmt", "bgr24", "-f", "rawvideo", "-",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        frame_bytes = w * h * 3
        watchdog_s = max(5.0, 3.0 / self.detect_fps)
        try:
            while not self._stop.is_set():
                assert proc.stdout is not None
                data = await asyncio.wait_for(
                    proc.stdout.readexactly(frame_bytes), timeout=watchdog_s)
                mono = time.monotonic()
                img = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)
                self.health.frame_received(mono)
                yield mono, time.time(), img
        except asyncio.IncompleteReadError as e:
            raise EOFError("ffmpeg stream ended") from e
        finally:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()

    async def _probe_resolution(self) -> tuple[int, int]:
        proc = await asyncio.create_subprocess_exec(
            "ffprobe", "-v", "error", "-rtsp_transport", "tcp",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0", self._rtsp_url,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=OPEN_TIMEOUT_S + 5)
        try:
            w, h = out.decode().strip().split(",")[:2]
            return int(w), int(h)
        except ValueError as e:
            raise ConnectionError("could not probe stream resolution") from e


def _pyav_available() -> bool:
    try:
        import av  # noqa: F401
        return True
    except ImportError:
        return False
