"""Aggregates per-camera + device state into the contract Heartbeat
(buildspec §9). This is the data source for the fleet/device-health view;
missing 3 heartbeats in a row flips the device red in the dashboard.
"""
from __future__ import annotations

import shutil
import time
from typing import Callable, Optional

from ..config import EdgeSettings
from ..contracts import (
    CONTRACT_VERSION,
    CameraHealth,
    DeviceHealth,
    Heartbeat,
    utcnow_rfc3339,
)
from ..streams import StreamHealth


class HealthCollector:
    def __init__(self, settings: EdgeSettings,
                 queue_depth: Callable[[], int],
                 queue_oldest_age: Callable[[], Optional[float]]):
        self.settings = settings
        self.queue_depth = queue_depth
        self.queue_oldest_age = queue_oldest_age
        self._cameras: dict[str, StreamHealth] = {}
        self.config_etag: str = ""

    def register_camera(self, health: StreamHealth) -> None:
        self._cameras[health.camera_id] = health

    def unregister_camera(self, camera_id: str) -> None:
        self._cameras.pop(camera_id, None)

    def event_recorded(self, camera_id: str) -> None:
        h = self._cameras.get(camera_id)
        if h:
            h.last_event_ts = time.monotonic()

    # ------------------------------------------------------------ build

    def build_heartbeat(self) -> Heartbeat:
        now_mono = time.monotonic()
        cameras = []
        any_down = False
        for h in self._cameras.values():
            state = {"streaming": "streaming", "connecting": "reconnecting",
                     "reconnecting": "reconnecting", "down": "down"}[h.state]
            if state != "streaming":
                any_down = True
            age = h.last_frame_age_s(now_mono)
            cameras.append(CameraHealth(
                camera_id=h.camera_id,
                state=state,  # type: ignore[arg-type]
                fps=round(h.fps, 2),
                last_frame_age_s=round(age, 1) if age != float("inf") else None,
                last_event_age_s=(round(now_mono - h.last_event_ts, 1)
                                  if h.last_event_ts else None),
            ))

        depth = self.queue_depth()
        oldest = self.queue_oldest_age()
        degraded = any_down or depth > 100
        device = DeviceHealth(
            status="degraded" if degraded else "online",
            cpu_pct=_cpu_pct(),
            mem_pct=_mem_pct(),
            disk_free_pct=_disk_free_pct(),
            queue_depth=depth,
            queue_oldest_age_s=round(oldest, 1) if oldest is not None else None,
        )
        return Heartbeat(
            device_id=self.settings.device_id,
            contract_version=CONTRACT_VERSION,
            model_version=self.settings.model_version,
            config_etag=self.config_etag,
            device=device,
            cameras=cameras,
            sent_at=utcnow_rfc3339(),
        )


def _cpu_pct() -> Optional[float]:
    try:
        import psutil
        return psutil.cpu_percent(interval=None)
    except ImportError:
        return None


def _mem_pct() -> Optional[float]:
    try:
        import psutil
        return psutil.virtual_memory().percent
    except ImportError:
        return None


def _disk_free_pct() -> Optional[float]:
    try:
        usage = shutil.disk_usage(".")
        return round(usage.free / usage.total * 100.0, 1)
    except OSError:
        return None
