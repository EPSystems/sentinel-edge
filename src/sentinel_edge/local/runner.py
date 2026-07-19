"""`sentinel-edge local` — the real pipeline (RTSP decode -> motion gate ->
detect -> track -> rules -> clips -> local queue) driven by a config file
instead of the cloud. No cloudlink, no publisher, no device credentials:
events accumulate in the local SQLite queue and are printed as they fire.

This is the bench harness for the first physical-camera test: point it at a
device-config JSON whose `rtsp_ref` is either a raw rtsp:// URL (dev
convenience) or a ref resolved via <data_dir>/secrets.json, walk across the
line, watch for `EVENT line_cross person ... queued`.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

from ..config import EdgeSettings, load_defaults
from ..contracts import DeviceConfig
from ..main import Supervisor

log = logging.getLogger(__name__)


async def run_local(config_path: Path, duration_s: Optional[float] = None,
                    preview_port: Optional[int] = None,
                    settings: Optional[EdgeSettings] = None) -> int:
    """Run until Ctrl+C (or duration_s). Returns the number of events queued
    during this run."""
    settings = settings or EdgeSettings()
    defaults = load_defaults(settings)
    config = DeviceConfig.model_validate_json(config_path.read_text(encoding="utf-8"))

    preview = None
    if preview_port is not None:
        from .preview import PreviewServer  # needs cv2 ([onnx] extra)
        preview = PreviewServer(preview_port, defaults)
        preview.start()
        log.info("preview at http://127.0.0.1:%d/ (open in a browser)", preview_port)

    sup = Supervisor(settings, defaults,
                     frame_tap=preview.push if preview else None)
    events_before = sup.queue.depth()
    try:
        await sup.inference.start()
        await sup.apply_config(config)
        if not sup._pipelines:
            raise SystemExit(
                "no camera pipeline started — every rtsp_ref failed to resolve. "
                "Use a raw rtsp:// URL in the config or add the ref to "
                f"{settings.data_dir / 'secrets.json'}")
        log.info("local mode: %d camera(s) running, backend=%s, "
                 "events -> %s (Ctrl+C to stop)",
                 len(sup._pipelines), defaults.backend,
                 settings.data_dir / "event-queue.sqlite3")
        try:
            if duration_s is not None:
                await asyncio.sleep(duration_s)
            else:
                await asyncio.Event().wait()  # forever; KeyboardInterrupt unwinds
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
    finally:
        line_counts = {
            pipeline.camera.name: pipeline.engine.line_counts()
            for pipeline in sup._pipelines.values() if pipeline.engine is not None
        }
        for camera_id in list(sup._pipelines):
            await sup._stop_pipeline(camera_id)
        await sup.inference.stop()
        if preview:
            preview.stop()
        produced = sup.queue.depth() - events_before
        _print_summary(sup, produced, line_counts)
        sup.queue.close()
    return produced


def _print_summary(sup: Supervisor, produced: int,
                   line_counts: Optional[dict] = None) -> None:
    print(f"\nlocal run done: {produced} new event(s) queued "
          f"({sup.queue.depth()} pending total, clips in the spool dir)")
    for item in sup.queue.next_batch(limit=10):
        payload = json.loads(item["payload_json"])
        print(f"  - {payload['rule_type']}: {payload['object_class']} "
              f"zone={payload.get('zone_id')} at {payload['occurred_at']}")
    for camera_name, zones in (line_counts or {}).items():
        for zone in zones.values():
            if zone["total"] == 0:
                continue
            print(f"  line count [{camera_name} / {zone['name']}]: "
                  f"lr={zone['lr']} rl={zone['rl']} total={zone['total']}")
