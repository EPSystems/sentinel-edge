"""Supervisor: one pipeline per camera, shared process-wide services
(cloudlink, publisher, inference). No business logic here beyond
spawn/monitor/restart (buildspec §2) — a crashed camera pipeline restarts
with backoff and never takes the device down.

CLI:
    sentinel-edge run                        # production: cloud-driven config
    sentinel-edge simulate [--config PATH]   # full loop on synthetic frames,
                                             # mock detector, no cloud needed
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import AsyncIterator, Optional

import numpy as np

from .cloudlink import CloudLink, LiveViewSignaling
from .config import ConfigCache, EdgeSettings, PipelineDefaults, SecretResolver, load_defaults
from .contracts import CameraConfig, DeviceConfig, EventPayload, RuleType, uuid7, utcnow_rfc3339
from .detect import Detector, MotionGate, get_backend
from .health import HealthCollector
from .publish import EventPublisher, EventQueue
from .record import ClipRecorder, SpoolManager
from .rules import DriveoffCorrelator, EventCandidate, RuleEngine, Signal
from .streams import StreamDecoder, StreamHealth
from .track import ByteTracker

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- inference

class InferenceService:
    """Serialises inference through one queue per device — the accelerator is
    the shared scarce resource; cameras are producers (buildspec §4.2)."""

    def __init__(self, detector: Detector, max_pending: int = 16):
        self.detector = detector
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_pending)
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.detector.load)
        await loop.run_in_executor(None, self.detector.warmup)
        self._task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    async def infer(self, frame: np.ndarray) -> list:
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        await self._queue.put((frame, fut))
        return await fut

    async def _worker(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            frame, fut = await self._queue.get()
            try:
                dets = await loop.run_in_executor(None, self.detector.infer, frame)
                if not fut.cancelled():
                    fut.set_result(dets)
            except Exception as e:
                if not fut.cancelled():
                    fut.set_exception(e)


# ---------------------------------------------------------------- pipeline

FrameSource = AsyncIterator[tuple[float, float, np.ndarray]]


class CameraPipeline:
    """decode -> motion gate -> detect -> track -> rules -> clip -> enqueue."""

    def __init__(self, camera: CameraConfig, defaults: PipelineDefaults,
                 inference: InferenceService, queue: EventQueue,
                 spool: SpoolManager, health: HealthCollector,
                 signal_sink: "Supervisor",
                 frame_source: Optional[FrameSource] = None,
                 decoder: Optional[StreamDecoder] = None):
        self.camera = camera
        self.defaults = defaults
        self.inference = inference
        self.queue = queue
        self.spool = spool
        self.health = health
        self.signal_sink = signal_sink
        self.frame_source = frame_source
        self.decoder = decoder
        self.stream_health = decoder.health if decoder else StreamHealth(camera.camera_id)
        self.gate = MotionGate(
            downscale=defaults.motion_downscale,
            pixel_thresh=defaults.motion_pixel_thresh,
            min_fraction=defaults.motion_min_fraction,
            learn_rate=defaults.motion_learn_rate,
        ) if defaults.motion_enabled else None
        self.tracker = ByteTracker(
            high_thresh=defaults.track_high_thresh,
            low_thresh=defaults.track_low_thresh,
            iou_thresh=defaults.track_iou_thresh,
            track_ttl_s=defaults.track_ttl_s,
            history_len=defaults.track_history_len,
        )
        self.recorder: Optional[ClipRecorder] = None
        self.engine: Optional[RuleEngine] = None
        self._pending_config: Optional[CameraConfig] = None
        self._tracks_alive_until: float = float("-inf")
        self._stop = asyncio.Event()
        self._clip_tasks: set[asyncio.Task] = set()

    def stop(self) -> None:
        self._stop.set()
        if self.decoder:
            self.decoder.stop()

    def update_config(self, camera: CameraConfig) -> None:
        """Hot-reload (Flow B): picked up atomically between frames."""
        self._pending_config = camera

    async def run(self) -> None:
        source = self.frame_source
        if source is None:
            assert self.decoder is not None
            source = self.decoder.frames()
        async for mono_ts, wall_ts, frame in source:
            if self._stop.is_set():
                break
            await self._process_frame(mono_ts, wall_ts, frame)
        if self._clip_tasks:
            await asyncio.gather(*self._clip_tasks, return_exceptions=True)

    async def _process_frame(self, mono_ts: float, wall_ts: float,
                             frame: np.ndarray) -> None:
        if self.frame_source is not None:  # decoder path updates health itself
            self.stream_health.frame_received(mono_ts)
        if self.engine is None or self.recorder is None:
            h, w = frame.shape[:2]
            self.engine = RuleEngine(self.camera, self.defaults, (w, h))
            self.recorder = ClipRecorder(
                camera_id=self.camera.camera_id, spool=self.spool,
                fps_hint=float(self.camera.detect_fps),
                pre_s=self.defaults.clip_pre_s, post_s=self.defaults.clip_post_s)
        if self._pending_config is not None:
            self.camera = self._pending_config
            self.engine.swap_config(self.camera)
            self._pending_config = None

        self.recorder.push(mono_ts, frame)

        # Always feed the gate (background model must keep learning), but a
        # quiet frame only skips detection when no track is alive: a loiterer
        # standing still generates no motion yet must keep accruing dwell.
        motion = self.gate.process(frame) if self.gate is not None else True
        if not motion and mono_ts > self._tracks_alive_until:
            return
        detections = await self.inference.infer(frame)
        tracks = self.tracker.update(detections, mono_ts)
        if tracks:
            self._tracks_alive_until = mono_ts + self.defaults.track_ttl_s
        candidates, signals = self.engine.update(tracks, mono_ts)
        if signals:
            candidates.extend(self.signal_sink.feed_signals(signals))
        for cand in candidates:
            self._start_clip(cand, mono_ts, wall_ts)

    def _start_clip(self, cand: EventCandidate, mono_ts: float, wall_ts: float) -> None:
        task = asyncio.create_task(self._record_and_enqueue(cand, mono_ts, wall_ts))
        self._clip_tasks.add(task)
        task.add_done_callback(self._clip_tasks.discard)

    async def _record_and_enqueue(self, cand: EventCandidate,
                                  mono_ts: float, wall_ts: float) -> None:
        assert self.recorder is not None
        micro = self.queue.depth() >= self.defaults.queue_microclip_depth
        try:
            clip = await self.recorder.record(mono_ts, micro=micro)
        except Exception:
            log.exception("clip recording failed for %s on %s — event dropped "
                          "(no contract path for clipless events in v1)",
                          cand.rule_type.value, cand.camera_id)
            return
        event_id = uuid7()
        payload = EventPayload(
            event_id=event_id,
            camera_id=cand.camera_id,
            zone_id=cand.zone_id,
            object_class=cand.object_class,
            rule_type=cand.rule_type,
            confidence=round(cand.confidence, 4),
            track_id=cand.track_id,
            clip_object_key="__pending__",  # replaced by presign's object_key
            speed_kmh=cand.speed_kmh,
            metadata=cand.metadata,
            occurred_at=_wall_ts_to_rfc3339(wall_ts, mono_ts, cand.ts),
        ).model_dump(mode="json")
        payload.pop("clip_object_key")
        payload.pop("thumb_object_key")
        self.queue.enqueue(event_id, payload, str(clip.clip_path),
                           str(clip.thumb_path) if clip.thumb_path else None)
        if self.queue.depth() >= self.defaults.queue_thumb_drop_depth:
            for p in self.queue.drop_thumbnails():
                Path(p).unlink(missing_ok=True)
        self.health.event_recorded(cand.camera_id)
        log.info("EVENT %s %s cam=%s zone=%s track=%s queued",
                 cand.rule_type.value, cand.object_class.value,
                 cand.camera_id, cand.zone_id, cand.track_id)


def _wall_ts_to_rfc3339(wall_ts: float, mono_ts: float, event_mono_ts: float) -> str:
    from datetime import datetime, timezone
    wall_event = wall_ts - (mono_ts - event_mono_ts)
    return datetime.fromtimestamp(wall_event, tz=timezone.utc).isoformat(
        timespec="milliseconds").replace("+00:00", "Z")


# ---------------------------------------------------------------- supervisor

class Supervisor:
    def __init__(self, settings: EdgeSettings, defaults: PipelineDefaults):
        self.settings = settings
        self.defaults = defaults
        data = settings.data_dir
        data.mkdir(parents=True, exist_ok=True)
        self.queue = EventQueue(data / "event-queue.sqlite3")
        self.spool = SpoolManager(data / "spool", defaults.spool_max_mb * 1024 * 1024)
        self.health = HealthCollector(settings, self.queue.depth, self.queue.oldest_age_s)
        self.secrets = SecretResolver(data)
        self.config_cache = ConfigCache(data)
        self.inference = InferenceService(get_backend(defaults))
        self.publisher = EventPublisher(settings, self.queue, self.spool)
        self.signaling = LiveViewSignaling()
        self.cloudlink = CloudLink(
            settings, self.config_cache,
            on_config=self.apply_config,
            heartbeat_builder=self.health.build_heartbeat,
            on_signaling=self.signaling.handle,
        )
        self.correlator = DriveoffCorrelator([])
        self._pipelines: dict[str, CameraPipeline] = {}
        self._pipeline_tasks: dict[str, asyncio.Task] = {}
        self._config: Optional[DeviceConfig] = None

    # -------------------------------------------------------- signals

    def feed_signals(self, signals: list[Signal]) -> list[EventCandidate]:
        return self.correlator.feed(signals)

    # -------------------------------------------------------- config

    async def apply_config(self, config: DeviceConfig) -> None:
        """Diff-apply: unchanged cameras keep running; ruleset-only changes
        hot-swap in place; rtsp/fps changes restart that camera's pipeline."""
        self._config = config
        self.health.config_etag = config.config_etag
        self.correlator = DriveoffCorrelator.from_config([
            (cam.camera_id, z.zone_id, r)
            for cam in config.cameras for z in cam.zones for r in z.rules
            if r.rule_type == RuleType.driveoff
        ])
        wanted = {cam.camera_id: cam for cam in config.cameras}
        for camera_id in list(self._pipelines):
            if camera_id not in wanted:
                await self._stop_pipeline(camera_id)
        for camera_id, cam in wanted.items():
            existing = self._pipelines.get(camera_id)
            if existing is None:
                self._start_pipeline(cam)
            elif (existing.camera.rtsp_ref != cam.rtsp_ref
                  or existing.camera.detect_fps != cam.detect_fps):
                await self._stop_pipeline(camera_id)
                self._start_pipeline(cam)
            else:
                existing.update_config(cam)

    def _start_pipeline(self, cam: CameraConfig) -> None:
        try:
            rtsp_url = self.secrets.resolve(cam.rtsp_ref)
        except KeyError as e:
            log.error("camera %s not started: %s", cam.camera_id, e)
            return
        decoder = StreamDecoder(cam.camera_id, rtsp_url, cam.detect_fps)
        pipeline = CameraPipeline(cam, self.defaults, self.inference, self.queue,
                                  self.spool, self.health, self, decoder=decoder)
        self._pipelines[cam.camera_id] = pipeline
        self.health.register_camera(pipeline.stream_health)
        self._pipeline_tasks[cam.camera_id] = asyncio.create_task(
            self._run_pipeline(pipeline))

    async def _run_pipeline(self, pipeline: CameraPipeline) -> None:
        """Crash isolation: one camera down != device down (buildspec §9)."""
        from .streams import Backoff
        backoff = Backoff(base=2.0, cap=60.0)
        camera_id = pipeline.camera.camera_id
        while not pipeline._stop.is_set():
            try:
                await pipeline.run()
                return  # clean stop
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("pipeline %s crashed — restarting", camera_id)
                pipeline.stream_health.state = "down"
                await asyncio.sleep(backoff.next_delay())

    async def _stop_pipeline(self, camera_id: str) -> None:
        pipeline = self._pipelines.pop(camera_id, None)
        task = self._pipeline_tasks.pop(camera_id, None)
        if pipeline:
            pipeline.stop()
            self.health.unregister_camera(camera_id)
        if task:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    # -------------------------------------------------------- lifecycle

    async def run(self) -> None:
        await self.inference.start()
        publisher_task = asyncio.create_task(self.publisher.run())
        try:
            await self.cloudlink.run()  # blocks until stop
        finally:
            for camera_id in list(self._pipelines):
                await self._stop_pipeline(camera_id)
            self.publisher.stop()
            await asyncio.gather(publisher_task, return_exceptions=True)
            await self.inference.stop()
            await self.signaling.close()
            self.queue.close()

    def stop(self) -> None:
        self.cloudlink.stop()


# ---------------------------------------------------------------- simulate

async def _synthetic_frames(duration_s: float, fps: int, size: tuple[int, int]
                            ) -> FrameSource:
    """A bright square walking left->right across a dark 'scene' — enough to
    exercise gate, mock detector, tracker, line-cross and zone rules."""
    w, h = size
    n = int(duration_s * fps)
    for i in range(n):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        x = int((i / max(n - 1, 1)) * (w - 60))
        y = h // 2 - 30
        frame[y:y + 60, x:x + 60] = 255
        yield time.monotonic(), time.time(), frame
        await asyncio.sleep(1.0 / fps)


async def run_simulation(config_path: Path, duration_s: float = 12.0) -> int:
    """Full pipeline on synthetic frames + mock detector; events are enqueued
    to a throwaway queue and printed as contract-validated payloads."""
    settings = EdgeSettings(data_dir=Path("./data/simulate"))
    defaults = PipelineDefaults(backend="mock", motion_min_fraction=0.0005,
                                min_object_px=8, clip_pre_s=1.0, clip_post_s=0.5,
                                rule_cooldown_s=5.0)
    config = DeviceConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
    sup = Supervisor(settings, defaults)
    await sup.inference.start()

    async def fake_encode(frames, fps, out_path):
        out_path.write_bytes(b"SIMCLIP" + len(frames).to_bytes(4, "big"))

    async def fake_thumb(frame, out_path):
        out_path.write_bytes(b"SIMTHUMB")

    events_before = sup.queue.depth()
    cam = config.cameras[0]
    fps = 10
    pipeline = CameraPipeline(
        cam, defaults, sup.inference, sup.queue, sup.spool, sup.health, sup,
        frame_source=_synthetic_frames(duration_s, fps, (640, 360)))
    sup.correlator = DriveoffCorrelator.from_config([
        (cam.camera_id, z.zone_id, r) for z in cam.zones for r in z.rules])

    # swap in dependency-free encoders after first frame builds the recorder
    async def patch_encoders():
        while pipeline.recorder is None:
            await asyncio.sleep(0.01)
        pipeline.recorder.encoder = fake_encode
        pipeline.recorder.thumb_encoder = fake_thumb

    await asyncio.gather(pipeline.run(), patch_encoders())
    produced = sup.queue.depth() - events_before
    print(f"\nsimulation done: {produced} event(s) queued")
    for item in sup.queue.next_batch(limit=20):
        payload = json.loads(item["payload_json"])
        print(f"  - {payload['rule_type']}: {payload['object_class']} "
              f"zone={payload.get('zone_id')} at {payload['occurred_at']}")
    sup.queue.close()
    return produced


# ---------------------------------------------------------------- CLI

def cli(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="sentinel-edge")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run", help="production mode: cloud-driven config")
    sim = sub.add_parser("simulate", help="full pipeline on synthetic frames")
    sim.add_argument("--config", type=Path,
                     default=Path("configs/examples/device-config.json"))
    sim.add_argument("--duration", type=float, default=12.0)
    args = parser.parse_args(argv)

    settings = EdgeSettings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.cmd == "simulate":
        asyncio.run(run_simulation(args.config, args.duration))
        return

    if not settings.device_token or not settings.device_id:
        raise SystemExit("SENTINEL_DEVICE_TOKEN / SENTINEL_DEVICE_ID not set — "
                         "run the claim flow first (deploy/provision/claim.py)")
    defaults = load_defaults(settings)
    sup = Supervisor(settings, defaults)
    try:
        asyncio.run(sup.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    cli()
