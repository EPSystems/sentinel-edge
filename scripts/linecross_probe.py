"""Standalone line-cross probe — NO cloud, NO queue, NO clips.

First on-site bring-up tool: point it at a real RTSP camera, draw one line,
and watch crossings print to the console (optionally annotated to an MP4).

It reuses the production components directly — the same StreamDecoder,
MotionGate, detector backend, ByteTracker and RuleEngine the real pipeline
runs — so a crossing you see here is a crossing the product would fire.

Two steps:

  1) grab a still so you can pick line endpoints:
       python scripts/linecross_probe.py grab \
           --rtsp "rtsp://user:pass@10.0.0.5/stream" --out frame.jpg

  2) run detection with your line (normalized 0..1, A->B):
       python scripts/linecross_probe.py run \
           --rtsp "rtsp://user:pass@10.0.0.5/stream" \
           --model models/yolox-s.onnx \
           --line 0.5,0.1,0.5,0.9 \
           --record annotated.mp4

Requirements:
  pip install -e ".[onnx,rtsp]"     # onnxruntime + opencv + PyAV
  a YOLOX ONNX model at --model     # stock COCO export works (see bring-up §2)
  ffmpeg on PATH is only needed if PyAV is absent (FFmpeg decode fallback)
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# import from the installed package (pip install -e .)
from sentinel_edge.config import PipelineDefaults
from sentinel_edge.contracts import CameraConfig, RuleType, Zone, Rule
from sentinel_edge.detect import MotionGate, get_backend
from sentinel_edge.rules import RuleEngine
from sentinel_edge.streams import StreamDecoder
from sentinel_edge.track import ByteTracker


def _parse_line(spec: str) -> list[list[float]]:
    """'x1,y1,x2,y2' (normalized 0..1) -> [[x1,y1],[x2,y2]]."""
    parts = [float(v) for v in spec.split(",")]
    if len(parts) != 4:
        raise SystemExit("--line must be x1,y1,x2,y2 (normalized 0..1)")
    if not all(0.0 <= v <= 1.0 for v in parts):
        raise SystemExit("--line coords must be normalized to 0..1")
    return [[parts[0], parts[1]], [parts[2], parts[3]]]


async def _first_frame(decoder: StreamDecoder) -> np.ndarray:
    async for _mono, _wall, frame in decoder.frames():
        decoder.stop()
        return frame
    raise SystemExit("no frame received — check the RTSP URL / network")


def cmd_grab(args: argparse.Namespace) -> None:
    import cv2
    decoder = StreamDecoder("probe", args.rtsp, args.fps)
    frame = asyncio.run(_first_frame(decoder))
    h, w = frame.shape[:2]
    cv2.imwrite(args.out, frame)
    print(f"saved {args.out}  ({w}x{h})")
    print("open it, read two pixel endpoints (px_x, px_y), then normalize:")
    print(f"   --line  x1/{w},y1/{h},x2/{w},y2/{h}")
    print("example vertical tripwire down the middle:  --line 0.5,0.1,0.5,0.9")


def _build_engine(line_norm: list[list[float]], direction: str,
                  frame_wh: tuple[int, int], defaults: PipelineDefaults) -> RuleEngine:
    zone = Zone(
        zone_id="probe-line",
        name="probe tripwire",
        polygon=line_norm,  # 2 points => directed line A->B
        rules=[Rule(rule_type=RuleType.line_cross, direction=direction,
                    min_confidence=defaults.conf_threshold)],
    )
    camera = CameraConfig(
        camera_id="probe", name="probe", rtsp_ref="rtsp://probe.local/x",
        detect_fps=frame_wh and 8 or 8, lpr_enabled=False, zones=[zone],
    )
    return RuleEngine(camera, defaults, frame_wh)


async def _run(args: argparse.Namespace) -> None:
    import cv2

    line_norm = _parse_line(args.line)
    defaults = PipelineDefaults(
        backend="onnx",
        model_path=args.model,
        conf_threshold=args.conf,
        min_object_px=args.min_object_px,
    )
    detector = get_backend(defaults)
    print(f"loading model {args.model} ...")
    detector.load()
    detector.warmup()
    print("model ready")

    gate = MotionGate(min_fraction=defaults.motion_min_fraction) if not args.no_motion else None
    tracker = ByteTracker(track_ttl_s=defaults.track_ttl_s)
    decoder = StreamDecoder("probe", args.rtsp, args.fps)

    engine: Optional[RuleEngine] = None
    writer = None
    crossings = 0
    tracks_alive_until = float("-inf")
    a_px = b_px = None

    print("running — Ctrl-C to stop\n")
    async for mono, wall, frame in decoder.frames():
        h, w = frame.shape[:2]
        if engine is None:
            engine = _build_engine(line_norm, args.direction, (w, h), defaults)
            a_px = (int(line_norm[0][0] * w), int(line_norm[0][1] * h))
            b_px = (int(line_norm[1][0] * w), int(line_norm[1][1] * h))
            if args.record:
                writer = cv2.VideoWriter(
                    args.record, cv2.VideoWriter_fourcc(*"mp4v"),
                    float(args.fps), (w, h))

        motion = gate.process(frame) if gate is not None else True
        if motion or mono <= tracks_alive_until:
            dets = detector.infer(frame)
            tracks = tracker.update(dets, mono)
            if tracks:
                tracks_alive_until = mono + defaults.track_ttl_s
            candidates, _signals = engine.update(tracks, mono)
            for c in candidates:
                crossings += 1
                print(f"[{time.strftime('%H:%M:%S')}] CROSS #{crossings}  "
                      f"class={c.object_class.value}  track={c.track_id}  "
                      f"dir={c.metadata.get('direction')}  conf={c.confidence:.2f}")
        else:
            tracks = []

        if writer is not None:
            cv2.line(frame, a_px, b_px, (0, 0, 255), 2)
            cv2.putText(frame, f"A", a_px, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f"B", b_px, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            for t in tracks:
                x1, y1, x2, y2 = (int(v) for v in t.bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{t.object_class.value}#{t.track_id}",
                            (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"crossings: {crossings}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            writer.write(frame)

    if writer is not None:
        writer.release()
        print(f"\nannotated video written to {args.record}")


def cmd_run(args: argparse.Namespace) -> None:
    if not Path(args.model).exists():
        raise SystemExit(
            f"model not found: {args.model}\n"
            "  export a YOLOX ONNX model (bring-up §2) and pass it via --model")
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("\nstopped")


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(prog="linecross_probe",
                                description="Cloud-free line-cross probe for on-site bring-up.")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("grab", help="save one frame so you can pick line coords")
    g.add_argument("--rtsp", required=True)
    g.add_argument("--out", default="frame.jpg")
    g.add_argument("--fps", type=int, default=6)
    g.set_defaults(func=cmd_grab)

    r = sub.add_parser("run", help="detect line crossings on the live stream")
    r.add_argument("--rtsp", required=True)
    r.add_argument("--model", default="models/yolox-s.onnx")
    r.add_argument("--line", required=True, help="x1,y1,x2,y2 normalized 0..1 (A->B)")
    r.add_argument("--direction", default="both", choices=["both", "lr", "rl"])
    r.add_argument("--fps", type=int, default=6)
    r.add_argument("--conf", type=float, default=0.4, help="alert confidence floor")
    r.add_argument("--min-object-px", type=int, default=12, dest="min_object_px")
    r.add_argument("--no-motion", action="store_true", help="disable the motion gate")
    r.add_argument("--record", help="write an annotated MP4 to this path")
    r.set_defaults(func=cmd_run)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
