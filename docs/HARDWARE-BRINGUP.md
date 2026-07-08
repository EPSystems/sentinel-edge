# sentinel-edge — hardware bring-up (Pavel's lane)

The software pipeline is built and verified: 96/96 tests, full loop proven in
`simulate` mode, wire format cross-validated against the cloud contract's own
examples. **Everything left requires physical hardware, real cameras, or
model artifacts** — your lane per buildspec `02-sentinel-edge-buildspec.md`.

Work top to bottom; each item lists where it plugs into the code and the
acceptance bar. Nothing here requires changing pipeline logic — if you find
you *do* need to, flag it first (the logic is under test).

---

## 0. Read first (30 min)

- `README.md` — build/run/test, config layering, zone semantics.
- Run it yourself: `pip install -e ".[dev]" && pytest && python -m sentinel_edge simulate`.
- Your prototype lives in `legacy/prototype_line_counter.py`; its detector
  (YOLOX) and tracker choices carried over — the counting logic was replaced
  (segment-intersection, bottom-centre anchor; see README §7 for why).

## 1. Bench hardware purchase — **blocks everything below**

Buy 1× each (buildspec §7.2, budget ~€1,100–1,300):
- Pi 5 (8 GB) + Hailo-8L HAT
- mini-PC + Hailo-8 M.2
- Jetson Orin Nano (8 GB)

## 2. Model export pipeline (buildspec §5.2)

Source model: **YOLOX (Apache-2.0)** — licence already aligned; keep the
Ultralytics Enterprise quote on file as fallback only.

1. Pick the variant (start: YOLOX-S @ 640) and pin it as
   `models/yolox-<variant>-vX.Y/`; class map for the custom 5-class export is
   already wired in `configs/profiles/accel-hailo8.yaml` / `accel-jetson-orin.yaml`
   (`person, car, truck, motorcycle, animal`). Until you train/re-head, the
   stock COCO export works with `accel-onnx-cpu.yaml`'s COCO map.
2. Export ONNX (fixed 640×640) → verify on a dev box:
   `pip install -e ".[onnx]"`, drop the file at `models/yolox-s.onnx`,
   set `SENTINEL_ACCEL_PROFILE=accel-onnx-cpu`, feed a test RTSP stream.
   Decode/NMS is already implemented (`detect/backend_onnx.py`) — if your
   export uses `decode_in_inference=True`, tell me and I'll add the flag.
3. **Hailo:** ONNX → HEF via the Dataflow Compiler, INT8 with a
   representative BG calibration set (forecourt + warehouse, day/night).
   Compile the NMS layer in — `detect/backend_hailo.py` expects
   NMS-on-chip output (per-class `[ymin, xmin, ymax, xmax, score]`,
   normalized). If your HEF emits raw head output instead, say so — the
   ONNX decode path can be reused.
4. **Jetson:** build the `.engine` **on the target Jetson** with `trtexec`
   (FP16 first). `detect/backend_tensorrt.py` reuses the ONNX pre/post.
5. Record mAP before/after quantisation on a held-out BG set; publish the
   artifact with semver + checksum to `models/`.

**Accept:** the same pipeline runs on Hailo, Jetson and ONNX by changing only
`SENTINEL_ACCEL_PROFILE`; post-INT8 mAP loss within the §7 threshold.

## 3. Backend validation on device

`detect/backend_hailo.py` and `backend_tensorrt.py` are written against the
documented runtime APIs but have **never touched real silicon**. Expect small
API-version fixes (HailoRT bindings especially). Validate:

- `Detector.load()` + `warmup()` succeed on the device image.
- Boxes are correctly scaled to source resolution (feed a person at a known
  position, check the logged bbox).
- `last_latency_ms` is sane — it feeds the §7 benchmark.

## 4. Real-camera RTSP validation (`streams/`)

Against at least 2 real camera brands (Hikvision/Dahua cover most BG sites):

- PyAV path (`pip install -e ".[rtsp]"`) and FFmpeg fallback both connect.
- **Cable-pull test:** yank the camera cable 60 s — pipeline must resume
  ≤ 35 s after link restore, no process restart (buildspec §4.1 DoD).
- 4 simultaneous 1080p streams on the Hailo profile for 24 h, zero unhandled
  exceptions.
- `grep -ri "hunter2\|rtsp://.*:.*@" logs/` → credentials never in logs
  (redaction is implemented + tested; verify on real URLs).

## 5. FPS benchmark (buildspec §7.2) — gates published camera counts

Replay BG footage as N synthetic RTSP feeds (go2rtc/MediaMTX looping files);
ramp N per device until a bar breaks:

| Metric | Pass bar |
| --- | --- |
| Detect FPS on active frames | ≥ 6 |
| Frame → metadata-POST-ready latency | ≤ 2 s |
| Idle-scene inference suppression | ≥ 80 % (motion gate exposes `suppression_rate`) |
| CPU/accelerator headroom at rated count | ≥ 15 % |
| 24 h soak | zero crashes, flat memory |

**Output:** the signed capacity table that replaces the "to be confirmed"
numbers in buildspec §6.1 — quoting depends on it.

## 6. Provisioning image (deploy/provision/)

`build-image.sh` documents the steps; automate per arch (pi-gen / JetPack /
FAI): base OS + Docker + compose stack + accelerator runtime + the
`first-boot.sh` claim service + Tailscale (outbound-only) + read-only root
where practical. **Accept:** flash → claim → green in dashboard in ≤ 15 min
with no inbound firewall change (buildspec §6.3 DoD).

## 7. LPR on real plates (`lpr/`)

BG normalisation + watchlist are implemented and tested on synthetic strings.
Needs you: camera angle/zoom guidance, a labelled test set of real BG plates,
accuracy measurement, and the per-track (not per-frame) trigger wired into
the pipeline once a plate-detector model is chosen (`fast-plate-ocr` via
`[lpr]` extra is pre-wired in `lpr/ocr.py`).

## 8. Speed calibration procedure (§8)

`rules/speed.py` accepts `px_per_m` or a 3×3 homography from
`cameras.calibration`. Needs you: the on-site calibration runbook (mark ≥4
known ground points at onboarding) and a known-speed drive-by test —
estimates within ±20 % across three runs.

## 9. 72 h resilience soak (§9) — final gate before pilot

Unattended, with injected faults: camera drop, 30-min WAN outage during
active events, power-cycle mid-event. **Accept:** zero lost event metadata
(the SQLite journal guarantees this in tests — prove it on-device), accurate
health state throughout, autonomous recovery.

---

## Coordination points with Emil (cloud side)

- WS hub `config.ack`/`config.reject` envelope shape (README §8.1).
- Presign/ingest are live on the cloud — end-to-end Flow A test needs his
  credentials wired (`sentinel-cloud/docs/CREDENTIALS-TODO.md`).
- Report actual queue caps you hit in §5 so backpressure thresholds
  (`queue_thumb_drop_depth` / `queue_microclip_depth`) get sized to a real
  busy-night outage.
