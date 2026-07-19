# sentinel-edge

On-premises AI video-analytics service (E&P Systems). One Python process per
device: RTSP in → motion gate → YOLOX → ByteTrack → rules → 10 s clip →
publish to the cloud. **Raw video never leaves the site** — only event
metadata, a clip and a thumbnail go out, over outbound-only HTTPS/WSS.

Implements `business-plan/03-software/02-sentinel-edge-buildspec.md` against
the shared `/v1` contract in `sentinel-cloud/packages/contracts`.

---

## 1. Repository layout

```
src/sentinel_edge/
├── contracts/   Pydantic mirrors of the shared JSON Schemas (wire types only)
├── config.py    env settings, profile layering, config cache, rtsp_ref secrets
├── streams/     RTSP ingest (PyAV or FFmpeg), backoff reconnect, watchdog
├── detect/      motion gate + Detector ABC: onnx | hailo | tensorrt | mock
├── track/       ByteTrack-style two-stage IoU tracker behind our own types
├── rules/       geometry, rule engine (hot-reload), speed (±20%), drive-off
├── lpr/         BG plate normalisation + OCR ([lpr] extra) + watchlist
├── record/      ring buffer, FFmpeg clip/thumb encode, size-capped spool
├── publish/     durable SQLite queue + presign→PUT→POST publisher
├── cloudlink/   outbound WSS: config down, heartbeat up, live-view signaling
├── health/      heartbeat builder (drives the 90 s device-red SLO)
├── local/       bench tools: `local` mode (real cameras, no cloud) + `zones` editor
└── main.py      supervisor + per-camera pipelines + CLI
configs/profiles/  accelerator + vertical presets (see §4)
configs/examples/  sample tenant DeviceConfig for simulate
deploy/            Dockerfile, compose (+go2rtc), provisioning + claim flow
scripts/           fetch_model.py — download + verify the stock YOLOX-S ONNX
tests/             104 tests — no camera, GPU, ffmpeg or cloud needed
docs/              HARDWARE-BRINGUP.md (Pavel's device lane),
                   FIRST-CAMERA.md (first physical line-cross test)
legacy/            Pavel's original prototype (superseded)
```

## 2. Requirements

| What | Version | Needed for |
| --- | --- | --- |
| Python | **3.11+** (3.13 works) | everything |
| FFmpeg on PATH | any recent | clip encoding + the FFmpeg RTSP fallback (`run` mode only) |
| Docker + compose | any recent | device deployment (§6) |

Optional extras (`pip install "sentinel-edge[<extra>]"`):

| Extra | Pulls in | When |
| --- | --- | --- |
| `dev` | pytest, pytest-asyncio | development + CI |
| `onnx` | onnxruntime, opencv-headless | ONNX bench/dev backend (**not sold**) |
| `rtsp` | PyAV | preferred RTSP decode path |
| `lpr` | fast-plate-ocr | plate reading on LPR-flagged cameras |
| `metrics` | psutil | cpu/mem fields in the heartbeat |

Hailo (HailoRT) and Jetson (TensorRT/PyCUDA) runtimes are **never installed
via pip** — they ship with the provisioned device image (§6).

## 3. Build, test, run — developer machine

```sh
cd sentinel-edge

# 1. create a venv and install editable with dev tools
python -m venv .venv
.venv\Scripts\activate            # Windows        (Linux/mac: source .venv/bin/activate)
pip install -e ".[dev]"

# 2. run the test suite (~40 s, fully offline)
pytest                            # expect: 104 passed
pytest tests/test_rules_engine.py -k line_cross -v     # single area
pytest -q tests/test_simulation.py                     # the end-to-end test

# 3. run the full pipeline with zero hardware
python -m sentinel_edge simulate
#   synthetic frames + mock detector + fake encoders; a simulated person
#   crosses the sample config's zone and tripwire. Expected output:
#   EVENT zone_enter person ... queued
#   EVENT line_cross person ... queued
#   simulation done: 2 event(s) queued
python -m sentinel_edge simulate --config configs/examples/device-config.json --duration 12

# 4. run against a REAL camera with no cloud (see docs/FIRST-CAMERA.md)
pip install -e ".[dev,onnx,rtsp]"
python scripts/fetch_model.py --verify          # models/yolox-s.onnx (36 MB)
python -m sentinel_edge zones --config my-site.json   # draw lines/zones in the
#   browser on a live snapshot (http://127.0.0.1:8091) and save to the config
python -m sentinel_edge local --config my-site.json --preview
#   real RTSP + ONNX detection; events queue locally, clips land in data/spool/;
#   annotated live view at http://127.0.0.1:8090
```

What the tests cover: geometry + line-cross edge cases (slow/fast/on-line/
overshoot), rule engine (enter/dwell/speed/cooldowns/hot-swap), drive-off
correlation, tracker identity/TTL/occlusion, motion-gate suppression, the
SQLite queue state machine, the publisher against a mock cloud (happy path,
409/422/5xx/offline), cloudlink envelope codec + config push/reject + ETag
sync + heartbeat fallback, config layering/cache/secrets/redaction, BG plate
normalisation, clip recorder pre/post-roll, and one end-to-end simulation.
`tests/test_contracts.py` additionally validates our models against the
**cloud contract's own example payloads** (skipped automatically if
`sentinel-cloud/` is not on disk next to this repo).

Build a wheel (CI/artifact): `pip install build && python -m build`.

## 4. Configuration — lowest to highest precedence

1. **Built-in defaults** — `config.PipelineDefaults`, every field documented.
2. **Accelerator profile** — `configs/profiles/accel-*.yaml`: backend, model
   path/size, class map. Chosen by `SENTINEL_ACCEL_PROFILE`.
3. **Vertical profile** — `configs/profiles/vertical-*.yaml`: petrol vs
   warehouse tuning (cooldowns, dwell, gate sensitivity). Chosen by
   `SENTINEL_VERTICAL_PROFILE`.
4. **Tenant config from the cloud** — the contract `DeviceConfig` pushed over
   the WS (or pulled with ETag): cameras, zones, rules, and per-rule
   overrides (`min_confidence`, `classes`, `dwell_s`, `direction`,
   `cooldown_s`, `speed_limit_kmh`, `exit_zone_ref`, `grace_s`, …).
   **Hot-reloads in ≤ one frame, no restart**, and is cached to disk so a
   reboot during a cloud outage runs the last good ruleset.

Per-device environment (set at claim time, `SENTINEL_` prefix, `.env` supported):

| Variable | Meaning | Default |
| --- | --- | --- |
| `SENTINEL_CLOUD_BASE_URL` | cloud origin; `/v1` appended automatically | `https://cloud.example.com` |
| `SENTINEL_DEVICE_ID` | this device's uuid (claim-issued) | — (required for `run`) |
| `SENTINEL_DEVICE_TOKEN` | per-device bearer token (claim-issued) | — (required for `run`) |
| `SENTINEL_DATA_DIR` | queue, spool, config cache, secrets | `./data` |
| `SENTINEL_PROFILES_DIR` | profile YAML directory | `./configs/profiles` |
| `SENTINEL_ACCEL_PROFILE` | accelerator profile name | `accel-onnx-cpu` |
| `SENTINEL_VERTICAL_PROFILE` | vertical profile name | `vertical-petrol` |
| `SENTINEL_MODEL_VERSION` | reported in heartbeat (OTA drift view) | `yolox-s-v0.0-dev` |
| `SENTINEL_HEARTBEAT_INTERVAL_S` | heartbeat cadence | `30` |
| `SENTINEL_LOG_LEVEL` | logging level | `INFO` |

**RTSP credentials** never travel in the tenant config — it carries an opaque
`rtsp_ref`. The mapping ref → real URL lives in `<data_dir>/secrets.json`
(`{"secret://cam/xyz": "rtsp://user:pass@10.0.0.5/stream"}`), written at
provision time, decrypted only in memory, redacted in every log line.

**Zone semantics:** a polygon with ≥3 points is an area; **exactly 2 points
is a directed line A→B** (used by `line_cross` and as the drive-off exit).
`direction: "lr" | "rl" | "both"` — sign convention documented in
`rules/geometry.py`; the dashboard renders the A→B arrow.

## 5. Running in production

```sh
sentinel-edge run          # equivalently: python -m sentinel_edge run
```

Refuses to start without `SENTINEL_DEVICE_ID`/`SENTINEL_DEVICE_TOKEN`. Boot
order: cached config applied first → cloudlink connects (outbound WSS) →
config re-synced with ETag → one pipeline task per camera → heartbeat every
30 s (HTTP fallback when the WS is down). A crashed camera pipeline restarts
with backoff and never takes the device down.

## 6. Device deployment (the real thing)

```sh
# on the device (or bake into the golden image — deploy/provision/build-image.sh)
cd /opt/sentinel-edge

# 1. operator claims the device in the dashboard -> gets the one-time token
python deploy/provision/claim.py --device-id <uuid> --cloud https://<cloud-host>
# 2. start the stack (sentinel-edge + go2rtc sidecar)
docker compose -f deploy/docker-compose.yml up -d
# 3. the device turns green in the dashboard within 90 s
```

Docker image: `docker build -f deploy/Dockerfile .` (multi-stage, ffmpeg
included; `--build-arg BASE=` for arch-specific bases). The compose file
publishes **no ports** — the device is outbound-only; go2rtc is reachable
only on the compose network. Accelerator passthrough (e.g. `/dev/hailo0`) is
commented in `deploy/docker-compose.yml` per profile.

## 7. Design decisions worth knowing

- **Line-cross is segment–segment intersection**, not a side-sign flip: no
  minimum/maximum speed window, no miss when a point lands exactly on the
  line, no false fire on the line's infinite extension (the bug class that
  broke the prototype).
- **Anchor is bottom-centre by default** (feet/wheels) — a person's box
  *centre* is at hip height and may never touch a ground line.
- **Detection keeps running while tracks are alive even without motion** — a
  motionless loiterer must keep accruing dwell time; idle-scene suppression
  (≥80%) is unaffected because idle scenes have no tracks.
- **Every event is journaled to SQLite before any network I/O**; the
  publisher drains oldest-first with presign-at-upload-time, idempotent on
  the UUIDv7 `event_id`. A 30-min WAN outage loses zero metadata.
- **Backpressure:** queue depth > threshold drops queued thumbnails first,
  then new events record single-frame *micro-clips* instead of 10 s clips.
- **Cooldown (default 30 s per track×rule) survives config hot-swaps** — a
  zone edit can't unleash an alert storm (WhatsApp is the dominant variable
  cost).
- Detector backends are lazy-imported; the package imports and tests cleanly
  on any machine. ONNX is bench/dev only — **not sold**. YOLOX (Apache-2.0)
  keeps the stack permissive (buildspec §5.1).

## 8. Cloud coordination points (v1-additive, tracked)

1. **config.ack / config.reject** — sent as `command` envelopes with
   `payload.command`; the cloud WS hub (an envelope-validating stub today)
   should adopt or amend this shape.
2. **True metadata-only degrade** needs the cloud to accept an event without
   a HEAD-checked clip; until then the edge ships micro-clips.
3. **Trickle ICE for live view** is dropped (non-trickle SDP only) until the
   cloud TURN work lands (Phase 4).

## 9. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `run` exits: token/id not set | run the claim flow (§6) or export `SENTINEL_DEVICE_*` |
| `DetectorError: onnxruntime not installed` | `pip install "sentinel-edge[onnx]"` and put the model at the profile's `model_path` |
| `HailoRT not present` / `TensorRT not present` | expected off-device; these backends only run on the provisioned image |
| `no secret material for rtsp_ref ...` | camera missing from `<data_dir>/secrets.json` |
| clips fail to encode | ffmpeg not on PATH inside the container/host |
| events queue but never POST | check `SENTINEL_CLOUD_BASE_URL`, token validity (401 backs off), and `/readyz` on the cloud (R2 creds) |
| profile YAML typo | fails loudly at boot with the offending key — fix the YAML |

## 10. What still needs real hardware

The Hailo/TensorRT backends, model export (§5.2), the FPS benchmark (§7),
the 72 h resilience soak, and real-camera RTSP behaviour. The full task list
with acceptance criteria lives in **`docs/HARDWARE-BRINGUP.md`** (Pavel's lane).
