# First physical camera — line-cross bench test (no cloud needed)

Goal: one real camera, one line drawn on its view, a person walks across it,
the pipeline fires `line_cross` and records a clip. Everything runs on a
normal Windows/Linux PC on the camera's LAN — no cloud credentials, no
accelerator hardware.

This whole flow was proven end-to-end against a synthetic RTSP camera
(MediaMTX + looped H.264 footage of a walking person): PyAV decode → YOLOX-S
ONNX → ByteTrack → `line_cross` → clip in the spool. The only untested part
is your specific camera's RTSP dialect — which is exactly what this test is
for (HARDWARE-BRINGUP.md §4).

## 1. One-time setup (~10 min + model download)

```sh
cd sentinel-edge
python -m venv .venv
.venv\Scripts\activate                   # Linux/mac: source .venv/bin/activate
pip install -e ".[dev,onnx,rtsp]"        # onnxruntime + opencv + PyAV
python scripts/fetch_model.py --verify   # 36 MB YOLOX-S ONNX -> models/yolox-s.onnx
```

FFmpeg must be on PATH (clip encoding): `winget install Gyan.FFmpeg`
(Windows) or `apt install ffmpeg`. Re-open the terminal after installing.

## 2. Point a config at the camera

Copy the template and put your camera's RTSP URL in it:

```sh
copy configs\examples\device-config.json my-site.json
```

Edit `my-site.json`:
- `cameras[0].rtsp_ref` → the real URL, e.g.
  `rtsp://user:pass@192.168.1.64:554/Streaming/Channels/101` (Hikvision) or
  `rtsp://user:pass@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0` (Dahua).
  A raw `rtsp://` URL in the config is a dev convenience; for anything shared,
  keep the URL out of the config: set `rtsp_ref` to an opaque name and map it
  in `data/secrets.json`: `{"cam-front": "rtsp://user:pass@..."}`.
- `cameras[0].detect_fps` → 5 is plenty for walking people on CPU.
- Delete the sample zones — you'll draw real ones next.

## 3. Draw the line on the live view

```sh
python -m sentinel_edge zones --config my-site.json
# -> zone editor: http://127.0.0.1:8091/
```

The browser shows a snapshot from the camera. Click **+ New line**, click
point A, then point B — the crossing direction ("lr"/"rl") is relative to
the A→B arrow, `both` fires either way. Adjust the rule (classes,
min confidence, direction) in the side panel and **Save to config**. The
file is validated through the same contract models the cloud uses, so
whatever saves here is exactly what production Flow B will push later.

Placement tips: the anchor is the person's **feet** (bottom-centre of the
box), so put the line on the ground where people step over it, not across
torso height. Keep it away from the frame edge — detection needs to see the
person on both sides of the line for at least a frame each.

## 4. Run it and walk the line

```sh
python -m sentinel_edge local --config my-site.json --preview
# preview:      http://127.0.0.1:8090/   (boxes, line, event flashes)
# expected log: EVENT line_cross person cam=... zone=... queued
```

Walk across the line. You should see: your box tracked in the preview, a red
EVENT flash, an `EVENT line_cross person ... queued` log line, and a ~10 s
`.mp4` + `.jpg` appearing in `data/spool/`. Events accumulate in
`data/event-queue.sqlite3` (local mode never uploads; `run` mode's publisher
drains this same queue to the cloud).

Ctrl+C prints a summary of everything queued.

## 5. What to check / tune

| Symptom | Fix |
| --- | --- |
| no frames / timeout | verify the URL in VLC first; try the camera's substream (`subtype=1` / channel `102`) |
| person detected but no event | line at torso height (anchor is feet — move it to the ground), or person never fully crosses |
| detections flicker below 0.7 | lower the rule's `min_confidence` to 0.5 — the walk test tells you the real number for your lighting |
| duplicate events for one crossing | shouldn't happen (30 s per-track cooldown); if it does, capture the log — that's a bug |
| CPU pegged, laggy preview | lower `detect_fps` to 3–4; CPU ONNX does ~250 ms/frame — the accelerators (Pavel's lane) are what make 6+ FPS real |
| clip missing, event dropped | ffmpeg not on PATH in this terminal |

## 6. What this proves / what it doesn't

Proves: RTSP ingest against your camera brand, real-model detection, tracking,
line-cross geometry, clip recording, durable queueing — i.e. buildspec Flow A
up to the publish step, on real hardware.

Doesn't prove: cloud publish (needs the claim flow + cloud creds —
`sentinel-cloud/docs/CREDENTIALS-TODO.md`), accelerator FPS (HARDWARE-BRINGUP
§5), 24 h stability (§4). Next step after this passes: the cable-pull test in
HARDWARE-BRINGUP §4 with the same setup.
