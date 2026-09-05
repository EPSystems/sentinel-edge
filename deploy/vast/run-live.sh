#!/usr/bin/env bash
# sentinel-edge — LIVE on-site camera runner (NVIDIA edge box).
# Runs the real pipeline against a real RTSP camera on the LOCAL network.
# This must run on a box that can reach the camera's IP (i.e. on-site, NOT a
# datacenter/Vast box). Credentials live in data/secrets.json (gitignored),
# never in the config that the zone editor / cloud round-trip.
#
# Prereqs (once):  bash deploy/vast/setup.sh          # venv + onnxruntime-gpu + model
# Usage:           bash deploy/vast/run-live.sh [rtsp_ref]        # default ref: cam-front
#                  DETECT_FPS=15 bash deploy/vast/run-live.sh cam-front
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

REF="${1:-cam-front}"
DETECT_FPS="${DETECT_FPS:-15}"
CONFIG="my-site.json"
SECRETS="data/secrets.json"

# --- venv --------------------------------------------------------------------
if [ ! -d .venv ]; then
  echo "ERROR: no .venv — run 'bash deploy/vast/setup.sh' first." >&2; exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# --- credentials (kept OUT of the config; data/ is gitignored) ---------------
if [ ! -f "$SECRETS" ] || ! grep -q "\"$REF\"" "$SECRETS" 2>/dev/null; then
  cat >&2 <<EOF
ERROR: no secret for rtsp_ref '$REF' in $SECRETS

Create it (never commit it) with your camera's real URL:

  mkdir -p data
  cat > $SECRETS <<'JSON'
  { "$REF": "rtsp://USER:PASS@192.168.1.64:554/Streaming/Channels/101" }
  JSON

  Hikvision: .../Streaming/Channels/101    Dahua: .../cam/realmonitor?channel=1&subtype=0
  Tip: try the substream (102 / subtype=1) first — lighter load, easier first light.
EOF
  exit 1
fi

# --- config pointed at the OPAQUE ref (no creds in the file) ------------------
if [ ! -f "$CONFIG" ]; then
  echo ">> creating $CONFIG (rtsp_ref='$REF') — its zones are placeholders, REDRAW them for this scene."
  python - "$REF" "$DETECT_FPS" <<'PY'
import json, sys, pathlib
ref, fps = sys.argv[1], int(sys.argv[2])
cfg = json.loads(pathlib.Path("configs/examples/device-config.json").read_text())
cam = cfg["cameras"][0]
cam["rtsp_ref"] = ref          # opaque name, resolved from data/secrets.json at runtime
cam["detect_fps"] = fps
pathlib.Path("my-site.json").write_text(json.dumps(cfg, indent=2))
print(f"   wrote my-site.json (rtsp_ref={ref!r}, detect_fps={fps})")
PY
else
  echo ">> using existing $CONFIG (delete it to regenerate from the example)"
fi

# --- preflight: actually open the stream, using the pipeline's own decoder ----
# Validates network reachability + credentials + RTSP dialect + decode in one
# shot. Credentials stay in memory; the URL is redacted in all output.
echo ">> preflight: opening camera '$REF' (this can take a few seconds)…"
python - "$REF" <<'PY'
import asyncio, sys
from sentinel_edge.config import SecretResolver, redact_rtsp
from sentinel_edge.local.zones_tool import grab_snapshot
ref = sys.argv[1]
url = SecretResolver("data").resolve(ref)   # raises if the ref is missing
jpeg = asyncio.run(grab_snapshot(url))       # raises SystemExit (redacted) on timeout
print(f"   camera OK: {redact_rtsp(url)}  (first frame decoded, {len(jpeg)} bytes)")
PY

echo ""
echo ">> if you have not drawn lines/zones for THIS camera yet, stop and run:"
echo "     python -m sentinel_edge zones --config $CONFIG        # http://127.0.0.1:8091/"
echo "   (line anchor: feet by default; pick 'head' for high-mount / heads-only views)"
echo ""
echo ">> starting sentinel-edge  |  preview: http://127.0.0.1:8090/  (SSH -L 8090 to view)"
echo "   watch for: EVENT line_cross person ... queued   |   clips -> data/spool/   |   Ctrl+C to stop"
echo ""
python -m sentinel_edge local --config "$CONFIG" --preview
