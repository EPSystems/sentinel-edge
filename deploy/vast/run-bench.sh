#!/usr/bin/env bash
# sentinel-edge — GPU bench runner.
# Serves a local synthetic RTSP "camera" (MediaMTX + a looped clip), points a
# config at it, and runs `sentinel-edge local` so YOLOX-S runs on the GPU.
# Ctrl+C stops the pipeline and tears down the camera + server.
#
# Usage:   bash deploy/vast/run-bench.sh [path/to/walking-person.mp4]
#          DETECT_FPS=25 bash deploy/vast/run-bench.sh clip.mp4
#
# With no clip, a synthetic test pattern is generated so the pipeline + GPU
# still run — but note YOLOX won't find a *person* in a test pattern, so you
# need a real clip of someone walking to actually see `line_cross` fire.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

RTSP_URL="rtsp://127.0.0.1:8554/cam"
DETECT_FPS="${DETECT_FPS:-20}"
MTX_VERSION="${MTX_VERSION:-latest}"
MTX_ARCH="${MTX_ARCH:-linux_amd64}"
LOG_DIR="$SCRIPT_DIR/.bench-logs"
mkdir -p "$LOG_DIR"
SAMPLE="${1:-}"

# --- venv --------------------------------------------------------------------
if [ ! -d .venv ]; then
  echo "ERROR: no .venv — run 'bash deploy/vast/setup.sh' first." >&2; exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# --- sample clip -------------------------------------------------------------
if [ -z "$SAMPLE" ]; then
  SAMPLE="$LOG_DIR/testclip.mp4"
  if [ ! -f "$SAMPLE" ]; then
    echo ">> no clip given — generating a synthetic test pattern (no person in it)"
    ffmpeg -y -loglevel error -f lavfi -i testsrc=size=1280x720:rate=15:duration=20 \
      -pix_fmt yuv420p -c:v libx264 -preset ultrafast "$SAMPLE"
  fi
  echo "   NOTE: a test pattern exercises the GPU but has no person to detect."
  echo "         Pass a real walking-person clip to see line_cross fire."
elif [ ! -f "$SAMPLE" ]; then
  echo "ERROR: clip not found: $SAMPLE" >&2; exit 1
fi
echo ">> camera source clip: $SAMPLE"

# --- MediaMTX (only start our own if 8554 is free) ---------------------------
STARTED_MTX=0
MTX_PID=""
FFMPEG_PID=""
port_open() { python - "$1" <<'PY'
import socket, sys
s = socket.socket(); s.settimeout(0.5)
sys.exit(0 if s.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
PY
}

cleanup() {
  echo ""; echo ">> tearing down camera + server"
  [ -n "$FFMPEG_PID" ] && kill "$FFMPEG_PID" 2>/dev/null || true
  if [ "$STARTED_MTX" = "1" ] && [ -n "$MTX_PID" ]; then
    kill "$MTX_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if port_open 8554; then
  echo ">> something is already serving RTSP on :8554 — reusing it"
else
  MTX_BIN="$SCRIPT_DIR/mediamtx"
  if [ ! -x "$MTX_BIN" ]; then
    echo ">> downloading MediaMTX ($MTX_VERSION, $MTX_ARCH)"
    url="https://github.com/bluenviron/mediamtx/releases/${MTX_VERSION}/download/mediamtx_${MTX_VERSION}_${MTX_ARCH}.tar.gz"
    if [ "$MTX_VERSION" = "latest" ]; then
      url="https://github.com/bluenviron/mediamtx/releases/latest/download/mediamtx_${MTX_ARCH}.tar.gz"
    fi
    curl -fsSL "$url" -o "$LOG_DIR/mediamtx.tgz"
    tar xzf "$LOG_DIR/mediamtx.tgz" -C "$SCRIPT_DIR" mediamtx
  fi
  echo ">> starting MediaMTX (log: $LOG_DIR/mediamtx.log)"
  ( cd "$SCRIPT_DIR" && ./mediamtx ) >"$LOG_DIR/mediamtx.log" 2>&1 &
  MTX_PID=$!
  STARTED_MTX=1
  for _ in $(seq 1 20); do if port_open 8554; then break; fi; sleep 0.5; done
  if ! port_open 8554; then
    echo "ERROR: MediaMTX did not come up on :8554 — see $LOG_DIR/mediamtx.log" >&2
    exit 1
  fi
fi

# --- push the looped clip in as the "camera" ---------------------------------
echo ">> pushing looped clip -> $RTSP_URL (log: $LOG_DIR/ffmpeg.log)"
ffmpeg -hide_banner -loglevel warning -re -stream_loop -1 -i "$SAMPLE" -an \
  -c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p \
  -f rtsp -rtsp_transport tcp "$RTSP_URL" >"$LOG_DIR/ffmpeg.log" 2>&1 &
FFMPEG_PID=$!
sleep 2  # let the first frames land

# --- config pointed at the local stream --------------------------------------
if [ ! -f my-site.json ]; then
  echo ">> creating my-site.json from the example, aimed at $RTSP_URL"
  python - "$RTSP_URL" "$DETECT_FPS" <<'PY'
import json, sys, pathlib
url, fps = sys.argv[1], int(sys.argv[2])
cfg = json.loads(pathlib.Path("configs/examples/device-config.json").read_text())
cam = cfg["cameras"][0]
cam["rtsp_ref"] = url
cam["detect_fps"] = fps
pathlib.Path("my-site.json").write_text(json.dumps(cfg, indent=2))
print(f"   wrote my-site.json (detect_fps={fps})")
PY
else
  echo ">> using existing my-site.json (delete it to regenerate from the example)"
fi

# --- run the pipeline (foreground; Ctrl+C exits and triggers cleanup) --------
echo ">> starting sentinel-edge  |  preview: http://127.0.0.1:8090/  (SSH -L 8090 to view)"
echo "   watch for: EVENT line_cross person ... queued   |   clips in data/spool/"
echo ""
python -m sentinel_edge local --config my-site.json --preview
