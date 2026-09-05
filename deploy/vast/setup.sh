#!/usr/bin/env bash
# sentinel-edge — one-shot setup for a Vast.ai (or any CUDA) box.
# Installs system deps, a GPU-enabled Python env, the YOLOX-S model, then
# verifies that ONNX Runtime can actually use the GPU. Safe to re-run.
#
# Usage:   bash deploy/vast/setup.sh
set -euo pipefail

# --- locate repo root (this script lives in deploy/vast/) --------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
echo ">> repo root: $REPO_ROOT"

SUDO=""
if [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1; then SUDO="sudo"; fi

# --- 1. system deps ----------------------------------------------------------
echo ">> [1/5] system deps (ffmpeg)"
if ! command -v ffmpeg >/dev/null 2>&1; then
  $SUDO apt-get update -qq
  $SUDO apt-get install -y -qq ffmpeg
else
  echo "   ffmpeg already present: $(ffmpeg -version | head -1)"
fi

# --- 2. python >= 3.11 -------------------------------------------------------
echo ">> [2/5] python check"
if ! python3 - <<'PY'
import sys
raise SystemExit(0 if sys.version_info[:2] >= (3, 11) else 1)
PY
then
  echo "   ERROR: need Python >= 3.11 (have $(python3 --version 2>&1))." >&2
  echo "   Install a newer python3 on this box, then re-run." >&2
  exit 1
fi
echo "   $(python3 --version)"

# --- 3. venv + python deps (GPU build) ---------------------------------------
echo ">> [3/5] venv + pip install (onnxruntime-gpu, NOT the cpu onnx extra)"
if [ ! -d .venv ]; then python3 -m venv .venv; fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip -q
# base + PyAV (rtsp) + psutil (metrics); deliberately skip the [onnx] extra,
# which pins CPU onnxruntime and conflicts with onnxruntime-gpu.
pip install -q -e ".[dev,rtsp,metrics]"
pip install -q onnxruntime-gpu opencv-python-headless

# --- 4. model ----------------------------------------------------------------
echo ">> [4/5] fetch YOLOX-S ONNX model"
if [ -f models/yolox-s.onnx ]; then
  echo "   models/yolox-s.onnx already present ($(du -h models/yolox-s.onnx | cut -f1))"
else
  python scripts/fetch_model.py --verify
fi

# --- 5. GPU verification -----------------------------------------------------
echo ">> [5/5] GPU verification"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
else
  echo "   WARN: nvidia-smi not found — is this a GPU instance?"
fi
# Real check: build a session on the actual model and report the ACTIVE provider
# (get_available_providers() can list CUDA even when cuDNN fails to load).
python - <<'PY'
import onnxruntime as ort, pathlib, sys
print("   compiled providers:", ort.get_available_providers())
m = pathlib.Path("models/yolox-s.onnx")
if not m.exists():
    print("   (model missing — skipping session check)"); sys.exit(0)
try:
    sess = ort.InferenceSession(
        str(m), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    active = sess.get_providers()
    print("   active providers:", active)
    if "CUDAExecutionProvider" in active:
        print("   OK: ONNX Runtime is using the GPU.")
    else:
        print("   WARN: running on CPU only. Usually missing cuDNN — try:")
        print("         pip install nvidia-cudnn-cu12")
        print("         (or match onnxruntime-gpu to this box's CUDA from nvidia-smi)")
except Exception as e:  # noqa: BLE001
    print(f"   WARN: could not create a CUDA session ({e}); pipeline will fall back to CPU.")
PY

echo ""
echo ">> done. Next:"
echo "     source .venv/bin/activate"
echo "     bash deploy/vast/run-bench.sh [path/to/walking-person.mp4]"
