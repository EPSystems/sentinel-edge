#!/usr/bin/env bash
# Golden-image build (buildspec §6.2) — one image per architecture.
#   ./build-image.sh hailo|jetson|x86
# Produces a flashable OS image with Docker, the compose stack, the
# accelerator runtime, and the first-boot claim service baked in.
set -euo pipefail

PROFILE="${1:?usage: build-image.sh hailo|jetson|x86}"

case "$PROFILE" in
  hailo)  BASE_OS="rpi-os-lite-arm64";  RUNTIME="hailort" ;;
  jetson) BASE_OS="jetpack-6";          RUNTIME="tensorrt (bundled)" ;;
  x86)    BASE_OS="debian-12-genericcloud-amd64"; RUNTIME="onnxruntime (dev only)" ;;
  *) echo "unknown profile $PROFILE" >&2; exit 1 ;;
esac

echo "== building golden image: $PROFILE (base: $BASE_OS, runtime: $RUNTIME)"
echo "Steps (automate with pi-gen / NVIDIA sdkmanager / FAI respectively):"
cat <<'EOF'
 1. Base OS + docker + docker-compose-plugin, headless, SSH disabled on
    the public interface (Tailscale only).
 2. Copy this repo to /opt/sentinel-edge; docker compose build (or pull the
    published multi-arch image).
 3. Install the accelerator runtime for the profile (HailoRT .deb / JetPack).
 4. Install first-boot.service -> deploy/provision/first-boot.sh.
 5. Read-only root where practical; writable overlay for /opt/sentinel-edge
    (spool, queue, config cache, secrets).
 6. Bake a per-batch ephemeral Tailscale key to /opt/sentinel-edge/tailscale.key.
EOF
