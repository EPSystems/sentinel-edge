#!/usr/bin/env bash
# First-boot service (baked into the golden image, buildspec §6.3):
# generates a human-readable claim code, displays it on the console, and
# waits for the operator-run claim.py to produce .env before starting the
# compose stack. Outbound-only: no SSH on the public path; remote access is
# Tailscale, joined here with an ephemeral auth key baked per-batch.
set -euo pipefail

EDGE_DIR=/opt/sentinel-edge
CLAIM_CODE_FILE="$EDGE_DIR/claim-code"

if [[ ! -f "$CLAIM_CODE_FILE" ]]; then
  # 8-char unambiguous code (no 0/O/1/I)
  tr -dc 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789' </dev/urandom | head -c 8 > "$CLAIM_CODE_FILE"
fi

echo "=============================================="
echo "  Sentinel Edge — claim code: $(cat "$CLAIM_CODE_FILE")"
echo "  Enter this code in the dashboard to bind the"
echo "  device, then run deploy/provision/claim.py"
echo "=============================================="

if command -v tailscale >/dev/null && [[ -f "$EDGE_DIR/tailscale.key" ]]; then
  tailscale up --authkey "file:$EDGE_DIR/tailscale.key" --ssh --hostname "sentinel-$(cat "$CLAIM_CODE_FILE")" || true
fi

until [[ -f "$EDGE_DIR/.env" ]]; do sleep 5; done
cd "$EDGE_DIR" && docker compose -f deploy/docker-compose.yml up -d
