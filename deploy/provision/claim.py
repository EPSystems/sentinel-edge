#!/usr/bin/env python3
"""First-boot claim helper (Flow D).

The claim exchange is operator-mediated in v1: the operator claims the device
in the dashboard (POST /v1/app/devices/claim shows the per-device token ONCE)
and pastes it here. This script stores it with tight permissions and writes
the .env consumed by docker compose.

Usage:  python claim.py --device-id <uuid> --cloud https://cloud.example.com
"""
from __future__ import annotations

import argparse
import getpass
import os
import stat
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", required=True)
    parser.add_argument("--cloud", required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()

    token = getpass.getpass("Per-device token (shown once in the dashboard): ").strip()
    if len(token) < 32:
        raise SystemExit("that does not look like a device token")

    args.env_file.write_text(
        f"SENTINEL_CLOUD_BASE_URL={args.cloud}\n"
        f"SENTINEL_DEVICE_ID={args.device_id}\n"
        f"SENTINEL_DEVICE_TOKEN={token}\n",
        encoding="utf-8",
    )
    if os.name == "posix":
        args.env_file.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0600
    print(f"claim material written to {args.env_file} — start the stack with "
          "`docker compose up -d`; the device should turn green in the "
          "dashboard within 90s.")


if __name__ == "__main__":
    main()
