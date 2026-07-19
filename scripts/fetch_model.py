#!/usr/bin/env python3
"""Download the stock YOLOX-S ONNX export (Apache-2.0, Megvii) to
models/yolox-s.onnx — the model path the accel-onnx-cpu profile expects.

The stock export is COCO-80-class with raw head output (grid-relative,
decode_in_inference=False), which is exactly what detect/backend_onnx.py
decodes; the COCO->contract class map ships in PipelineDefaults. Good enough
for the first physical-camera test — the custom 5-class re-head comes later
(HARDWARE-BRINGUP.md §2).

Usage:
    python scripts/fetch_model.py            # download + sha256
    python scripts/fetch_model.py --verify   # also load it and check the IO
                                             # shapes (needs the [onnx] extra)
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

URL = ("https://github.com/Megvii-BaseDetection/YOLOX/releases/download/"
       "0.1.1rc0/yolox_s.onnx")
# sha256 of the 0.1.1rc0 release asset, pinned 2026-07-12
EXPECTED_SHA256 = "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063"
DEST = Path(__file__).resolve().parent.parent / "models" / "yolox-s.onnx"
EXPECTED_OUTPUT = (1, 8400, 85)  # 640x640: (80^2+40^2+20^2) anchors, 5+80 cols


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}\n         -> {dest}")

    last_pct = [-1]

    def hook(blocks: int, block_size: int, total: int) -> None:
        if total <= 0:
            return
        done = blocks * block_size
        pct = int(min(done / total * 100, 100))
        if pct != last_pct[0]:  # one line per percent, log-friendly
            last_pct[0] = pct
            sys.stdout.write(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB  ({pct:3d}%)")
            sys.stdout.flush()

    tmp = dest.with_suffix(".onnx.part")
    urllib.request.urlretrieve(url, tmp, reporthook=hook)  # noqa: S310 — pinned https URL
    print()
    tmp.replace(dest)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: Path) -> None:
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        raise SystemExit("--verify needs the [onnx] extra: "
                         "pip install 'sentinel-edge[onnx]'")
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    print(f"  input : {inp.name} {inp.shape}")
    out = sess.run(None, {inp.name: np.zeros((1, 3, 640, 640), dtype=np.float32)})[0]
    print(f"  output: {out.shape}")
    if tuple(out.shape) != EXPECTED_OUTPUT:
        raise SystemExit(
            f"unexpected output shape {out.shape} != {EXPECTED_OUTPUT} — this "
            "export decodes in-graph (decode_in_inference=True) or is not "
            "640x640; backend_onnx.py expects the raw head. Re-export or flag "
            "it so the decode path gets a bypass.")
    print("  OK — raw-head 640x640 export, matches backend_onnx.py")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=URL)
    ap.add_argument("--dest", type=Path, default=DEST)
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    ap.add_argument("--verify", action="store_true", help="load and shape-check")
    args = ap.parse_args()

    if args.dest.exists() and not args.force:
        print(f"already present: {args.dest} ({args.dest.stat().st_size / 1e6:.1f} MB) "
              "— use --force to re-download")
    else:
        download(args.url, args.dest)
    digest = sha256(args.dest)
    print(f"sha256: {digest}")
    if args.url == URL and digest != EXPECTED_SHA256:
        print(f"WARNING: checksum differs from the pinned release asset "
              f"({EXPECTED_SHA256}) — the upstream file changed; run --verify "
              "before trusting it.")
    if args.verify:
        verify(args.dest)
    print(f"\nready. Point the accel-onnx-cpu profile at it (default model_path "
          f"is models/yolox-s.onnx) and run:\n"
          f"  python -m sentinel_edge local --config <your-config.json> --preview")


if __name__ == "__main__":
    main()
