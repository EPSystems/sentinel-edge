"""`sentinel-edge zones` — draw lines and zones on a live camera snapshot in
the browser and write them into a device-config JSON.

Bench-only editor for the local loop: the page (zones_page.html) posts the
zone list back here, we validate through the same contract models the cloud
uses (`contracts.Zone`), and rewrite the config atomically with a bumped
config_etag — so the file stays a valid `DeviceConfig` that `local` mode (or
later the real cloud) can serve unchanged. Localhost-only bind.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

from ..config import SecretResolver, redact_rtsp
from ..contracts import DeviceConfig, Zone, uuid7

log = logging.getLogger(__name__)

SNAPSHOT_TIMEOUT_S = 30.0

# Rule fields the page may send; anything else is dropped before validation
# (contracts.Rule is additive/extra-allow, so unknown junk would slip through).
_RULE_FIELDS = ("rule_type", "classes", "min_confidence", "direction", "anchor",
                "dwell_s", "cooldown_s", "speed_limit_kmh")


# ---------------------------------------------------------------- config edit


def apply_zones(config: DeviceConfig, camera_id: str,
                zones_payload: list[dict[str, Any]]) -> DeviceConfig:
    """Replace one camera's zone list. Validates every zone through the
    contract model; raises ValueError with a readable message on bad input."""
    camera_ids = [c.camera_id for c in config.cameras]
    if camera_id not in camera_ids:
        raise ValueError(f"unknown camera_id {camera_id!r} (have {camera_ids})")
    zones: list[Zone] = []
    for i, raw in enumerate(zones_payload):
        cleaned = {
            "zone_id": raw.get("zone_id") or uuid7(),
            "name": (raw.get("name") or f"Zone {i + 1}").strip(),
            "polygon": raw.get("polygon"),
            "rules": [_clean_rule(r) for r in raw.get("rules", [])],
        }
        try:
            zones.append(Zone.model_validate(cleaned))
        except Exception as e:
            raise ValueError(f"zone {i + 1} ({cleaned['name']!r}) invalid: {e}") from e
    data = config.model_dump(mode="json")
    for cam in data["cameras"]:
        if cam["camera_id"] == camera_id:
            cam["zones"] = [z.model_dump(mode="json", exclude_none=True) for z in zones]
    data["config_etag"] = f'W/"local-{int(time.time())}"'
    return DeviceConfig.model_validate(data)


def _clean_rule(raw: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k in _RULE_FIELDS:
        v = raw.get(k)
        if v in (None, "", []):
            continue
        out[k] = v
    return out


def write_config(path: Path, config: DeviceConfig) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(config.model_dump(mode="json"), indent=2) + "\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------- snapshot


async def grab_snapshot(rtsp_url: str) -> bytes:
    """One JPEG frame off the camera, via the same decoder the pipeline uses
    (PyAV preferred, FFmpeg fallback)."""
    try:
        import cv2
    except ImportError as e:
        raise SystemExit(
            "the zones tool needs OpenCV — pip install 'sentinel-edge[onnx]'") from e
    from ..streams import StreamDecoder
    decoder = StreamDecoder("snapshot", rtsp_url, detect_fps=2)
    gen = decoder.frames()
    try:
        _, _, frame = await asyncio.wait_for(gen.__anext__(), SNAPSHOT_TIMEOUT_S)
    except asyncio.TimeoutError:
        raise SystemExit(f"no frame from {redact_rtsp(rtsp_url)} within "
                         f"{SNAPSHOT_TIMEOUT_S:.0f}s — check the URL/network")
    finally:
        decoder.stop()
        await gen.aclose()
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise SystemExit("could not JPEG-encode the snapshot frame")
    return buf.tobytes()


# ---------------------------------------------------------------- server


class ZonesServer:
    def __init__(self, config_path: Path, camera_id: str, snapshot_jpeg: bytes):
        self.config_path = config_path
        self.camera_id = camera_id
        self.snapshot = snapshot_jpeg
        self._page_path = Path(__file__).parent / "zones_page.html"

    # each request re-reads the file so an external edit isn't clobbered
    def _load(self) -> DeviceConfig:
        return DeviceConfig.model_validate_json(
            self.config_path.read_text(encoding="utf-8"))

    def state_json(self) -> bytes:
        config = self._load()
        cam = next(c for c in config.cameras if c.camera_id == self.camera_id)
        return json.dumps({
            "camera_id": cam.camera_id,
            "camera_name": cam.name,
            "config_etag": config.config_etag,
            "zones": [z.model_dump(mode="json", exclude_none=True) for z in cam.zones],
        }).encode()

    def save(self, body: bytes) -> bytes:
        payload = json.loads(body.decode("utf-8"))
        config = apply_zones(self._load(), self.camera_id, payload["zones"])
        write_config(self.config_path, config)
        log.info("saved %d zone(s) to %s (etag %s)",
                 len(payload["zones"]), self.config_path, config.config_etag)
        return json.dumps({"ok": True, "config_etag": config.config_etag}).encode()

    def make_server(self, port: int) -> ThreadingHTTPServer:
        """Bind (port 0 = ephemeral, used by tests) and return the server."""
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def _send(self, code: int, ctype: str, body: bytes) -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    page = server._page_path.read_bytes()
                    self._send(200, "text/html; charset=utf-8", page)
                elif self.path == "/snapshot.jpg":
                    self._send(200, "image/jpeg", server.snapshot)
                elif self.path == "/state":
                    self._send(200, "application/json", server.state_json())
                else:
                    self.send_error(404)

            def do_POST(self):
                if self.path != "/save":
                    self.send_error(404)
                    return
                length = int(self.headers.get("Content-Length", "0"))
                try:
                    body = server.save(self.rfile.read(length))
                    self._send(200, "application/json", body)
                except (ValueError, KeyError, json.JSONDecodeError) as e:
                    self._send(400, "application/json",
                               json.dumps({"ok": False, "error": str(e)}).encode())

        return ThreadingHTTPServer(("127.0.0.1", port), Handler)

    def serve_forever(self, port: int) -> None:
        httpd = self.make_server(port)
        print(f"zone editor: http://127.0.0.1:{httpd.server_address[1]}/  (Ctrl+C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            httpd.server_close()


def run_zones_tool(config_path: Path, camera_id: Optional[str] = None,
                   port: int = 8091, data_dir: Path = Path("./data")) -> None:
    config = DeviceConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
    if not config.cameras:
        raise SystemExit("config has no cameras")
    if camera_id is None:
        cam = config.cameras[0]
        if len(config.cameras) > 1:
            log.info("multiple cameras in config — editing %s (%s); pass --camera "
                     "to pick another", cam.camera_id, cam.name)
    else:
        try:
            cam = next(c for c in config.cameras if c.camera_id == camera_id)
        except StopIteration:
            raise SystemExit(f"camera {camera_id!r} not in config "
                             f"(have {[c.camera_id for c in config.cameras]})")
    rtsp_url = SecretResolver(data_dir).resolve(cam.rtsp_ref)
    print(f"grabbing a snapshot from {redact_rtsp(rtsp_url)} ...")
    snapshot = asyncio.run(grab_snapshot(rtsp_url))
    ZonesServer(config_path, cam.camera_id, snapshot).serve_forever(port)
