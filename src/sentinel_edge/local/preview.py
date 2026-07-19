"""Live preview for `sentinel-edge local` — an MJPEG stream over localhost
HTTP with zones, lines (A->B arrow), track boxes and event flashes drawn in.

Served over HTTP instead of cv2.imshow because the [onnx] extra ships
opencv-*headless* (no GUI); imencode works everywhere. Localhost-only bind —
this is a bench tool, not a product surface, and raw frames must not leave
the machine (the no-raw-video invariant).
"""
from __future__ import annotations

import html
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Optional

import numpy as np

from ..rules import geometry

if TYPE_CHECKING:
    from ..config import PipelineDefaults
    from ..contracts import CameraConfig

log = logging.getLogger(__name__)

JPEG_QUALITY = 80
EVENT_FLASH_S = 3.0

# BGR
COL_ZONE = (80, 200, 80)
COL_LINE = (0, 165, 255)
COL_TRACK = (255, 200, 0)
COL_EVENT = (0, 0, 255)


class PreviewServer:
    """push() is called from the pipeline's event loop; HTTP threads block on
    a Condition until a fresh frame for their camera arrives."""

    def __init__(self, port: int, defaults: "PipelineDefaults"):
        try:
            import cv2  # noqa: F401
        except ImportError as e:
            raise SystemExit(
                "--preview needs OpenCV — pip install 'sentinel-edge[onnx]'") from e
        self.port = port
        self.defaults = defaults
        self._jpegs: dict[str, bytes] = {}          # camera_id -> latest frame
        self._names: dict[str, str] = {}            # camera_id -> display name
        self._flash: dict[str, tuple[float, str]] = {}  # camera_id -> (until, text)
        self._cond = threading.Condition()
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------ lifecycle

    def start(self) -> None:
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):  # keep pipeline logs readable
                pass

            def do_GET(self):
                server._handle(self)

        self._httpd = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        name="preview-http", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._httpd:
            self._httpd.shutdown()
            self._httpd.server_close()
        with self._cond:
            self._cond.notify_all()

    # ------------------------------------------------------------ pipeline tap

    def push(self, camera: "CameraConfig", frame: np.ndarray,
             tracks: list, candidates: list, line_counts: dict) -> None:
        """FrameTap: render the overlay and publish the JPEG. Exceptions are
        contained by the pipeline's tap guard, but stay cheap here."""
        now = time.monotonic()
        if candidates:
            text = ", ".join(f"{c.rule_type.value} {c.object_class.value}"
                             for c in candidates)
            self._flash[camera.camera_id] = (now + EVENT_FLASH_S, text)
        annotated = self._render(camera, frame, tracks, now, line_counts)
        import cv2
        ok, buf = cv2.imencode(".jpg", annotated,
                               [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            return
        with self._cond:
            self._jpegs[camera.camera_id] = buf.tobytes()
            self._names[camera.camera_id] = camera.name
            self._cond.notify_all()

    def _render(self, camera: "CameraConfig", frame: np.ndarray,
                tracks: list, now: float, line_counts: dict) -> np.ndarray:
        import cv2
        img = frame.copy()
        h, w = img.shape[:2]
        for zone in camera.zones:
            pts = geometry.denormalize(zone.polygon, w, h)
            if geometry.polygon_is_line(zone.polygon):
                a, b = (tuple(map(int, p)) for p in pts)
                cv2.arrowedLine(img, a, b, COL_LINE, 2, tipLength=0.04)
                counts = line_counts.get(zone.zone_id)
                label = f"{zone.name} (A->B)"
                if counts:
                    label += f"  lr={counts['lr']} rl={counts['rl']} total={counts['total']}"
                cv2.putText(img, label, (a[0] + 6, a[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_LINE, 2)
            else:
                poly = np.array(pts, dtype=np.int32)
                cv2.polylines(img, [poly], True, COL_ZONE, 2)
                cx, cy = poly.mean(axis=0).astype(int)
                cv2.putText(img, zone.name, (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_ZONE, 2)
        for t in tracks:
            x1, y1, x2, y2 = (int(v) for v in t.bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), COL_TRACK, 2)
            ax, ay = (int(v) for v in t.anchor(self.defaults.anchor))
            cv2.circle(img, (ax, ay), 4, COL_TRACK, -1)
            cv2.putText(img, f"{t.object_class.value} {t.confidence:.2f} #{t.track_id}",
                        (x1, max(y1 - 8, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_TRACK, 2)
        flash = self._flash.get(camera.camera_id)
        if flash and now < flash[0]:
            cv2.rectangle(img, (0, 0), (w - 1, h - 1), COL_EVENT, 6)
            cv2.putText(img, f"EVENT: {flash[1]}", (16, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, COL_EVENT, 2)
        return img

    # ------------------------------------------------------------ http

    def _handle(self, req: BaseHTTPRequestHandler) -> None:
        if req.path in ("/", "/index.html"):
            self._serve_index(req)
        elif req.path.startswith("/stream/"):
            camera_id = req.path[len("/stream/"):].removesuffix(".mjpg")
            self._serve_stream(req, camera_id)
        else:
            req.send_error(404)

    def _serve_index(self, req: BaseHTTPRequestHandler) -> None:
        with self._cond:
            cams = dict(self._names)
        streams = "".join(
            f"<figure><figcaption>{html.escape(name)}</figcaption>"
            f'<img src="/stream/{cid}.mjpg" alt="{html.escape(name)}"></figure>'
            for cid, name in cams.items()
        ) or "<p>Waiting for the first frame&hellip; refresh in a few seconds.</p>"
        body = (
            "<!doctype html><meta charset='utf-8'><title>sentinel-edge preview</title>"
            "<style>body{background:#111;color:#ddd;font-family:sans-serif;margin:1rem}"
            "img{max-width:100%;border:1px solid #444}figure{margin:0 0 1rem 0}"
            "figcaption{margin-bottom:.3rem}</style>"
            f"<h1>sentinel-edge local preview</h1>{streams}"
        ).encode()
        req.send_response(200)
        req.send_header("Content-Type", "text/html; charset=utf-8")
        req.send_header("Content-Length", str(len(body)))
        req.end_headers()
        req.wfile.write(body)

    def _serve_stream(self, req: BaseHTTPRequestHandler, camera_id: str) -> None:
        req.send_response(200)
        req.send_header("Content-Type",
                        "multipart/x-mixed-replace; boundary=frame")
        req.send_header("Cache-Control", "no-store")
        req.end_headers()
        last: Optional[bytes] = None
        try:
            while self._httpd is not None:
                with self._cond:
                    self._cond.wait(timeout=2.0)
                    jpeg = self._jpegs.get(camera_id)
                if jpeg is None or jpeg is last:
                    continue
                last = jpeg
                req.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                                + jpeg + b"\r\n")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # viewer closed the tab
