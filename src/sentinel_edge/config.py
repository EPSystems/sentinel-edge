"""Device configuration, layered lowest- to highest-precedence:

1. built-in defaults          (PipelineDefaults below)
2. accelerator profile YAML   (configs/profiles/accel-*.yaml)
3. vertical profile YAML      (configs/profiles/vertical-*.yaml)
4. cloud-pushed tenant config (contracts.DeviceConfig — per tenant, hot-reloaded)
5. per-camera / per-rule fields inside the tenant config

Cloud authors, edge executes: zones/rules are never edited on-device in
production; the tenant config arrives via cloudlink and is cached to disk
(last-known-config) so a reboot during a cloud outage still runs the last
good ruleset.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .contracts import DeviceConfig

log = logging.getLogger(__name__)


class EdgeSettings(BaseSettings):
    """Per-device environment (set at provision time; never in the repo)."""

    model_config = SettingsConfigDict(env_prefix="SENTINEL_", env_file=".env", extra="ignore")

    cloud_base_url: str = "https://cloud.example.com"  # /v1 is appended by clients
    device_token: str = ""                             # per-device bearer, claim-issued
    device_id: str = ""
    data_dir: Path = Path("./data")
    profiles_dir: Path = Path("./configs/profiles")
    accel_profile: str = "accel-onnx-cpu"
    vertical_profile: str = "vertical-petrol"
    model_version: str = "yolox-s-v0.0-dev"
    heartbeat_interval_s: float = 30.0
    log_level: str = "INFO"

    @property
    def api_base(self) -> str:
        return self.cloud_base_url.rstrip("/") + "/v1"

    @property
    def ws_url(self) -> str:
        base = self.cloud_base_url.rstrip("/")
        base = base.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
        return base + "/v1/edge/link"


class PipelineDefaults(BaseModel):
    """Tunables merged from profile YAMLs. Every value here can be overridden
    by an accelerator or vertical profile; rule-level fields in the tenant
    config override these again per rule."""

    model_config = ConfigDict(extra="forbid")

    # detect
    backend: str = "onnx"                 # onnx | hailo | tensorrt | mock
    model_path: str = "models/yolox-s.onnx"
    model_input_size: int = 640
    conf_threshold: float = 0.70          # CANONICAL-FACTS §6 default
    min_object_px: int = 12               # drop far-field noise (bbox min side)
    class_map: dict[int, str] = Field(
        # COCO ids -> contract classes; a custom-trained 5-class model replaces this
        default_factory=lambda: {
            0: "person", 2: "car", 3: "motorcycle", 5: "truck", 7: "truck",
            14: "animal", 15: "animal", 16: "animal", 17: "animal", 18: "animal", 19: "animal",
        }
    )
    # motion gate
    motion_enabled: bool = True
    motion_downscale: int = 4
    motion_pixel_thresh: int = 25
    motion_min_fraction: float = 0.003
    motion_learn_rate: float = 0.05
    # tracker
    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.1
    track_iou_thresh: float = 0.3
    track_ttl_s: float = 2.0
    track_history_len: int = 90
    # rules
    rule_cooldown_s: float = 30.0         # debounce per (track, rule) — alert-storm guard
    min_track_age_frames: int = 3
    dwell_default_s: float = 60.0
    anchor: str = "bottom"                # bottom = feet/wheels; center for gates
    # record
    clip_pre_s: float = 5.0
    clip_post_s: float = 5.0
    spool_max_mb: int = 2048
    # publish backpressure
    queue_thumb_drop_depth: int = 2000
    queue_microclip_depth: int = 5000


def load_defaults(settings: EdgeSettings) -> PipelineDefaults:
    """defaults <- accel profile <- vertical profile (later wins)."""
    merged: dict[str, Any] = {}
    for name in (settings.accel_profile, settings.vertical_profile):
        path = settings.profiles_dir / f"{name}.yaml"
        if not path.exists():
            log.warning("profile %s not found at %s — skipping", name, path)
            continue
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        merged.update(data)
    if "class_map" in merged:
        merged["class_map"] = {int(k): v for k, v in merged["class_map"].items()}
    return PipelineDefaults(**merged)


class ConfigCache:
    """Atomic last-known-config cache so a reboot during a cloud outage
    still runs the last good ruleset (buildspec §4.8 step 3)."""

    def __init__(self, data_dir: Path):
        self.path = Path(data_dir) / "last-known-config.json"

    def load(self) -> Optional[DeviceConfig]:
        if not self.path.exists():
            return None
        try:
            return DeviceConfig.model_validate_json(self.path.read_text(encoding="utf-8"))
        except Exception:
            log.exception("cached config is corrupt — ignoring %s", self.path)
            return None

    def store(self, config: DeviceConfig) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(config.model_dump_json())
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise


class SecretResolver:
    """Resolves the opaque `rtsp_ref` from the tenant config to a real RTSP URL.

    Credentials are delivered out-of-band at provision/claim time into
    <data_dir>/secrets.json ({"<rtsp_ref>": "rtsp://user:pass@host/path"}),
    decrypted only in edge memory, never logged (use redact_rtsp for logs).
    """

    def __init__(self, data_dir: Path):
        self.path = Path(data_dir) / "secrets.json"

    def resolve(self, rtsp_ref: str) -> str:
        if rtsp_ref.startswith(("rtsp://", "rtsps://")):
            return rtsp_ref  # dev/simulate convenience — production refs are opaque
        if self.path.exists():
            secrets = json.loads(self.path.read_text(encoding="utf-8"))
            if rtsp_ref in secrets:
                return secrets[rtsp_ref]
        raise KeyError(f"no secret material for rtsp_ref {rtsp_ref!r}")


def redact_rtsp(url: str) -> str:
    """rtsp://user:pass@host/... -> rtsp://user:***@host/... (grep-tested)."""
    if "@" not in url or "://" not in url:
        return url
    scheme, rest = url.split("://", 1)
    creds, host = rest.rsplit("@", 1)
    user = creds.split(":", 1)[0]
    return f"{scheme}://{user}:***@{host}"
