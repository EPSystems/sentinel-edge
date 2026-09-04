"""Pydantic mirrors of the shared edge<->cloud contract (sentinel-contracts /v1).

Source of truth: sentinel-cloud/packages/contracts/schemas/*.schema.json.
These models must stay byte-compatible with those JSON Schemas — additive
changes only within v1. Do NOT add wire fields here without changing the
schema first (ARCHITECTURE.md §10).

Forward-compat rules implemented per common.schema.json:
- unknown object_class values coerce to "unknown" (never reject),
- unknown WS envelope `type` values are tolerated by the codec (link.py),
- top-level unknown fields ARE rejected (additionalProperties: false),
  except Rule, which is additive by design.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

CONTRACT_VERSION = "1.0.0"
SCHEMA_VERSION: Literal["1"] = "1"


class ObjectClass(str, Enum):
    person = "person"
    car = "car"
    truck = "truck"
    motorcycle = "motorcycle"
    animal = "animal"
    unknown = "unknown"

    @classmethod
    def coerce(cls, value: str) -> "ObjectClass":
        try:
            return cls(value)
        except ValueError:
            return cls.unknown


class RuleType(str, Enum):
    zone_enter = "zone_enter"
    loiter_dwell = "loiter_dwell"
    line_cross = "line_cross"
    speed = "speed"
    driveoff = "driveoff"
    lpr_hit = "lpr_hit"


# Which point of a track's bbox defines a line crossing. "bottom" (feet/wheels,
# on the ground plane) is the default; "top" (head) is for high-mount cameras
# where only the head is reliably in frame at the line. Applies to line_cross.
ANCHOR_KINDS = ("bottom", "center", "top")


def utcnow_rfc3339() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------- event


class EventPayload(_Strict):
    """POST /v1/edge/events body. event_id is the idempotency key."""

    schema_version: Literal["1"] = SCHEMA_VERSION
    event_id: str
    camera_id: str
    zone_id: Optional[str] = None
    object_class: ObjectClass
    rule_type: RuleType
    confidence: float = Field(ge=0, le=1)
    track_id: Optional[int] = None
    clip_object_key: str = Field(max_length=512)
    thumb_object_key: Optional[str] = Field(default=None, max_length=512)
    plate_number: Optional[str] = Field(default=None, max_length=16)
    plate_confidence: Optional[float] = Field(default=None, ge=0, le=1)
    speed_kmh: Optional[float] = Field(default=None, ge=0, le=400)
    metadata: dict[str, Any] = Field(default_factory=dict)
    occurred_at: str

    @field_validator("object_class", mode="before")
    @classmethod
    def _coerce_class(cls, v: Any) -> Any:
        return ObjectClass.coerce(v) if isinstance(v, str) else v


# ---------------------------------------------------------------- config


class Rule(BaseModel):
    """A rule attached to a zone. additionalProperties:true in the schema —
    per-rule_type extras stay additive, so extra='allow' here. The typed
    optional fields below are the extras the edge engine understands today."""

    model_config = ConfigDict(extra="allow")

    rule_type: RuleType
    classes: Optional[list[ObjectClass]] = None
    min_confidence: Optional[float] = Field(default=None, ge=0, le=1)
    watchlist_ref: Optional[str] = None
    # engine-understood additive extras
    dwell_s: Optional[float] = None            # loiter_dwell
    direction: Optional[str] = None            # line_cross: "lr" | "rl" | "both"
    anchor: Optional[str] = None               # line_cross: "bottom" | "center" | "top"
    speed_limit_kmh: Optional[float] = None    # speed
    cooldown_s: Optional[float] = None         # any rule; default from profile
    exit_zone_ref: Optional[str] = None        # driveoff: zone_id of the exit line
    shop_camera_ref: Optional[str] = None      # driveoff: camera watched for person entry
    grace_s: Optional[float] = None            # driveoff

    @field_validator("classes", mode="before")
    @classmethod
    def _coerce_classes(cls, v: Any) -> Any:
        if isinstance(v, list):
            return [ObjectClass.coerce(c) if isinstance(c, str) else c for c in v]
        return v

    @field_validator("anchor")
    @classmethod
    def _validate_anchor(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ANCHOR_KINDS:
            raise ValueError(f"anchor must be one of {ANCHOR_KINDS} or omitted, got {v!r}")
        return v


class Zone(_Strict):
    zone_id: str
    name: str
    polygon: list[list[float]]  # normalized 0..1; 2 points = directed line A->B
    rules: list[Rule]

    @field_validator("polygon")
    @classmethod
    def _validate_polygon(cls, v: list[list[float]]) -> list[list[float]]:
        if len(v) < 2:
            raise ValueError("polygon needs at least 2 points")
        for pt in v:
            if len(pt) != 2 or not all(0.0 <= c <= 1.0 for c in pt):
                raise ValueError("polygon points must be [x, y] normalized to 0..1")
        return v


class CameraConfig(_Strict):
    camera_id: str
    name: str
    rtsp_ref: str  # OPAQUE — creds resolved out-of-band, never cleartext on the wire
    detect_fps: int = Field(ge=1)
    lpr_enabled: bool
    calibration: Optional[dict[str, Any]] = None  # {"px_per_m": ...} or {"homography": [[..]x3]}
    zones: list[Zone] = Field(default_factory=list)


class DeviceConfig(_Strict):
    """GET /v1/edge/config body; also the config.push WS payload."""

    schema_version: Literal["1"] = SCHEMA_VERSION
    config_etag: str
    device_id: str
    retention_days: int = Field(ge=0)
    cameras: list[CameraConfig]


# ---------------------------------------------------------------- heartbeat


class DeviceHealth(_Strict):
    status: Literal["online", "degraded", "offline"]
    cpu_pct: Optional[float] = None
    mem_pct: Optional[float] = None
    accel_temp_c: Optional[float] = None
    accel_util_pct: Optional[float] = None
    disk_free_pct: Optional[float] = None
    queue_depth: Optional[int] = None
    queue_oldest_age_s: Optional[float] = None
    clock_skew_s: Optional[float] = None


class CameraHealth(_Strict):
    camera_id: str
    state: Literal["streaming", "down", "reconnecting"]
    fps: Optional[float] = None
    last_frame_age_s: Optional[float] = None
    last_event_age_s: Optional[float] = None


class Heartbeat(_Strict):
    schema_version: Literal["1"] = SCHEMA_VERSION
    device_id: str
    contract_version: str
    model_version: str
    config_etag: str
    device: DeviceHealth
    cameras: list[CameraHealth]
    sent_at: str


# ---------------------------------------------------------------- WS envelope


KNOWN_WS_TYPES = {
    "config.push",
    "model.update",
    "heartbeat",
    "command",
    "liveview.offer",
    "liveview.answer",
    "liveview.ice",
    "ping",
    "pong",
}


class WsEnvelope(_Strict):
    """Every WS frame. `type` is a plain str: unknown values from a newer
    peer must be tolerated (logged + ignored), not rejected at parse time."""

    v: Literal[1] = 1
    type: str
    id: str
    ts: str
    ack: Optional[str] = None
    payload: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------- presign


class PresignRequest(_Strict):
    event_id: str
    camera_id: str
    kind: Literal["clip", "thumb"]
    content_type: str
    content_length: int = Field(ge=1, le=20_971_520)


class PresignResponse(_Strict):
    url: str
    method: Literal["PUT"]
    object_key: str
    expires_in: int
    headers: dict[str, str] = Field(default_factory=dict)
