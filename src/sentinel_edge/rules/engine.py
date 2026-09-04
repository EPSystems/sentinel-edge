"""Per-camera rule engine: evaluates the cloud-authored ruleset against
tracks and emits EventCandidates. Zones are authored in the dashboard and
hot-reloaded here (Flow B) — swap_config() atomically replaces the compiled
ruleset between frames; an invalid config never reaches this module because
it fails contract validation upstream and the previous ruleset stays live.

Debounce: one continuous presence = one event. Cooldown per (track, rule),
default 30 s — WhatsApp/Twilio is the dominant variable cost, so alert storms
are a money bug, not just noise (CANONICAL-FACTS §8).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from ..config import PipelineDefaults
from ..contracts import CameraConfig, ObjectClass, Rule, RuleType
from ..track import Track, anchor_of
from . import geometry
from .speed import SpeedEstimator

log = logging.getLogger(__name__)


@dataclass
class EventCandidate:
    """A rule firing, pre-clip. The pipeline attaches clip/thumb keys and
    turns it into a contract EventPayload."""

    camera_id: str
    zone_id: Optional[str]
    rule_type: RuleType
    object_class: ObjectClass
    confidence: float
    track_id: int
    ts: float  # wall clock, epoch seconds
    speed_kmh: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Signal:
    """Internal cross-camera primitive consumed by the supervisor-level
    drive-off correlator. Never published to the cloud directly."""

    kind: str  # "zone_enter" | "line_cross" | "dwell_complete"
    camera_id: str
    zone_id: str
    track_id: int
    object_class: ObjectClass
    ts: float
    direction: Optional[str] = None


@dataclass
class _CompiledZone:
    zone_id: str
    name: str
    polygon_px: list[tuple[float, float]]
    is_line: bool
    rules: list[Rule]
    anchor: str  # which bbox point defines a crossing (line zones); default from profile


class RuleEngine:
    def __init__(self, camera: CameraConfig, defaults: PipelineDefaults,
                 frame_size: tuple[int, int]):
        self.defaults = defaults
        self.frame_w, self.frame_h = frame_size
        self.camera_id = camera.camera_id
        self._zones: list[_CompiledZone] = []
        self._speed = SpeedEstimator(camera.calibration)
        # state
        self._inside_since: dict[tuple[int, str], float] = {}   # (track, zone) -> entered ts
        self._dwell_fired: set[tuple[int, str]] = set()          # fired this presence
        # last bbox per track, so a line's own anchor (feet/center/head) can be
        # recomputed for the previous frame even when lines use different anchors.
        self._last_bbox: dict[int, tuple[float, float, float, float]] = {}
        self._cooldown: dict[tuple[int, str], float] = {}        # (track, rule_key) -> last fired
        self._last_seen: dict[int, float] = {}
        self._line_counts: dict[str, dict[str, int]] = {}        # zone_id -> {"lr": n, "rl": n}
        self._compile(camera)

    # ------------------------------------------------------------ config

    def _compile(self, camera: CameraConfig) -> None:
        zones: list[_CompiledZone] = []
        for z in camera.zones:
            try:
                geometry.validate_zone_polygon(z.polygon)
            except ValueError as e:
                log.warning("zone %s (%s) skipped: %s", z.zone_id, z.name, e)
                continue
            is_line = geometry.polygon_is_line(z.polygon)
            zones.append(_CompiledZone(
                zone_id=z.zone_id,
                name=z.name,
                polygon_px=geometry.denormalize(z.polygon, self.frame_w, self.frame_h),
                is_line=is_line,
                rules=list(z.rules),
                anchor=self._resolve_line_anchor(z.rules) if is_line else self.defaults.anchor,
            ))
        self._zones = zones

    def _resolve_line_anchor(self, rules: list[Rule]) -> str:
        """A line's crossing anchor comes from its line_cross rule(s) — the whole
        line (crossing test, count, signal, every line_cross rule on it) shares
        one anchor. First line_cross rule that pins an anchor wins; otherwise the
        camera/profile default. Authored per-line in the zones editor."""
        for r in rules:
            if r.rule_type == RuleType.line_cross and r.anchor:
                return r.anchor
        return self.defaults.anchor

    def swap_config(self, camera: CameraConfig) -> None:
        """Hot-reload (Flow B). Presence/line state resets — the new geometry
        invalidates it — but cooldowns survive so a config edit can't unleash
        an alert storm from already-alerted tracks."""
        self._compile(camera)
        self._speed = SpeedEstimator(camera.calibration)
        self._inside_since.clear()
        self._dwell_fired.clear()
        self._last_bbox.clear()

    def driveoff_rules(self) -> list[tuple[_CompiledZone, Rule]]:
        return [(z, r) for z in self._zones for r in z.rules
                if r.rule_type == RuleType.driveoff]

    def lpr_rules(self) -> list[tuple[_CompiledZone, Rule]]:
        return [(z, r) for z in self._zones for r in z.rules
                if r.rule_type == RuleType.lpr_hit]

    def line_counts(self) -> dict[str, dict[str, Any]]:
        """zone_id -> {"name", "lr", "rl", "total"} for every line zone that has
        registered at least one crossing. Counts every geometric crossing
        (independent of alert cooldown — the cooldown throttles notifications,
        not the underlying physical event)."""
        names = {z.zone_id: z.name for z in self._zones}
        return {
            zid: {"name": names.get(zid, zid), "lr": c.get("lr", 0), "rl": c.get("rl", 0),
                  "total": c.get("lr", 0) + c.get("rl", 0)}
            for zid, c in self._line_counts.items()
        }

    # ------------------------------------------------------------ evaluate

    def update(self, tracks: list[Track], ts: float) -> tuple[list[EventCandidate], list[Signal]]:
        candidates: list[EventCandidate] = []
        signals: list[Signal] = []

        for track in tracks:
            self._last_seen[track.track_id] = ts
            # Presence and speed use the profile default anchor (feet stay on the
            # ground plane — required for metric speed). Line zones may override
            # per line, so their anchor is resolved inside the loop.
            default_anchor = track.anchor(self.defaults.anchor)
            prev_bbox = self._last_bbox.get(track.track_id)
            speed_kmh = self._speed.update(track.track_id, ts, default_anchor)
            eligible = track.age_frames >= self.defaults.min_track_age_frames

            for zone in self._zones:
                if zone.is_line:
                    curr = track.anchor(zone.anchor)
                    prev = anchor_of(prev_bbox, zone.anchor) if prev_bbox is not None else None
                    self._eval_line(zone, track, prev, curr, ts, eligible,
                                    candidates, signals)
                else:
                    self._eval_polygon(zone, track, default_anchor, ts, eligible, speed_kmh,
                                       candidates, signals)

            self._last_bbox[track.track_id] = track.bbox

        self._prune(ts, {t.track_id for t in tracks})
        return candidates, signals

    # ------------------------------------------------------------ line zones

    def _eval_line(self, zone: _CompiledZone, track: Track,
                   prev_anchor: Optional[tuple[float, float]], anchor: tuple[float, float],
                   ts: float, eligible: bool,
                   candidates: list[EventCandidate], signals: list[Signal]) -> None:
        if prev_anchor is None or not eligible:
            return
        a, b = zone.polygon_px
        crossed = geometry.crossing_direction(prev_anchor, anchor, a, b, overshoot=0.05)
        if crossed is None:
            return
        counts = self._line_counts.setdefault(zone.zone_id, {"lr": 0, "rl": 0})
        counts[crossed] = counts.get(crossed, 0) + 1
        log.info("line_cross count zone=%s (%s) direction=%s lr=%d rl=%d total=%d",
                 zone.zone_id, zone.name, crossed, counts.get("lr", 0), counts.get("rl", 0),
                 counts.get("lr", 0) + counts.get("rl", 0))
        # always signal (drive-off correlation needs exit crossings even when
        # no explicit line_cross rule is configured on the exit line)
        signals.append(Signal("line_cross", self.camera_id, zone.zone_id,
                              track.track_id, track.object_class, ts, direction=crossed))
        for rule in zone.rules:
            if rule.rule_type != RuleType.line_cross:
                continue
            wanted = rule.direction or "both"
            if wanted != "both" and wanted != crossed:
                continue
            self._fire(rule, zone, track, ts, candidates,
                       metadata={"direction": crossed})

    # ------------------------------------------------------------ polygon zones

    def _eval_polygon(self, zone: _CompiledZone, track: Track, anchor: tuple[float, float],
                      ts: float, eligible: bool, speed_kmh: Optional[float],
                      candidates: list[EventCandidate], signals: list[Signal]) -> None:
        key = (track.track_id, zone.zone_id)
        inside = geometry.point_in_polygon(anchor, zone.polygon_px)
        was_inside = key in self._inside_since

        if inside and not was_inside and eligible:
            self._inside_since[key] = ts
            signals.append(Signal("zone_enter", self.camera_id, zone.zone_id,
                                  track.track_id, track.object_class, ts))
            for rule in zone.rules:
                if rule.rule_type == RuleType.zone_enter:
                    self._fire(rule, zone, track, ts, candidates)

        elif not inside and was_inside:
            self._inside_since.pop(key, None)
            self._dwell_fired.discard(key)

        if inside and was_inside:
            dwell = ts - self._inside_since[key]
            for rule in zone.rules:
                if rule.rule_type == RuleType.loiter_dwell and key not in self._dwell_fired:
                    dwell_s = rule.dwell_s if rule.dwell_s is not None else self.defaults.dwell_default_s
                    if dwell >= dwell_s:
                        if self._fire(rule, zone, track, ts, candidates,
                                      metadata={"dwell_s": round(dwell, 1)}):
                            self._dwell_fired.add(key)
                elif rule.rule_type == RuleType.driveoff:
                    dwell_s = rule.dwell_s if rule.dwell_s is not None else self.defaults.dwell_default_s
                    if dwell >= dwell_s and key not in self._dwell_fired:
                        if self._class_ok(rule, track):
                            self._dwell_fired.add(key)
                            signals.append(Signal("dwell_complete", self.camera_id,
                                                  zone.zone_id, track.track_id,
                                                  track.object_class, ts))

        if inside and speed_kmh is not None:
            for rule in zone.rules:
                if rule.rule_type == RuleType.speed:
                    if rule.speed_limit_kmh is None:
                        continue  # unconfigured speed rule can never fire
                    if speed_kmh >= rule.speed_limit_kmh:
                        cand_meta = {"speed_limit_kmh": rule.speed_limit_kmh,
                                     "accuracy_note": "±20% deterrent-grade estimate"}
                        self._fire(rule, zone, track, ts, candidates,
                                   speed_kmh=round(speed_kmh, 1), metadata=cand_meta)

    # ------------------------------------------------------------ helpers

    def _class_ok(self, rule: Rule, track: Track) -> bool:
        if rule.classes and track.object_class not in rule.classes:
            return False
        min_conf = rule.min_confidence if rule.min_confidence is not None else self.defaults.conf_threshold
        return track.confidence >= min_conf

    def _fire(self, rule: Rule, zone: _CompiledZone, track: Track, ts: float,
              candidates: list[EventCandidate], speed_kmh: Optional[float] = None,
              metadata: Optional[dict[str, Any]] = None) -> bool:
        if not self._class_ok(rule, track):
            return False
        rule_key = f"{zone.zone_id}:{rule.rule_type.value}"
        cd_key = (track.track_id, rule_key)
        cooldown = rule.cooldown_s if rule.cooldown_s is not None else self.defaults.rule_cooldown_s
        last = self._cooldown.get(cd_key)
        if last is not None and ts - last < cooldown:
            return False
        self._cooldown[cd_key] = ts
        meta = {"zone_name": zone.name, "bbox": [round(v, 1) for v in track.bbox]}
        if metadata:
            meta.update(metadata)
        candidates.append(EventCandidate(
            camera_id=self.camera_id,
            zone_id=zone.zone_id,
            rule_type=rule.rule_type,
            object_class=track.object_class,
            confidence=track.confidence,
            track_id=track.track_id,
            ts=ts,
            speed_kmh=speed_kmh,
            metadata=meta,
        ))
        return True

    def _prune(self, ts: float, live_ids: set[int]) -> None:
        stale_after = 60.0
        dead = {tid for tid, seen in self._last_seen.items()
                if tid not in live_ids and ts - seen > stale_after}
        if not dead:
            return
        for tid in dead:
            self._last_seen.pop(tid, None)
            self._last_bbox.pop(tid, None)
        self._inside_since = {k: v for k, v in self._inside_since.items() if k[0] not in dead}
        self._dwell_fired = {k for k in self._dwell_fired if k[0] not in dead}
        self._cooldown = {k: v for k, v in self._cooldown.items() if k[0] not in dead}
        self._speed.prune(live_ids)
