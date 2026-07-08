"""FuelGuard drive-off correlation (buildspec §4.4 note).

Composite rule spanning two cameras, so it runs in the supervisor, not a
single camera pipeline: *vehicle dwells in a pump zone, then crosses the exit
line, with no person `zone_enter` on the shop-entrance camera within grace_s
looking BACK from the exit* (a paying customer walks into the shop before
driving away; a drive-off doesn't).

Consumes internal Signals from every camera's RuleEngine and emits a
driveoff EventCandidate attributed to the pump zone.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional

from ..contracts import ObjectClass, Rule, RuleType
from .engine import EventCandidate, Signal

log = logging.getLogger(__name__)

VEHICLE_CLASSES = {ObjectClass.car, ObjectClass.truck, ObjectClass.motorcycle}


@dataclass
class DriveoffRule:
    camera_id: str        # camera hosting the pump zone
    pump_zone_id: str
    exit_zone_id: str     # the exit line's zone_id (usually same camera)
    grace_s: float = 90.0
    exit_window_s: float = 180.0
    shop_camera_id: Optional[str] = None  # None -> person entries on ANY camera count


class DriveoffCorrelator:
    def __init__(self, rules: list[DriveoffRule]):
        self.rules = rules
        self._dwell_done: dict[tuple[str, str, int], float] = {}  # (cam, pump_zone, track) -> ts
        self._person_entries: deque[tuple[float, str]] = deque()  # (ts, camera_id)
        self._max_lookback = max((r.grace_s for r in rules), default=90.0) + 60.0

    @classmethod
    def from_config(cls, per_camera_rules: list[tuple[str, str, Rule]]) -> "DriveoffCorrelator":
        """per_camera_rules: (camera_id, pump_zone_id, driveoff Rule)."""
        compiled = []
        for camera_id, zone_id, rule in per_camera_rules:
            if rule.rule_type != RuleType.driveoff:
                continue
            if not rule.exit_zone_ref:
                log.warning("driveoff rule on zone %s has no exit_zone_ref — inert", zone_id)
                continue
            compiled.append(DriveoffRule(
                camera_id=camera_id,
                pump_zone_id=zone_id,
                exit_zone_id=rule.exit_zone_ref,
                grace_s=rule.grace_s if rule.grace_s is not None else 90.0,
                shop_camera_id=rule.shop_camera_ref,
            ))
        return cls(compiled)

    def feed(self, signals: list[Signal]) -> list[EventCandidate]:
        out: list[EventCandidate] = []
        for sig in signals:
            if sig.kind == "zone_enter" and sig.object_class == ObjectClass.person:
                self._person_entries.append((sig.ts, sig.camera_id))
            elif sig.kind == "dwell_complete":
                self._dwell_done[(sig.camera_id, sig.zone_id, sig.track_id)] = sig.ts
            elif sig.kind == "line_cross" and sig.object_class in VEHICLE_CLASSES:
                out.extend(self._on_exit_cross(sig))
        self._gc(max((s.ts for s in signals), default=0.0))
        return out

    def _on_exit_cross(self, sig: Signal) -> list[EventCandidate]:
        fired: list[EventCandidate] = []
        for rule in self.rules:
            if sig.zone_id != rule.exit_zone_id:
                continue
            key = (rule.camera_id, rule.pump_zone_id, sig.track_id)
            dwell_ts = self._dwell_done.get(key)
            if dwell_ts is None or sig.ts - dwell_ts > rule.exit_window_s:
                continue
            if self._person_entered(sig.ts, rule):
                log.info("driveoff suppressed: person entered shop within %.0fs grace",
                         rule.grace_s)
                self._dwell_done.pop(key, None)
                continue
            self._dwell_done.pop(key, None)
            fired.append(EventCandidate(
                camera_id=rule.camera_id,
                zone_id=rule.pump_zone_id,
                rule_type=RuleType.driveoff,
                object_class=sig.object_class,
                confidence=1.0,  # composite: constituent signals already conf-gated
                track_id=sig.track_id,
                ts=sig.ts,
                metadata={
                    "exit_zone_id": rule.exit_zone_id,
                    "dwell_completed_at": dwell_ts,
                    "grace_s": rule.grace_s,
                },
            ))
        return fired

    def _person_entered(self, exit_ts: float, rule: DriveoffRule) -> bool:
        for ts, camera_id in reversed(self._person_entries):
            if exit_ts - ts > rule.grace_s:
                break
            if ts > exit_ts:
                continue
            if rule.shop_camera_id is None or camera_id == rule.shop_camera_id:
                return True
        return False

    def _gc(self, now: float) -> None:
        if now <= 0:
            return
        while self._person_entries and now - self._person_entries[0][0] > self._max_lookback:
            self._person_entries.popleft()
        self._dwell_done = {k: v for k, v in self._dwell_done.items()
                            if now - v <= 600.0}
