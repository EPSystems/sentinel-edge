from collections import deque

from sentinel_edge.config import PipelineDefaults
from sentinel_edge.contracts import CameraConfig, ObjectClass, RuleType
from sentinel_edge.rules import RuleEngine
from sentinel_edge.track import Track

FRAME = (1000, 1000)


def make_camera(zones):
    return CameraConfig(
        camera_id="cam-1", name="test", rtsp_ref="secret://x",
        detect_fps=8, lpr_enabled=False, calibration={"px_per_m": 10.0},
        zones=zones)


def make_track(tid, x, y, cls="person", conf=0.9, age=10):
    # 20x40 box whose bottom-center anchor is (x, y)
    return Track(track_id=tid, object_class=ObjectClass(cls), confidence=conf,
                 bbox=(x - 10, y - 40, x + 10, y), age_frames=age,
                 history=deque(maxlen=90))


ZONE = {
    "zone_id": "z-1", "name": "restricted",
    "polygon": [[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]],  # px 400..600
    "rules": [{"rule_type": "zone_enter", "classes": ["person"], "min_confidence": 0.7}],
}
LINE = {
    "zone_id": "z-line", "name": "tripwire",
    "polygon": [[0.5, 0.1], [0.5, 0.9]],  # vertical x=500
    "rules": [{"rule_type": "line_cross", "direction": "both"}],
}


def engine_for(zones, **overrides):
    defaults = PipelineDefaults(**overrides)
    return RuleEngine(make_camera(zones), defaults, FRAME)


def test_zone_enter_fires_once_and_respects_cooldown():
    eng = engine_for([ZONE])
    cands, _ = eng.update([make_track(1, 300, 500)], ts=100.0)   # outside
    assert cands == []
    cands, _ = eng.update([make_track(1, 500, 500)], ts=101.0)   # entered
    assert len(cands) == 1 and cands[0].rule_type == RuleType.zone_enter
    cands, _ = eng.update([make_track(1, 510, 500)], ts=102.0)   # still inside
    assert cands == []
    # leave and re-enter within cooldown -> suppressed
    eng.update([make_track(1, 300, 500)], ts=103.0)
    cands, _ = eng.update([make_track(1, 500, 500)], ts=104.0)
    assert cands == []
    # re-enter after cooldown -> fires again
    eng.update([make_track(1, 300, 500)], ts=200.0)
    cands, _ = eng.update([make_track(1, 500, 500)], ts=201.0)
    assert len(cands) == 1


def test_zone_enter_filters_class_and_confidence():
    eng = engine_for([ZONE])
    cands, _ = eng.update([make_track(1, 500, 500, cls="car")], ts=1.0)
    assert cands == []
    cands, _ = eng.update([make_track(2, 500, 500, conf=0.5)], ts=2.0)
    assert cands == []


def test_young_track_cannot_fire():
    eng = engine_for([ZONE])
    cands, _ = eng.update([make_track(1, 500, 500, age=1)], ts=1.0)
    assert cands == []


def test_loiter_dwell_fires_once_per_presence():
    zone = dict(ZONE)
    zone["rules"] = [{"rule_type": "loiter_dwell", "dwell_s": 5,
                      "classes": ["person"], "cooldown_s": 1}]
    eng = engine_for([zone])
    eng.update([make_track(1, 500, 500)], ts=0.0)
    cands, _ = eng.update([make_track(1, 505, 500)], ts=3.0)
    assert cands == []                                            # not yet
    cands, _ = eng.update([make_track(1, 510, 500)], ts=6.0)
    assert len(cands) == 1
    assert cands[0].metadata["dwell_s"] == 6.0
    cands, _ = eng.update([make_track(1, 515, 500)], ts=20.0)     # same presence
    assert cands == []


def test_line_cross_direction_filter():
    line = dict(LINE)
    line["rules"] = [{"rule_type": "line_cross", "direction": "rl",
                      "min_confidence": 0.5}]
    eng = engine_for([line])
    eng.update([make_track(1, 400, 500)], ts=1.0)
    cands, sigs = eng.update([make_track(1, 600, 500)], ts=2.0)   # l->r = "rl"
    assert len(cands) == 1 and cands[0].metadata["direction"] == "rl"
    # opposite direction filtered out, but the internal signal still emits
    eng2 = engine_for([line])
    eng2.update([make_track(2, 600, 500)], ts=1.0)
    cands, sigs = eng2.update([make_track(2, 400, 500)], ts=2.0)
    assert cands == []
    assert any(s.kind == "line_cross" and s.direction == "lr" for s in sigs)


def test_line_counts_track_both_directions_regardless_of_cooldown():
    line = dict(LINE)
    line["rules"] = [{"rule_type": "line_cross", "direction": "both",
                      "cooldown_s": 1000}]  # long cooldown must not suppress the count
    eng = engine_for([line])
    eng.update([make_track(1, 400, 500)], ts=1.0)
    eng.update([make_track(1, 600, 500)], ts=2.0)   # rl
    eng.update([make_track(1, 400, 500)], ts=3.0)   # lr, but cooldown blocks the alert
    counts = eng.line_counts()
    assert counts["z-line"]["name"] == "tripwire"
    assert counts["z-line"]["rl"] == 1
    assert counts["z-line"]["lr"] == 1
    assert counts["z-line"]["total"] == 2


def test_speed_rule_fires_over_limit():
    zone = {
        "zone_id": "z-s", "name": "speedzone",
        "polygon": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        "rules": [{"rule_type": "speed", "speed_limit_kmh": 30,
                   "min_confidence": 0.5, "classes": ["car"]}],
    }
    eng = engine_for([zone])
    # px_per_m=10 -> 100 px/s = 10 m/s = 36 km/h
    x = 100.0
    fired = []
    for i in range(12):
        ts = i * 0.1
        cands, _ = eng.update([make_track(1, x, 500, cls="car")], ts=ts)
        fired.extend(cands)
        x += 10.0
    assert any(c.rule_type == RuleType.speed for c in fired)
    speed_cand = next(c for c in fired if c.rule_type == RuleType.speed)
    assert speed_cand.speed_kmh and speed_cand.speed_kmh > 30
    assert "±20%" in speed_cand.metadata["accuracy_note"]


def test_driveoff_dwell_emits_signal_not_event():
    zone = dict(ZONE)
    zone["rules"] = [{"rule_type": "driveoff", "classes": ["car"], "dwell_s": 5,
                      "exit_zone_ref": "z-exit", "min_confidence": 0.5}]
    eng = engine_for([zone])
    eng.update([make_track(1, 500, 500, cls="car")], ts=0.0)
    cands, sigs = eng.update([make_track(1, 505, 500, cls="car")], ts=6.0)
    assert cands == []
    assert any(s.kind == "dwell_complete" for s in sigs)


def test_hot_swap_resets_presence_keeps_cooldown():
    eng = engine_for([ZONE])
    eng.update([make_track(1, 300, 500)], ts=1.0)
    cands, _ = eng.update([make_track(1, 500, 500)], ts=2.0)
    assert len(cands) == 1
    eng.swap_config(make_camera([ZONE]))  # same geometry, fresh presence
    cands, _ = eng.update([make_track(1, 500, 500)], ts=3.0)
    assert cands == []  # cooldown survived the swap — no alert storm


def test_invalid_zone_skipped_engine_still_works():
    bad = {"zone_id": "z-bad", "name": "degenerate",
           "polygon": [[0.1, 0.1], [0.1, 0.1]], "rules": [{"rule_type": "zone_enter"}]}
    eng = engine_for([bad, ZONE])
    eng.update([make_track(1, 300, 500)], ts=1.0)
    cands, _ = eng.update([make_track(1, 500, 500)], ts=2.0)
    assert len(cands) == 1  # good zone unaffected by the bad one
