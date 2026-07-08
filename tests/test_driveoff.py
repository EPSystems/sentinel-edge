from sentinel_edge.contracts import ObjectClass, Rule, RuleType
from sentinel_edge.rules import DriveoffCorrelator
from sentinel_edge.rules.engine import Signal

CAM = "cam-pump"
PUMP = "z-pump"
EXIT = "z-exit"
SHOP_CAM = "cam-shop"


def correlator(shop_camera=SHOP_CAM):
    rule = Rule(rule_type=RuleType.driveoff, exit_zone_ref=EXIT,
                grace_s=90, shop_camera_ref=shop_camera)
    return DriveoffCorrelator.from_config([(CAM, PUMP, rule)])


def dwell(ts, tid=7):
    return Signal("dwell_complete", CAM, PUMP, tid, ObjectClass.car, ts)


def exit_cross(ts, tid=7):
    return Signal("line_cross", CAM, EXIT, tid, ObjectClass.car, ts, direction="lr")


def person_enter(ts, camera=SHOP_CAM):
    return Signal("zone_enter", camera, "z-door", 99, ObjectClass.person, ts)


def test_driveoff_fires_when_no_person_entered_shop():
    c = correlator()
    assert c.feed([dwell(100.0)]) == []
    events = c.feed([exit_cross(150.0)])
    assert len(events) == 1
    ev = events[0]
    assert ev.rule_type == RuleType.driveoff
    assert ev.camera_id == CAM and ev.zone_id == PUMP
    assert ev.metadata["dwell_completed_at"] == 100.0


def test_driveoff_suppressed_by_person_entry_within_grace():
    c = correlator()
    c.feed([dwell(100.0)])
    c.feed([person_enter(120.0)])          # customer walked into the shop
    assert c.feed([exit_cross(150.0)]) == []


def test_person_entry_outside_grace_does_not_suppress():
    c = correlator()
    c.feed([dwell(100.0)])
    c.feed([person_enter(10.0)])           # 140s before exit > 90s grace
    assert len(c.feed([exit_cross(150.0)])) == 1


def test_person_entry_on_other_camera_does_not_suppress():
    c = correlator()
    c.feed([dwell(100.0)])
    c.feed([person_enter(120.0, camera="cam-unrelated")])
    assert len(c.feed([exit_cross(150.0)])) == 1


def test_exit_without_dwell_is_not_a_driveoff():
    c = correlator()
    assert c.feed([exit_cross(150.0)]) == []


def test_exit_too_long_after_dwell_expires():
    c = correlator()
    c.feed([dwell(100.0)])
    assert c.feed([exit_cross(100.0 + 400.0)]) == []  # > exit_window_s


def test_driveoff_fires_once_per_track():
    c = correlator()
    c.feed([dwell(100.0)])
    assert len(c.feed([exit_cross(150.0)])) == 1
    assert c.feed([exit_cross(151.0)]) == []


def test_rule_without_exit_ref_is_inert():
    rule = Rule(rule_type=RuleType.driveoff)  # missing exit_zone_ref
    c = DriveoffCorrelator.from_config([(CAM, PUMP, rule)])
    assert c.rules == []
