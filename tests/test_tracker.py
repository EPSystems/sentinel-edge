from sentinel_edge.contracts import ObjectClass
from sentinel_edge.track import ByteTracker, Detection, iou_matrix


def det(x, conf=0.9, cls=ObjectClass.person, size=20):
    return Detection(object_class=cls, confidence=conf,
                     bbox=(x, 0.0, x + size, float(size * 2)))


def test_iou_matrix_values():
    m = iou_matrix([(0, 0, 10, 10)], [(0, 0, 10, 10), (100, 100, 110, 110)])
    assert m[0, 0] == 1.0 and m[0, 1] == 0.0


def test_persistent_id_across_frames():
    t = ByteTracker()
    (a,) = t.update([det(0)], ts=0.0)
    (b,) = t.update([det(3)], ts=0.1)
    (c,) = t.update([det(6)], ts=0.2)
    assert a.track_id == b.track_id == c.track_id
    assert c.age_frames == 3
    assert len(c.history) == 3


def test_id_survives_brief_occlusion():
    t = ByteTracker(track_ttl_s=2.0)
    (a,) = t.update([det(0)], ts=0.0)
    assert t.update([], ts=0.5) == []          # occluded for 1s
    (b,) = t.update([det(4)], ts=1.0)
    assert b.track_id == a.track_id


def test_track_expires_after_ttl():
    t = ByteTracker(track_ttl_s=2.0)
    (a,) = t.update([det(0)], ts=0.0)
    t.update([], ts=1.0)
    (b,) = t.update([det(0)], ts=5.0)          # 5s later: new identity
    assert b.track_id != a.track_id


def test_low_score_detection_rescues_track_but_spawns_nothing():
    t = ByteTracker(high_thresh=0.5, low_thresh=0.1)
    (a,) = t.update([det(0, conf=0.9)], ts=0.0)
    (b,) = t.update([det(3, conf=0.3)], ts=0.1)  # low-score keeps the id
    assert b.track_id == a.track_id
    fresh = ByteTracker()
    assert fresh.update([det(0, conf=0.3)], ts=0.0) == []  # low alone: no track


def test_two_objects_keep_separate_ids():
    t = ByteTracker()
    tracks = t.update([det(0), det(300)], ts=0.0)
    ids0 = sorted(tr.track_id for tr in tracks)
    tracks = t.update([det(4), det(296)], ts=0.1)
    by_x = sorted(tracks, key=lambda tr: tr.bbox[0])
    assert [by_x[0].track_id, by_x[1].track_id] == ids0


def test_history_is_bounded():
    t = ByteTracker(history_len=10)
    for i in range(50):
        (tr,) = t.update([det(float(i))], ts=i * 0.1)
    assert len(tr.history) == 10


def test_anchor_points():
    t = ByteTracker()
    (tr,) = t.update([det(10)], ts=0.0)  # bbox (10,0,30,40)
    assert tr.anchor("bottom") == (20.0, 40.0)
    assert tr.anchor("center") == (20.0, 20.0)
