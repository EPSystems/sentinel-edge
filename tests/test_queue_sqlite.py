import time

from sentinel_edge.publish import CLIP_UPLOADED, DONE, FAILED_TERMINAL, PENDING, EventQueue


def make_queue(tmp_path):
    return EventQueue(tmp_path / "q.sqlite3")


def payload(camera="cam-1"):
    return {"camera_id": camera, "rule_type": "zone_enter"}


def test_enqueue_and_state_machine(tmp_path):
    q = make_queue(tmp_path)
    q.enqueue("ev-1", payload(), "/spool/ev-1.mp4", "/spool/ev-1.jpg")
    (item,) = q.next_batch()
    assert item["state"] == PENDING
    q.mark_clip_uploaded("ev-1", "clips/t/ev-1/clip.mp4", "clips/t/ev-1/thumb.jpg")
    (item,) = q.next_batch()
    assert item["state"] == CLIP_UPLOADED and item["clip_key"].endswith("clip.mp4")
    q.mark_done("ev-1")
    assert q.next_batch() == [] and q.depth() == 0


def test_enqueue_is_idempotent_on_event_id(tmp_path):
    q = make_queue(tmp_path)
    q.enqueue("ev-1", payload(), "/a.mp4", None)
    q.enqueue("ev-1", payload(), "/a.mp4", None)
    assert q.depth() == 1


def test_oldest_first_drain(tmp_path):
    q = make_queue(tmp_path)
    for i in range(5):
        q.enqueue(f"ev-{i}", payload(), f"/{i}.mp4", None)
        time.sleep(0.01)
    ids = [it["event_id"] for it in q.next_batch(limit=5)]
    assert ids == [f"ev-{i}" for i in range(5)]


def test_terminal_failure_leaves_the_queue(tmp_path):
    q = make_queue(tmp_path)
    q.enqueue("ev-1", payload(), "/a.mp4", None)
    q.mark_failed("ev-1", "HTTP 422", terminal=True)
    assert q.depth() == 0
    q2 = make_queue(tmp_path)  # visible after reopen too


def test_retryable_failure_stays_queued_and_counts_attempts(tmp_path):
    q = make_queue(tmp_path)
    q.enqueue("ev-1", payload(), "/a.mp4", None)
    q.mark_failed("ev-1", "timeout", terminal=False)
    q.mark_failed("ev-1", "timeout", terminal=False)
    (item,) = q.next_batch()
    assert item["attempts"] == 2 and item["state"] == PENDING


def test_drop_thumbnails_backpressure(tmp_path):
    q = make_queue(tmp_path)
    q.enqueue("ev-1", payload(), "/a.mp4", "/a.jpg")
    q.enqueue("ev-2", payload(), "/b.mp4", "/b.jpg")
    dropped = q.drop_thumbnails()
    assert sorted(dropped) == ["/a.jpg", "/b.jpg"]
    for item in q.next_batch(limit=10):
        assert item["thumb_path"] is None


def test_depth_and_oldest_age(tmp_path):
    q = make_queue(tmp_path)
    assert q.depth() == 0 and q.oldest_age_s() is None
    q.enqueue("ev-1", payload(), "/a.mp4", None)
    assert q.depth() == 1
    assert q.oldest_age_s() is not None and q.oldest_age_s() >= 0


def test_survives_reopen(tmp_path):
    q = make_queue(tmp_path)
    q.enqueue("ev-1", payload(), "/a.mp4", None)
    q.close()
    q2 = make_queue(tmp_path)
    assert q2.depth() == 1  # durable journal — outage loses nothing
