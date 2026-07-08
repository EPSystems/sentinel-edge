"""Publisher happy path + failure classification against a mock cloud."""
import asyncio
import json

import httpx
import pytest

from sentinel_edge.config import EdgeSettings
from sentinel_edge.publish import DONE, FAILED_TERMINAL, PENDING, EventPublisher, EventQueue


def make_payload(event_id):
    return {
        "schema_version": "1",
        "event_id": event_id,
        "camera_id": "0190f3a1-1111-7000-8000-000000000001",
        "zone_id": None,
        "object_class": "person",
        "rule_type": "zone_enter",
        "confidence": 0.91,
        "track_id": 1,
        "plate_number": None,
        "plate_confidence": None,
        "speed_kmh": None,
        "metadata": {},
        "occurred_at": "2026-07-07T10:00:00.000Z",
    }


class MockCloud:
    def __init__(self):
        self.puts: list[str] = []
        self.events: list[dict] = []
        self.fail_events_with: int | None = None
        self.fail_once_409 = False

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/edge/clips/presign"):
            body = json.loads(request.content)
            key = f"clips/tenant/{body['event_id']}/{body['kind']}"
            return httpx.Response(200, json={
                "url": f"https://r2.test/{key}", "method": "PUT",
                "object_key": key, "expires_in": 300,
                "headers": {"Content-Type": body["content_type"]}})
        if request.url.host == "r2.test":
            self.puts.append(request.url.path)
            return httpx.Response(200)
        if path.endswith("/edge/events"):
            if self.fail_events_with:
                return httpx.Response(self.fail_events_with, json={"title": "nope"})
            if self.fail_once_409:
                self.fail_once_409 = False
                return httpx.Response(409)
            body = json.loads(request.content)
            assert request.headers["Idempotency-Key"] == body["event_id"]
            self.events.append(body)
            return httpx.Response(201, json={"event_id": body["event_id"]})
        return httpx.Response(404)


def make_publisher(tmp_path, cloud: MockCloud):
    settings = EdgeSettings(cloud_base_url="https://cloud.test",
                            device_token="tok", device_id="dev-1",
                            data_dir=tmp_path)
    queue = EventQueue(tmp_path / "q.sqlite3")
    client = httpx.AsyncClient(transport=httpx.MockTransport(cloud.handler),
                               base_url=settings.api_base,
                               headers={"Authorization": "Bearer tok"})
    return EventPublisher(settings, queue, client=client), queue


def enqueue_event(queue, tmp_path, event_id="0190f3c2-7a1e-7c3b-9f00-2b1a4c9d8e77",
                  with_thumb=True):
    clip = tmp_path / f"{event_id}.mp4"
    clip.write_bytes(b"clipbytes")
    thumb = None
    if with_thumb:
        thumb = tmp_path / f"{event_id}.jpg"
        thumb.write_bytes(b"thumbbytes")
    queue.enqueue(event_id, make_payload(event_id), str(clip),
                  str(thumb) if thumb else None)
    return event_id, clip, thumb


def test_happy_path_uploads_and_posts(tmp_path):
    cloud = MockCloud()
    pub, queue = make_publisher(tmp_path, cloud)
    event_id, clip, thumb = enqueue_event(queue, tmp_path)
    asyncio.run(pub.drain_once())
    assert queue.depth() == 0
    assert len(cloud.puts) == 2                      # clip + thumb
    (event,) = cloud.events
    assert event["clip_object_key"] == f"clips/tenant/{event_id}/clip"
    assert event["thumb_object_key"] == f"clips/tenant/{event_id}/thumb"
    assert not clip.exists() and not thumb.exists()  # spooled files cleaned


def test_409_resets_to_reupload_then_succeeds(tmp_path):
    cloud = MockCloud()
    cloud.fail_once_409 = True
    pub, queue = make_publisher(tmp_path, cloud)
    event_id, *_ = enqueue_event(queue, tmp_path, with_thumb=False)
    asyncio.run(pub.drain_once())                    # upload ok, POST 409 -> reset
    (item,) = queue.next_batch()
    assert item["state"] == PENDING
    asyncio.run(pub.drain_once())                    # re-upload + POST 201
    assert queue.depth() == 0 and len(cloud.events) == 1


def test_422_is_terminal(tmp_path):
    cloud = MockCloud()
    cloud.fail_events_with = 422
    pub, queue = make_publisher(tmp_path, cloud)
    enqueue_event(queue, tmp_path, with_thumb=False)
    asyncio.run(pub.drain_once())
    assert queue.depth() == 0                        # poison item removed from drain
    assert cloud.events == []


def test_500_is_retryable_and_backs_off(tmp_path):
    cloud = MockCloud()
    cloud.fail_events_with = 500
    pub, queue = make_publisher(tmp_path, cloud)
    enqueue_event(queue, tmp_path, with_thumb=False)
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(pub.drain_once())
    (item,) = queue.next_batch()
    assert item["attempts"] == 1                     # still queued for replay
    cloud.fail_events_with = None
    asyncio.run(pub.drain_once())
    assert queue.depth() == 0


def test_network_error_keeps_event_queued(tmp_path):
    cloud = MockCloud()
    pub, queue = make_publisher(tmp_path, cloud)

    def offline(request):
        raise httpx.ConnectError("no route to cloud")
    pub._client._transport = httpx.MockTransport(offline)
    enqueue_event(queue, tmp_path, with_thumb=False)
    with pytest.raises(httpx.TransportError):
        asyncio.run(pub.drain_once())
    assert queue.depth() == 1                        # zero metadata loss offline
