import asyncio
import json

import httpx

from sentinel_edge.cloudlink import CloudLink, decode_envelope, make_envelope
from sentinel_edge.config import ConfigCache, EdgeSettings
from sentinel_edge.contracts import (
    CONTRACT_VERSION,
    DeviceConfig,
    DeviceHealth,
    Heartbeat,
    utcnow_rfc3339,
)

VALID_CONFIG = {
    "schema_version": "1",
    "config_etag": 'W/"etag-1"',
    "device_id": "0190f3a0-0000-7000-8000-000000000009",
    "retention_days": 14,
    "cameras": [],
}


def make_heartbeat():
    return Heartbeat(
        device_id="0190f3a0-0000-7000-8000-000000000009",
        contract_version=CONTRACT_VERSION, model_version="test",
        config_etag="", device=DeviceHealth(status="online"),
        cameras=[], sent_at=utcnow_rfc3339())


def make_link(tmp_path, handler=None):
    settings = EdgeSettings(cloud_base_url="https://cloud.test",
                            device_token="tok", device_id="dev", data_dir=tmp_path)
    applied: list[DeviceConfig] = []

    async def on_config(cfg):
        applied.append(cfg)

    http = None
    if handler is not None:
        http = httpx.AsyncClient(transport=httpx.MockTransport(handler),
                                 base_url=settings.api_base)
    link = CloudLink(settings, ConfigCache(tmp_path), on_config,
                     make_heartbeat, http_client=http)
    sent: list[dict] = []

    async def capture_send(env):
        sent.append(env.model_dump(mode="json"))
    link._send = capture_send  # type: ignore[method-assign]
    return link, applied, sent


def test_envelope_roundtrip_and_malformed():
    env = make_envelope("heartbeat", {"x": 1})
    decoded = decode_envelope(env.model_dump_json())
    assert decoded and decoded.type == "heartbeat" and decoded.payload == {"x": 1}
    assert decode_envelope("not json at all") is None
    assert decode_envelope(json.dumps({"v": 2, "type": "ping"})) is None


def test_unknown_ws_type_is_ignored(tmp_path):
    link, applied, sent = make_link(tmp_path)
    env = make_envelope("hologram.start", {})  # from a newer peer
    asyncio.run(link._dispatch(env.model_dump_json()))
    assert applied == [] and sent == []


def test_ping_gets_pong_ack(tmp_path):
    link, _, sent = make_link(tmp_path)
    ping = make_envelope("ping", {})
    asyncio.run(link._dispatch(ping.model_dump_json()))
    (pong,) = sent
    assert pong["type"] == "pong" and pong["ack"] == ping.id


def test_config_push_applies_caches_and_acks(tmp_path):
    link, applied, sent = make_link(tmp_path)
    env = make_envelope("config.push", VALID_CONFIG)
    asyncio.run(link._dispatch(env.model_dump_json()))
    assert len(applied) == 1 and applied[0].config_etag == 'W/"etag-1"'
    assert link.last_config_etag == 'W/"etag-1"'
    (ack,) = sent
    assert ack["payload"]["command"] == "config.ack" and ack["ack"] == env.id
    # cached for reboot-during-outage
    assert ConfigCache(tmp_path).load().config_etag == 'W/"etag-1"'


def test_invalid_config_push_rejected_without_disruption(tmp_path):
    link, applied, sent = make_link(tmp_path)
    bad = dict(VALID_CONFIG)
    bad.pop("retention_days")
    env = make_envelope("config.push", bad)
    asyncio.run(link._dispatch(env.model_dump_json()))
    assert applied == []                       # running ruleset untouched
    assert link.last_config_etag == ""
    (rej,) = sent
    assert rej["payload"]["command"] == "config.reject"


def test_duplicate_etag_not_reapplied(tmp_path):
    link, applied, _ = make_link(tmp_path)
    env = make_envelope("config.push", VALID_CONFIG)
    asyncio.run(link._dispatch(env.model_dump_json()))
    asyncio.run(link._dispatch(make_envelope("config.push", VALID_CONFIG)
                               .model_dump_json()))
    assert len(applied) == 1


def test_http_config_sync_with_etag(tmp_path):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if request.headers.get("If-None-Match") == 'W/"etag-1"':
            return httpx.Response(304)
        return httpx.Response(200, json=VALID_CONFIG)

    link, applied, _ = make_link(tmp_path, handler=handler)
    asyncio.run(link.sync_config())
    assert len(applied) == 1
    asyncio.run(link.sync_config())            # 304 — nothing re-applied
    assert len(applied) == 1 and calls["n"] == 2


def test_heartbeat_http_fallback_when_ws_down(tmp_path):
    posted = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/edge/heartbeat"):
            posted.append(json.loads(request.content))
            return httpx.Response(204)
        return httpx.Response(404)

    link, _, _ = make_link(tmp_path, handler=handler)
    asyncio.run(link.send_heartbeat())         # _ws is None -> HTTP fallback
    (hb,) = posted
    assert hb["device"]["status"] == "online" and hb["schema_version"] == "1"
