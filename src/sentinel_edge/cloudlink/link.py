"""The single persistent outbound WSS to the cloud (buildspec §4.8):
config down, heartbeat up, live-view signaling both ways. The edge opens the
connection; the cloud never connects in.

Envelope handling follows the contract's forward-compat rules: the envelope
is validated first; an unknown `type` from a newer peer is logged + ignored.
Heartbeats fall back to POST /v1/edge/heartbeat while the WS is down, so the
90s device-red SLO holds through WS trouble.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable, Optional

import httpx

from ..config import ConfigCache, EdgeSettings
from ..contracts import (
    KNOWN_WS_TYPES,
    DeviceConfig,
    Heartbeat,
    WsEnvelope,
    utcnow_rfc3339,
    uuid7,
)
from ..streams import Backoff

log = logging.getLogger(__name__)

ConfigHandler = Callable[[DeviceConfig], Awaitable[None]]
SignalingHandler = Callable[[WsEnvelope], Awaitable[Optional[dict[str, Any]]]]


def make_envelope(type_: str, payload: dict[str, Any],
                  ack: Optional[str] = None) -> WsEnvelope:
    return WsEnvelope(v=1, type=type_, id=uuid7(), ts=utcnow_rfc3339(),
                      ack=ack, payload=payload)


def decode_envelope(raw: str | bytes) -> Optional[WsEnvelope]:
    """Strict envelope parse; unknown *type* is tolerated by the dispatcher,
    a malformed envelope is dropped with a log line."""
    try:
        return WsEnvelope.model_validate_json(raw)
    except Exception:
        log.warning("dropping malformed WS envelope: %.200s", raw)
        return None


class CloudLink:
    def __init__(
        self,
        settings: EdgeSettings,
        config_cache: ConfigCache,
        on_config: ConfigHandler,
        heartbeat_builder: Callable[[], Heartbeat],
        on_signaling: Optional[SignalingHandler] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        self.settings = settings
        self.config_cache = config_cache
        self.on_config = on_config
        self.heartbeat_builder = heartbeat_builder
        self.on_signaling = on_signaling
        self._http = http_client or httpx.AsyncClient(
            base_url=settings.api_base,
            headers={"Authorization": f"Bearer {settings.device_token}"},
            timeout=httpx.Timeout(20.0, connect=10.0),
        )
        self._stop = asyncio.Event()
        self._ws = None
        self.last_config_etag: str = ""
        self.ws_connected: bool = False

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------ lifecycle

    async def run(self) -> None:
        """Boot order: cached config first (a reboot during a cloud outage
        still runs the last good ruleset), then sync, then hold the WS open."""
        cached = self.config_cache.load()
        if cached is not None:
            log.info("applying cached config etag=%s", cached.config_etag)
            self.last_config_etag = cached.config_etag
            await self.on_config(cached)

        heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        try:
            await self._ws_loop()
        finally:
            heartbeat_task.cancel()
            await asyncio.gather(heartbeat_task, return_exceptions=True)
            await self._http.aclose()

    async def _ws_loop(self) -> None:
        import websockets

        backoff = Backoff(base=1.0, cap=30.0)
        while not self._stop.is_set():
            try:
                async with websockets.connect(
                    self.settings.ws_url,
                    additional_headers={
                        "Authorization": f"Bearer {self.settings.device_token}"},
                    ping_interval=20, ping_timeout=20,
                ) as ws:
                    self._ws = ws
                    self.ws_connected = True
                    backoff.reset()
                    log.info("cloudlink connected")
                    await self.sync_config()  # re-sync on every (re)connect
                    async for raw in ws:
                        await self._dispatch(raw)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("cloudlink dropped: %s", e)
            finally:
                self._ws = None
                self.ws_connected = False
            if self._stop.is_set():
                return
            delay = backoff.next_delay()
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass

    # ------------------------------------------------------------ inbound

    async def _dispatch(self, raw: str | bytes) -> None:
        env = decode_envelope(raw)
        if env is None:
            return
        if env.type not in KNOWN_WS_TYPES:
            log.info("ignoring unknown WS type %r (newer peer)", env.type)
            return
        if env.type == "ping":
            await self._send(make_envelope("pong", {}, ack=env.id))
        elif env.type == "config.push":
            await self._handle_config_push(env)
        elif env.type == "command":
            await self._handle_command(env)
        elif env.type.startswith("liveview."):
            if self.on_signaling is not None:
                reply = await self.on_signaling(env)
                if reply is not None:
                    await self._send(make_envelope(reply["type"], reply["payload"],
                                                   ack=env.id))
        # heartbeat/model.update inbound are cloud-side concerns; ignored here

    async def _handle_config_push(self, env: WsEnvelope) -> None:
        try:
            config = DeviceConfig.model_validate(env.payload)
        except Exception as e:
            # invalid config must NOT disrupt the running ruleset; report it
            log.error("rejected invalid config.push: %s", e)
            await self._send(make_envelope("command", {
                "command": "config.reject",
                "config_etag": env.payload.get("config_etag"),
                "error": str(e)[:500],
            }, ack=env.id))
            return
        await self._apply_config(config)
        await self._send(make_envelope("command", {
            "command": "config.ack", "config_etag": config.config_etag,
        }, ack=env.id))

    async def _handle_command(self, env: WsEnvelope) -> None:
        cmd = env.payload.get("command")
        if cmd == "request-heartbeat":
            await self.send_heartbeat()
        else:
            log.info("unhandled command %r", cmd)

    # ------------------------------------------------------------ config sync

    async def sync_config(self) -> None:
        """Pull-fallback with ETag; 304 = already current."""
        headers = {}
        if self.last_config_etag:
            headers["If-None-Match"] = self.last_config_etag
        resp = await self._http.get("/edge/config", headers=headers)
        if resp.status_code == 304:
            return
        resp.raise_for_status()
        config = DeviceConfig.model_validate(resp.json())
        await self._apply_config(config)

    async def _apply_config(self, config: DeviceConfig) -> None:
        if config.config_etag == self.last_config_etag:
            return
        self.config_cache.store(config)
        self.last_config_etag = config.config_etag
        await self.on_config(config)
        log.info("config applied etag=%s cameras=%d",
                 config.config_etag, len(config.cameras))

    # ------------------------------------------------------------ heartbeat

    async def _heartbeat_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self.send_heartbeat()
            except Exception as e:
                log.warning("heartbeat failed: %s", e)
            try:
                await asyncio.wait_for(self._stop.wait(),
                                       timeout=self.settings.heartbeat_interval_s)
            except asyncio.TimeoutError:
                pass

    async def send_heartbeat(self) -> None:
        hb = self.heartbeat_builder()
        if self._ws is not None:
            await self._send(make_envelope("heartbeat", hb.model_dump(mode="json")))
        else:  # HTTP fallback keeps the 90s SLO through WS trouble
            resp = await self._http.post("/edge/heartbeat",
                                         json=hb.model_dump(mode="json"))
            resp.raise_for_status()

    async def _send(self, env: WsEnvelope) -> None:
        if self._ws is None:
            raise ConnectionError("WS not connected")
        await self._ws.send(json.dumps(env.model_dump(mode="json")))
