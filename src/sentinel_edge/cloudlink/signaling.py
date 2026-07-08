"""Live-view WebRTC signaling relay (Flow C).

The edge only brokers SDP/ICE between the browser (via the cloud WS) and the
local go2rtc sidecar; media flows peer-to-peer or via the cloud TURN — the
edge opens no inbound ports. go2rtc's WebRTC API: POST /api/webrtc?src=<cam>
with the offer SDP, response body is the answer SDP.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

from ..contracts import WsEnvelope

log = logging.getLogger(__name__)


class LiveViewSignaling:
    def __init__(self, go2rtc_base_url: str = "http://127.0.0.1:1984",
                 client: Optional[httpx.AsyncClient] = None):
        self._client = client or httpx.AsyncClient(base_url=go2rtc_base_url,
                                                   timeout=10.0)

    async def handle(self, env: WsEnvelope) -> Optional[dict[str, Any]]:
        """Returns a reply envelope spec ({type, payload}) or None."""
        if env.type == "liveview.offer":
            camera_id = env.payload.get("camera_id")
            offer_sdp = env.payload.get("sdp")
            if not camera_id or not offer_sdp:
                log.warning("liveview.offer missing camera_id/sdp")
                return None
            resp = await self._client.post(
                "/api/webrtc", params={"src": camera_id},
                content=offer_sdp, headers={"Content-Type": "application/sdp"},
            )
            resp.raise_for_status()
            return {"type": "liveview.answer",
                    "payload": {"camera_id": camera_id, "sdp": resp.text}}
        if env.type == "liveview.ice":
            # go2rtc's whep endpoint handles trickle ICE via PATCH; MVP relies
            # on non-trickle SDP (candidates inline), so ICE frames are ack'd
            # and dropped here. Revisit with the cloud TURN work (Phase 4).
            return None
        return None

    async def close(self) -> None:
        await self._client.aclose()
