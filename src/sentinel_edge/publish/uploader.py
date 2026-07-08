"""Event publisher: presign -> PUT clip/thumb to R2 -> POST metadata
(Flow A steps 4-5), draining the durable queue oldest-first.

Idempotency: event_id (UUIDv7) is both the payload id and the
Idempotency-Key header; a replay after an ambiguous failure returns the
existing event (201) with no duplicate. Retry classification:
- network / 5xx / 429  -> retryable (stays queued, backoff)
- 409 (clip not in R2) -> clip state reset, re-upload
- 422 / other 4xx      -> terminal (poison item, never blocks the queue)
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Optional

import httpx

from ..config import EdgeSettings
from ..contracts import EventPayload, PresignRequest, PresignResponse
from ..record import SpoolManager
from ..streams import Backoff
from .queue_sqlite import CLIP_UPLOADED, PENDING, EventQueue

log = logging.getLogger(__name__)


class EventPublisher:
    def __init__(self, settings: EdgeSettings, queue: EventQueue,
                 spool: Optional[SpoolManager] = None,
                 client: Optional[httpx.AsyncClient] = None,
                 poll_interval_s: float = 1.0):
        self.settings = settings
        self.queue = queue
        self.spool = spool
        self.poll_interval_s = poll_interval_s
        self._client = client or httpx.AsyncClient(
            base_url=settings.api_base,
            headers={"Authorization": f"Bearer {settings.device_token}"},
            timeout=httpx.Timeout(30.0, connect=10.0),
        )
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        """Drain loop; safe to run forever. Outages back off, recovery drains
        the backlog oldest-first."""
        backoff = Backoff(base=2.0, cap=60.0)
        while not self._stop.is_set():
            try:
                worked = await self.drain_once()
                if worked:
                    backoff.reset()
                    continue
                delay = self.poll_interval_s
            except Exception:
                log.exception("publisher drain error")
                delay = backoff.next_delay()
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass
        await self._client.aclose()

    async def drain_once(self) -> bool:
        """Process one batch; returns True if anything was processed."""
        batch = self.queue.next_batch(limit=5)
        for item in batch:
            await self._process(item)
        return bool(batch)

    # ------------------------------------------------------------ single item

    async def _process(self, item: dict[str, Any]) -> None:
        event_id = item["event_id"]
        try:
            if item["state"] == PENDING:
                clip_key, thumb_key = await self._upload_files(item)
                self.queue.mark_clip_uploaded(event_id, clip_key, thumb_key)
                item.update(state=CLIP_UPLOADED, clip_key=clip_key, thumb_key=thumb_key)
            if item["state"] == CLIP_UPLOADED:
                await self._post_metadata(item)
                self.queue.mark_done(event_id)
                self._cleanup_files(item)
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status == 409:
                # metadata POSTed before the clip landed — reset to re-upload
                log.warning("event %s: 409 clip missing in R2 — re-uploading", event_id)
                self.queue._update(event_id, state=PENDING)  # noqa: SLF001 (same package)
            elif status in (401, 429) or status >= 500:
                self.queue.mark_failed(event_id, f"HTTP {status}", terminal=False)
                raise  # let run() back off — auth/rate/server issues are global
            else:
                log.error("event %s rejected terminally: HTTP %s %s",
                          event_id, status, e.response.text[:200])
                self.queue.mark_failed(event_id, f"HTTP {status}", terminal=True)
        except (httpx.TransportError, asyncio.TimeoutError) as e:
            self.queue.mark_failed(event_id, str(e), terminal=False)
            raise
        except FileNotFoundError as e:
            log.error("event %s clip file lost (%s) — terminal", event_id, e)
            self.queue.mark_failed(event_id, str(e), terminal=True)

    async def _upload_files(self, item: dict[str, Any]) -> tuple[str, Optional[str]]:
        clip_key = await self._presign_and_put(
            item, kind="clip", path=item["clip_path"], content_type="video/mp4")
        thumb_key: Optional[str] = None
        if item.get("thumb_path"):
            try:
                thumb_key = await self._presign_and_put(
                    item, kind="thumb", path=item["thumb_path"], content_type="image/jpeg")
            except Exception:
                log.exception("thumb upload failed — event ships without one")
        return clip_key, thumb_key

    async def _presign_and_put(self, item: dict[str, Any], kind: str,
                               path: str, content_type: str) -> str:
        data = Path(path).read_bytes()
        req = PresignRequest(
            event_id=item["event_id"],
            camera_id=json_field(item, "camera_id"),
            kind=kind,  # type: ignore[arg-type]
            content_type=content_type,
            content_length=len(data),
        )
        resp = await self._client.post("/edge/clips/presign",
                                       json=req.model_dump())
        resp.raise_for_status()
        presign = PresignResponse.model_validate(resp.json())
        put = await self._client.put(presign.url, content=data,
                                     headers=presign.headers)
        put.raise_for_status()
        return presign.object_key

    async def _post_metadata(self, item: dict[str, Any]) -> None:
        payload = dict_payload(item)
        payload["clip_object_key"] = item["clip_key"]
        payload["thumb_object_key"] = item.get("thumb_key")
        event = EventPayload.model_validate(payload)  # contract-validate before wire
        resp = await self._client.post(
            "/edge/events",
            json=event.model_dump(mode="json"),
            headers={"Idempotency-Key": item["event_id"]},
        )
        resp.raise_for_status()

    def _cleanup_files(self, item: dict[str, Any]) -> None:
        for key in ("clip_path", "thumb_path"):
            p = item.get(key)
            if p and self.spool:
                self.spool.mark_uploaded(Path(p))
            elif p:
                Path(p).unlink(missing_ok=True)


def dict_payload(item: dict[str, Any]) -> dict[str, Any]:
    import json
    return json.loads(item["payload_json"])


def json_field(item: dict[str, Any], field: str) -> Any:
    return dict_payload(item)[field]
