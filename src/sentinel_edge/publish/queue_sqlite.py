"""Durable offline event journal (buildspec §4.7).

Every event is enqueued FIRST, then a worker drains it — so a WAN outage
during active events loses zero metadata. State machine per item:

    pending -> clip_uploaded -> meta_posted(=done)      (+ failed_terminal)

Presigned URLs are requested at upload time and never persisted (they expire
in 5 min). object keys returned by presign ARE persisted so a crash between
clip upload and metadata POST does not re-upload.

Backpressure (depth thresholds from PipelineDefaults):
- level 1: queued thumbnails are dropped (thumb_object_key is nullable),
- level 2: callers should record micro-clips (single frame) so the fact of
  the event survives at minimal disk cost. True metadata-only ingest would
  need a v1 contract change (the cloud HEAD-checks the clip) — coordination
  point recorded in README.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

PENDING = "pending"
CLIP_UPLOADED = "clip_uploaded"
DONE = "done"
FAILED_TERMINAL = "failed_terminal"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS event_queue (
    event_id     TEXT PRIMARY KEY,
    state        TEXT NOT NULL DEFAULT 'pending',
    payload_json TEXT NOT NULL,
    clip_path    TEXT,
    thumb_path   TEXT,
    clip_key     TEXT,
    thumb_key    TEXT,
    attempts     INTEGER NOT NULL DEFAULT 0,
    last_error   TEXT,
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_queue_state ON event_queue(state, created_at);
"""


class EventQueue:
    def __init__(self, db_path: Path | str):
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA)
            self._conn.execute("PRAGMA journal_mode=WAL")

    # ------------------------------------------------------------ write path

    def enqueue(self, event_id: str, payload: dict[str, Any],
                clip_path: Optional[str], thumb_path: Optional[str]) -> None:
        now = time.time()
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO event_queue "
                "(event_id, payload_json, clip_path, thumb_path, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (event_id, json.dumps(payload), clip_path, thumb_path, now, now),
            )

    def mark_clip_uploaded(self, event_id: str, clip_key: str,
                           thumb_key: Optional[str]) -> None:
        self._update(event_id, state=CLIP_UPLOADED, clip_key=clip_key, thumb_key=thumb_key)

    def mark_done(self, event_id: str) -> None:
        self._update(event_id, state=DONE)

    def mark_failed(self, event_id: str, error: str, terminal: bool) -> None:
        with self._lock, self._conn:
            if terminal:
                self._conn.execute(
                    "UPDATE event_queue SET state=?, last_error=?, updated_at=? "
                    "WHERE event_id=?",
                    (FAILED_TERMINAL, error[:500], time.time(), event_id))
            else:
                self._conn.execute(
                    "UPDATE event_queue SET attempts=attempts+1, last_error=?, updated_at=? "
                    "WHERE event_id=?",
                    (error[:500], time.time(), event_id))

    def _update(self, event_id: str, **fields: Any) -> None:
        sets = ", ".join(f"{k}=?" for k in fields)
        with self._lock, self._conn:
            self._conn.execute(
                f"UPDATE event_queue SET {sets}, updated_at=? WHERE event_id=?",
                (*fields.values(), time.time(), event_id),
            )

    # ------------------------------------------------------------ drain path

    def next_batch(self, limit: int = 10) -> list[dict[str, Any]]:
        """Oldest-first replay (buildspec §4.7 step 3)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM event_queue WHERE state IN (?, ?) "
                "ORDER BY created_at ASC LIMIT ?",
                (PENDING, CLIP_UPLOADED, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------ backpressure

    def depth(self) -> int:
        with self._lock:
            (n,) = self._conn.execute(
                "SELECT COUNT(*) FROM event_queue WHERE state IN (?, ?)",
                (PENDING, CLIP_UPLOADED)).fetchone()
        return int(n)

    def oldest_age_s(self) -> Optional[float]:
        with self._lock:
            row = self._conn.execute(
                "SELECT MIN(created_at) FROM event_queue WHERE state IN (?, ?)",
                (PENDING, CLIP_UPLOADED)).fetchone()
        return None if row[0] is None else time.time() - row[0]

    def drop_thumbnails(self) -> list[str]:
        """Level-1 degrade: null out queued thumbs; returns file paths for
        the caller to delete from the spool."""
        with self._lock, self._conn:
            rows = self._conn.execute(
                "SELECT event_id, thumb_path FROM event_queue "
                "WHERE state=? AND thumb_path IS NOT NULL", (PENDING,)).fetchall()
            self._conn.execute(
                "UPDATE event_queue SET thumb_path=NULL, thumb_key=NULL WHERE state=?",
                (PENDING,))
        paths = [r["thumb_path"] for r in rows]
        if paths:
            log.warning("backpressure: dropped %d queued thumbnails", len(paths))
        return paths

    def purge_done(self, keep_last: int = 500) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "DELETE FROM event_queue WHERE state=? AND event_id NOT IN "
                "(SELECT event_id FROM event_queue WHERE state=? "
                " ORDER BY updated_at DESC LIMIT ?)",
                (DONE, DONE, keep_last))

    def close(self) -> None:
        with self._lock:
            self._conn.close()
