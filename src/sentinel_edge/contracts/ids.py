"""UUIDv7 generation (RFC 9562) — time-sortable ids required by the contract.

The stdlib gains uuid.uuid7 only in 3.14; this is the standard layout:
48-bit unix-ms timestamp | ver=7 | 12 random bits | var=10 | 62 random bits.
"""
from __future__ import annotations

import os
import time
import uuid


def uuid7() -> str:
    ts_ms = time.time_ns() // 1_000_000
    rand = os.urandom(10)
    b = bytearray(16)
    b[0:6] = ts_ms.to_bytes(6, "big")
    b[6] = 0x70 | (rand[0] & 0x0F)
    b[7] = rand[1]
    b[8] = 0x80 | (rand[2] & 0x3F)
    b[9:16] = rand[3:10]
    return str(uuid.UUID(bytes=bytes(b)))
