import asyncio
import time

import numpy as np

from sentinel_edge.record import ClipRecorder, RingBuffer, SpoolManager


def frame(v=0):
    return np.full((36, 64, 3), v, dtype=np.uint8)


def test_ringbuffer_evicts_by_time_window():
    rb = RingBuffer(window_s=2.0)
    for i in range(50):
        rb.push(i * 0.1, frame(i))
    times = [ts for ts, _ in rb.frames()]
    assert min(times) >= 4.9 - 2.0 and max(times) == 4.9
    assert all(ts >= 4.0 for ts, _ in rb.snapshot(4.0))


async def _record_clip(tmp_path, micro=False):
    encoded = {}

    async def fake_encode(frames, fps, out_path):
        encoded["frames"] = list(frames)
        out_path.write_bytes(b"x" * len(frames))

    async def fake_thumb(f, out_path):
        out_path.write_bytes(b"t")

    spool = SpoolManager(tmp_path / "spool", max_bytes=10_000)
    rec = ClipRecorder("cam-1", spool, fps_hint=10, pre_s=0.5, post_s=0.3,
                       encoder=fake_encode, thumb_encoder=fake_thumb)
    t0 = time.monotonic()
    for i in range(10):  # 1s of pre-roll at 10fps
        rec.push(t0 - 1.0 + i * 0.1, frame(i))

    async def keep_pushing():
        end = time.monotonic() + 0.4
        while time.monotonic() < end:
            rec.push(time.monotonic(), frame(99))
            await asyncio.sleep(0.05)

    result, _ = await asyncio.gather(rec.record(t0, micro=micro), keep_pushing())
    return result, encoded


def test_clip_contains_pre_and_post_roll(tmp_path):
    result, encoded = asyncio.run(_record_clip(tmp_path))
    assert result.clip_path.exists() and result.thumb_path.exists()
    values = {int(f[0, 0, 0]) for f in encoded["frames"]}
    assert 99 in values          # post-roll frames made it in
    assert any(v < 99 for v in values)  # pre-roll frames made it in


def test_micro_clip_is_single_frame(tmp_path):
    result, encoded = asyncio.run(_record_clip(tmp_path, micro=True))
    assert len(encoded["frames"]) == 1
    assert result.clip_path.exists()


def test_spool_deletes_uploaded_files(tmp_path):
    spool = SpoolManager(tmp_path / "spool", max_bytes=10_000)
    p = spool.path_for("a.mp4")
    p.write_bytes(b"x" * 100)
    spool.register(p)
    assert spool.total_bytes() == 100
    spool.mark_uploaded(p)
    assert not p.exists() and spool.total_bytes() == 0
