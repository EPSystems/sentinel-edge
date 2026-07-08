"""End-to-end: synthetic frames -> motion gate -> mock detector -> tracker ->
rule engine -> clip recorder -> durable queue. Proves the whole pipeline is
operational with no camera, model, accelerator, ffmpeg, or cloud."""
import asyncio
import json
import shutil
from pathlib import Path

from sentinel_edge.main import run_simulation

CONFIG = Path(__file__).resolve().parents[1] / "configs/examples/device-config.json"


def test_full_pipeline_produces_zone_and_line_events(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    shutil.copy(CONFIG, tmp_path / "device-config.json")
    produced = asyncio.run(run_simulation(tmp_path / "device-config.json",
                                          duration_s=8.0))
    assert produced >= 2

    # inspect the queued payloads directly from the durable journal
    import sqlite3
    conn = sqlite3.connect(tmp_path / "data/simulate/event-queue.sqlite3")
    rows = conn.execute("SELECT payload_json, clip_path FROM event_queue").fetchall()
    conn.close()
    rule_types = set()
    for payload_json, clip_path in rows:
        payload = json.loads(payload_json)
        rule_types.add(payload["rule_type"])
        assert payload["object_class"] == "person"
        assert Path(clip_path).exists()          # clip really written to spool
    assert "zone_enter" in rule_types
    assert "line_cross" in rule_types
