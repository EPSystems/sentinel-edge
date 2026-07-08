import json

import pytest

from sentinel_edge.config import (
    ConfigCache,
    EdgeSettings,
    PipelineDefaults,
    SecretResolver,
    load_defaults,
    redact_rtsp,
)
from sentinel_edge.contracts import DeviceConfig


def test_settings_urls():
    s = EdgeSettings(cloud_base_url="https://cloud.example.bg")
    assert s.api_base == "https://cloud.example.bg/v1"
    assert s.ws_url == "wss://cloud.example.bg/v1/edge/link"


def test_profile_layering_later_wins(tmp_path):
    (tmp_path / "accel-test.yaml").write_text(
        "backend: hailo\nconf_threshold: 0.6\n", encoding="utf-8")
    (tmp_path / "vertical-test.yaml").write_text(
        "conf_threshold: 0.8\ndwell_default_s: 120\n", encoding="utf-8")
    s = EdgeSettings(profiles_dir=tmp_path, accel_profile="accel-test",
                     vertical_profile="vertical-test")
    d = load_defaults(s)
    assert d.backend == "hailo"          # from accel
    assert d.conf_threshold == 0.8       # vertical overrides accel
    assert d.dwell_default_s == 120
    assert d.rule_cooldown_s == 30.0     # untouched built-in default


def test_missing_profile_is_skipped_not_fatal(tmp_path):
    s = EdgeSettings(profiles_dir=tmp_path, accel_profile="nope",
                     vertical_profile="also-nope")
    d = load_defaults(s)
    assert d.conf_threshold == 0.70


def test_unknown_profile_key_is_rejected(tmp_path):
    (tmp_path / "accel-bad.yaml").write_text("cnf_treshold: 0.9\n", encoding="utf-8")
    s = EdgeSettings(profiles_dir=tmp_path, accel_profile="accel-bad",
                     vertical_profile="none")
    with pytest.raises(Exception):
        load_defaults(s)  # typos in ops files must fail loud, not silently no-op


def test_config_cache_roundtrip_and_corruption(tmp_path):
    cache = ConfigCache(tmp_path)
    assert cache.load() is None
    cfg = DeviceConfig(config_etag="e1", device_id="d1", retention_days=7, cameras=[])
    cache.store(cfg)
    assert cache.load().config_etag == "e1"
    cache.path.write_text("{corrupt", encoding="utf-8")
    assert cache.load() is None          # corrupt cache ignored, not fatal


def test_secret_resolver(tmp_path):
    r = SecretResolver(tmp_path)
    assert r.resolve("rtsp://direct/dev") == "rtsp://direct/dev"
    with pytest.raises(KeyError):
        r.resolve("secret://cam/abc")
    (tmp_path / "secrets.json").write_text(
        json.dumps({"secret://cam/abc": "rtsp://u:p@10.0.0.5/s"}), encoding="utf-8")
    assert r.resolve("secret://cam/abc") == "rtsp://u:p@10.0.0.5/s"


def test_redact_rtsp_never_leaks_password():
    url = "rtsp://admin:hunter2@10.0.0.5:554/stream1"
    red = redact_rtsp(url)
    assert "hunter2" not in red
    assert red == "rtsp://admin:***@10.0.0.5:554/stream1"
    assert redact_rtsp("rtsp://10.0.0.5/nocreds") == "rtsp://10.0.0.5/nocreds"


def test_pipeline_defaults_class_map_types():
    d = PipelineDefaults()
    assert d.class_map[0] == "person" and d.class_map[7] == "truck"
