"""Cross-validates the edge Pydantic models against the cloud contract's own
example payloads — the closest thing to a wire-compat test without a cloud."""
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from sentinel_edge.contracts import (
    DeviceConfig,
    EventPayload,
    Heartbeat,
    ObjectClass,
    WsEnvelope,
    utcnow_rfc3339,
    uuid7,
)

EXAMPLES = (Path(__file__).resolve().parents[2]
            / "sentinel-cloud/packages/contracts/examples")

needs_examples = pytest.mark.skipif(not EXAMPLES.exists(),
                                    reason="sentinel-cloud examples not on disk")


@needs_examples
@pytest.mark.parametrize("name,model", [
    ("event", EventPayload),
    ("config", DeviceConfig),
    ("heartbeat", Heartbeat),
    ("ws-envelope", WsEnvelope),
])
def test_valid_examples_parse(name, model):
    model.model_validate_json((EXAMPLES / f"{name}.valid.json").read_text())


@needs_examples
@pytest.mark.parametrize("name,model", [
    ("event", EventPayload),          # confidence 1.4 + unknown top-level field
    ("config", DeviceConfig),         # rtsp_url instead of rtsp_ref
    ("heartbeat", Heartbeat),         # device.status "exploded"
    ("ws-envelope", WsEnvelope),      # v: 2
])
def test_invalid_examples_rejected(name, model):
    with pytest.raises(ValidationError):
        model.model_validate_json((EXAMPLES / f"{name}.invalid.json").read_text())


def test_unknown_object_class_coerces_to_unknown():
    assert ObjectClass.coerce("drone") is ObjectClass.unknown
    assert ObjectClass.coerce("person") is ObjectClass.person


def test_uuid7_format_and_time_ordering():
    a, b = uuid7(), uuid7()
    pattern = r"^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
    assert re.match(pattern, a) and re.match(pattern, b)
    assert a[:8] <= b[:8]  # time-sortable prefix


def test_timestamp_is_rfc3339_utc():
    ts = utcnow_rfc3339()
    assert ts.endswith("Z") and "T" in ts
