"""Bench tooling: run the real pipeline against a real camera with no cloud
(`sentinel-edge local`) and draw zones/lines in a browser against a live
snapshot (`sentinel-edge zones`). Dev/bench only — production devices are
cloud-authored (Flow B) and never edit zones locally.
"""
from .runner import run_local
from .zones_tool import run_zones_tool

__all__ = ["run_local", "run_zones_tool"]
