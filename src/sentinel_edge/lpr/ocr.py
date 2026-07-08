"""Plate OCR + Bulgarian-plate post-processing (buildspec §4.5).

Plates are property identifiers, not GDPR biometrics (CANONICAL-FACTS §6).
The OCR engine (fast-plate-ocr, [lpr] extra) is lazy-loaded; the BG
normalisation + watchlist matching below are pure and unit-tested.

BG civilian format: 1-2 region letters + 4 digits + 2 series letters,
using only letters shared by Cyrillic and Latin glyphs (A B C E H K M O P T X Y).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

BG_PLATE_LETTERS = set("ABCEHKMOPTXY")
BG_PLATE_RE = re.compile(r"^[ABCEHKMOPTXY]{1,2}\d{4}[ABCEHKMOPTXY]{2}$")

# confusable corrections applied positionally: digits slots vs letter slots
_TO_DIGIT = {"O": "0", "I": "1", "Z": "2", "S": "5", "B": "8", "G": "6", "T": "7"}
_TO_LETTER = {"0": "O", "1": "I", "8": "B", "5": "S", "2": "Z"}
# Cyrillic homoglyphs -> Latin
_CYR = str.maketrans("АВСЕНКМОРТХУ", "ABCEHKMOPTXY")


@dataclass
class PlateRead:
    plate: str
    confidence: float
    raw: str


def normalize_bg_plate(raw: str, confidence: float, min_confidence: float = 0.5
                       ) -> Optional[PlateRead]:
    """Uppercase, strip separators, fix homoglyphs/confusables positionally,
    validate against the BG format. Below min_confidence the read is logged
    but not returned (alert floor, buildspec §4.5 step 3)."""
    text = re.sub(r"[\s\-\.]", "", raw.upper()).translate(_CYR)
    if not (6 <= len(text) <= 8):
        return None

    # find the 4-digit core; letters before are region, after are series
    for region_len in (2, 1):
        if len(text) != region_len + 6:
            continue
        region = "".join(_TO_LETTER.get(c, c) for c in text[:region_len])
        digits = "".join(_TO_DIGIT.get(c, c) for c in text[region_len:region_len + 4])
        series = "".join(_TO_LETTER.get(c, c) for c in text[region_len + 4:])
        candidate = region + digits + series
        if BG_PLATE_RE.match(candidate):
            if confidence < min_confidence:
                log.info("plate %s below confidence floor (%.2f) — logged, not alerted",
                         candidate, confidence)
                return None
            return PlateRead(plate=candidate, confidence=confidence, raw=raw)
    return None


class PlateReader:
    """Runs once per vehicle track (not per frame) to protect the accelerator
    budget. Crops around the vehicle bbox and OCRs it."""

    def __init__(self, min_confidence: float = 0.5):
        self.min_confidence = min_confidence
        self._ocr = None

    def _ensure_loaded(self) -> None:
        if self._ocr is None:
            try:
                from fast_plate_ocr import ONNXPlateRecognizer  # type: ignore[import-not-found]
            except ImportError as e:
                raise RuntimeError(
                    "fast-plate-ocr not installed — pip install 'sentinel-edge[lpr]'") from e
            self._ocr = ONNXPlateRecognizer("european-plates-mobile-vit-v2-model")

    def read(self, vehicle_crop_bgr: np.ndarray) -> Optional[PlateRead]:
        self._ensure_loaded()
        assert self._ocr is not None
        results = self._ocr.run(vehicle_crop_bgr, return_confidence=True)
        if not results:
            return None
        text, conf = results[0] if isinstance(results, list) else results
        conf_val = float(np.mean(conf)) if np.ndim(conf) else float(conf)
        return normalize_bg_plate(str(text), conf_val, self.min_confidence)


class Watchlist:
    """Per-client manual watchlist, synced via config (watchlist_ref). The
    shared cross-customer registry on the landing page is out of MVP scope
    and legally gated — this is deliberately per-tenant only."""

    def __init__(self, plates: Optional[set[str]] = None):
        self._plates = {p.upper() for p in (plates or set())}

    def update(self, plates: set[str]) -> None:
        self._plates = {p.upper() for p in plates}

    def hit(self, plate: str) -> bool:
        return plate.upper() in self._plates
