from sentinel_edge.lpr import Watchlist, normalize_bg_plate


def test_clean_plate_passes():
    r = normalize_bg_plate("CB1234AB", 0.9)
    assert r and r.plate == "CB1234AB"


def test_separators_and_case_are_stripped():
    r = normalize_bg_plate("cb 1234-ab", 0.9)
    assert r and r.plate == "CB1234AB"


def test_cyrillic_homoglyphs_normalized():
    r = normalize_bg_plate("СВ1234АВ", 0.9)  # Cyrillic С, В, А
    assert r and r.plate == "CB1234AB"


def test_confusables_fixed_positionally():
    # O in the digit block -> 0; 8 in the series block -> B
    r = normalize_bg_plate("CBI2O4A8", 0.9)
    assert r and r.plate == "CB1204AB"


def test_single_region_letter():
    r = normalize_bg_plate("A1234BC", 0.9)
    assert r and r.plate == "A1234BC"


def test_garbage_rejected():
    assert normalize_bg_plate("HELLO", 0.99) is None
    assert normalize_bg_plate("12345678", 0.99) is None
    assert normalize_bg_plate("QQ1234ZZ", 0.99) is None  # Q/Z not BG plate letters


def test_low_confidence_logged_not_alerted():
    assert normalize_bg_plate("CB1234AB", 0.3) is None


def test_watchlist_case_insensitive():
    wl = Watchlist({"cb1234ab"})
    assert wl.hit("CB1234AB")
    assert not wl.hit("CB9999XX")
    wl.update({"CB9999XX"})
    assert wl.hit("cb9999xx") and not wl.hit("CB1234AB")
