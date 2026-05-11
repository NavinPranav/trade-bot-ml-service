"""Phase 4.5 — per-(direction, horizon) calibration map tests.

These extend the Phase 4.3 suite with cases that exercise the new keyed
storage and the lookup fallback chain. The Phase 4.3 suite (
``test_calibration.py``) continues to test the back-compat path where samples
omit ``direction`` and ``horizon`` entirely.

What we verify
--------------
1. Sample bucketing — direction/horizon labelled samples land in the
   corresponding ``(d, h)`` bucket plus the ``(d, "*")`` and ``("*", h)``
   wildcards plus the global ``("*", "*")``.
2. Lookup chain — ``apply`` returns the most-specific available map and
   falls back gracefully when the specific bucket is below ``min_samples``.
3. Direction labels are normalised (``BULLISH`` → ``BUY``).
4. Persistence migration — a v1 (Phase 4.3) state file on disk is auto-read
   into the global ``("*", "*")`` slot of the new store without operator
   action.
5. Per-key Brier improvement — fitting different win rates for BUY vs SELL
   produces different calibrated outputs for the same raw confidence.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest

from app.inference.calibration import (
    CalibrationSample,
    WILDCARD,
    _normalize_direction,
    _normalize_horizon,
    get_calibration_store,
    reset_calibration_store_for_tests,
)


@pytest.fixture(autouse=True)
def _isolate_store(tmp_path, monkeypatch):
    monkeypatch.setattr("app.inference.calibration.settings.model_dir", str(tmp_path))
    monkeypatch.setattr("app.inference.calibration.settings.calibration_min_samples", 30)
    reset_calibration_store_for_tests()
    yield
    reset_calibration_store_for_tests()


def _samples_with_winrate(n: int, win_rate: float, direction: str, horizon: str) -> List[CalibrationSample]:
    """Build N samples with linearly-spaced confidences and the requested
    target win rate. Wins are deterministic (every Kth sample is a win) so
    the per-bucket Brier comparisons are reproducible."""
    out: List[CalibrationSample] = []
    target_wins = int(round(n * win_rate))
    for i in range(n):
        # Spread confidences over [40, 95] so all bins get hit.
        conf = 40.0 + (55.0 * i / max(1, n - 1))
        win = i < target_wins  # deterministic block of wins at the start
        out.append(CalibrationSample(
            confidence=conf, win=win, direction=direction, horizon=horizon,
        ))
    return out


# ── Normalisation ───────────────────────────────────────────────────────────


def test_normalize_direction_handles_aliases():
    assert _normalize_direction("BUY") == "BUY"
    assert _normalize_direction("bullish") == "BUY"
    assert _normalize_direction("LONG") == "BUY"
    assert _normalize_direction("SELL") == "SELL"
    assert _normalize_direction("bearish") == "SELL"
    assert _normalize_direction("HOLD") == WILDCARD
    assert _normalize_direction(None) == WILDCARD
    assert _normalize_direction("") == WILDCARD


def test_normalize_horizon_uppercases_and_wildcards_empty():
    assert _normalize_horizon("15m") == "15M"
    assert _normalize_horizon(" 1H ") == "1H"
    assert _normalize_horizon(None) == WILDCARD
    assert _normalize_horizon("") == WILDCARD


# ── Fit + apply with labelled samples ───────────────────────────────────────


def test_fit_builds_per_direction_per_horizon_maps():
    """Two distinct buckets (BUY/15M with 80% win rate, SELL/15M with 30%)
    should each produce their own map plus the wildcard fallbacks."""
    store = get_calibration_store()
    samples = (
        _samples_with_winrate(60, 0.80, "BUY", "15M")
        + _samples_with_winrate(60, 0.30, "SELL", "15M")
    )
    result = store.fit(samples)
    assert result["active"] is True

    keys = {(m["direction"], m["horizon"]) for m in result["maps"]}
    # Specific buckets + per-direction collapses + per-horizon collapse + global
    assert ("BUY", "15M") in keys
    assert ("SELL", "15M") in keys
    assert ("BUY", WILDCARD) in keys
    assert ("SELL", WILDCARD) in keys
    assert (WILDCARD, "15M") in keys
    assert (WILDCARD, WILDCARD) in keys


def test_apply_uses_specific_map_when_available():
    """For the same raw confidence, BUY and SELL should map to different
    calibrated values when their underlying win rates differ markedly."""
    store = get_calibration_store()
    store.fit(
        _samples_with_winrate(80, 0.80, "BUY", "15M")
        + _samples_with_winrate(80, 0.30, "SELL", "15M")
    )
    cal_buy = store.apply(70.0, direction="BUY", horizon="15M")
    cal_sell = store.apply(70.0, direction="SELL", horizon="15M")
    # BUY samples win ~80% of the time, SELL ~30% — the calibrated values
    # must be ordered the same way (isotonic + per-bucket fit).
    assert cal_buy > cal_sell + 5.0


def test_apply_falls_back_to_global_when_specific_bucket_short():
    """A direction/horizon combo below min_samples must walk the chain to
    the next-most-specific map that exists."""
    store = get_calibration_store()
    # Fit only the global bucket: omit direction/horizon labels so only
    # ("*", "*") gets enough samples.
    store.fit([
        CalibrationSample(confidence=c, win=(c >= 60), direction=None, horizon=None)
        for c in [40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90] * 5
    ])
    # Asking for a specific (BUY, 15M) → falls back to global.
    cal_specific = store.apply(70.0, direction="BUY", horizon="15M")
    cal_global = store.apply(70.0)
    assert cal_specific == pytest.approx(cal_global, abs=1e-6)


def test_apply_with_key_reports_used_key():
    store = get_calibration_store()
    store.fit(
        _samples_with_winrate(80, 0.80, "BUY", "15M")
        + _samples_with_winrate(80, 0.30, "SELL", "15M")
    )

    val_specific, key_specific = store.apply_with_key(70.0, direction="BUY", horizon="15M")
    assert key_specific == ("BUY", "15M")
    assert val_specific != 70.0  # must have been calibrated through the bucket

    # An untrained horizon falls back through the chain.
    val_fallback, key_fallback = store.apply_with_key(70.0, direction="BUY", horizon="3D")
    # 3D was never fit, so we expect (BUY, "*") which IS fit since the BUY
    # direction collapse reaches min_samples.
    assert key_fallback == ("BUY", WILDCARD)
    assert val_fallback != 70.0


def test_lookup_chain_dedupes_when_arguments_are_wildcards():
    """When direction and horizon are both omitted the chain collapses to a
    single key; the apply call must still work without raising."""
    store = get_calibration_store()
    store.fit([
        CalibrationSample(confidence=c, win=(c >= 60))
        for c in [40, 50, 60, 70, 80] * 10
    ])
    # Identical to legacy Phase 4.3 call signature.
    val = store.apply(70.0)
    assert val != 70.0  # calibration is active and applied


# ── Status + persistence ────────────────────────────────────────────────────


def test_status_reports_each_map_individually():
    store = get_calibration_store()
    store.fit(
        _samples_with_winrate(80, 0.80, "BUY", "15M")
        + _samples_with_winrate(80, 0.30, "SELL", "15M")
    )
    status = store.status()

    assert status["active"] is True
    assert status["n_maps"] >= 4  # at least the four per-direction/horizon variants
    assert "maps" in status and isinstance(status["maps"], dict)
    # Back-compat: top-level bins/metrics still surface the global map.
    assert isinstance(status.get("bins"), list)
    # Per-map payload includes the raw inputs.
    sample_payload = next(iter(status["maps"].values()))
    assert "direction" in sample_payload and "horizon" in sample_payload
    assert "n_samples" in sample_payload and "bins" in sample_payload


def test_persistence_roundtrip_keyed(tmp_path, monkeypatch):
    """Fitted multi-map state must round-trip through disk."""
    monkeypatch.setattr("app.inference.calibration.settings.model_dir", str(tmp_path))
    monkeypatch.setattr("app.inference.calibration.settings.calibration_min_samples", 30)

    store_a = get_calibration_store()
    store_a.fit(
        _samples_with_winrate(80, 0.80, "BUY", "15M")
        + _samples_with_winrate(80, 0.30, "SELL", "15M")
    )
    cal_a = store_a.apply(70.0, direction="BUY", horizon="15M")

    reset_calibration_store_for_tests()
    store_b = get_calibration_store()
    assert store_b.is_active()
    cal_b = store_b.apply(70.0, direction="BUY", horizon="15M")
    assert cal_a == pytest.approx(cal_b, abs=1e-6)


def test_persistence_migrates_legacy_v1_format(tmp_path, monkeypatch):
    """A v1 (Phase 4.3) state file with a flat ``bins`` list at the top must
    be auto-migrated into the global ``("*", "*")`` slot when the new store
    boots, without requiring any operator action."""
    monkeypatch.setattr("app.inference.calibration.settings.model_dir", str(tmp_path))
    monkeypatch.setattr("app.inference.calibration.settings.calibration_min_samples", 30)

    legacy_path = Path(tmp_path) / "calibration_state.json"
    legacy_path.write_text(json.dumps({
        "meta": {"active": True, "n_samples": 200, "fitted_at_unix_ms": 0},
        # NB: no "version" key, mimicking what Phase 4.3 wrote.
        "bins": [
            {"lower": 0.0,  "upper": 25.0, "calibrated": 20.0, "raw_win_rate": 18.0, "n_samples": 50},
            {"lower": 25.0, "upper": 50.0, "calibrated": 35.0, "raw_win_rate": 32.0, "n_samples": 50},
            {"lower": 50.0, "upper": 75.0, "calibrated": 55.0, "raw_win_rate": 60.0, "n_samples": 50},
            {"lower": 75.0, "upper": 100.0,"calibrated": 80.0, "raw_win_rate": 85.0, "n_samples": 50},
        ],
    }))

    store = get_calibration_store()
    assert store.is_active()
    # Lookup against the global slot — should pick up the migrated bins.
    assert store.apply(60.0) == pytest.approx(55.0, abs=1e-6)
    # And asking for a specific (direction, horizon) falls back to that same global map.
    assert store.apply(60.0, direction="BUY", horizon="15M") == pytest.approx(55.0, abs=1e-6)


# ── Integration through _coerce_result ──────────────────────────────────────


def test_coerce_result_uses_keyed_calibration_per_direction():
    """End-to-end: a BUY prediction at conf=70 should pick up the BUY map's
    calibration and a SELL at the same raw conf should pick up the SELL map's,
    and the two final ``confidence`` values should differ."""
    from app.inference.gemini_predictor import _coerce_result

    store = get_calibration_store()
    store.fit(
        _samples_with_winrate(80, 0.85, "BUY", "15M")
        + _samples_with_winrate(80, 0.30, "SELL", "15M")
    )

    realtime = 47000.0
    raw_buy = {
        "direction": "BUY", "magnitude": 0.5, "confidence": 70.0,
        "predicted_volatility": 1.0, "valid_minutes": 15, "risk_reward": 2.0,
        "entry_price": realtime, "stop_loss": realtime * 0.99, "target_price": realtime * 1.02,
    }
    raw_sell = dict(raw_buy)
    raw_sell["direction"] = "SELL"
    raw_sell["stop_loss"] = realtime * 1.01
    raw_sell["target_price"] = realtime * 0.98

    res_buy = _coerce_result(raw_buy, realtime, horizon="15M")
    res_sell = _coerce_result(raw_sell, realtime, horizon="15M")

    diag_buy = res_buy.get("_diagnostics", {})
    diag_sell = res_sell.get("_diagnostics", {})

    # Both must have applied calibration (active store, directional model).
    assert diag_buy.get("calibration_applied") is True
    assert diag_sell.get("calibration_applied") is True
    # And they must have used DIFFERENT keys.
    assert diag_buy.get("calibration_key") != diag_sell.get("calibration_key")
    # The BUY calibrated confidence should sit higher than the SELL one given
    # the underlying 85% vs 30% win rates.
    assert diag_buy.get("calibrated_confidence") > diag_sell.get("calibrated_confidence") + 5.0
