"""Tests for the outcome-driven confidence calibration (Phase 4.3).

Two layers under test:

1. ``CalibrationStore`` — pure function: PAV-based isotonic fit of
   per-bin win rates. No Gemini, no FastAPI. Verifies cold-start guard,
   monotonicity, identity passthrough, persistence, and a simple metric.
2. End-to-end through ``_coerce_result`` — once the store is active, raw
   Gemini confidence should be replaced by the calibrated value before the
   confidence-floor check, surfacing both numbers in ``_diagnostics``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.inference.calibration import (
    CalibrationSample,
    _build_bins,
    _lookup_calibrated,
    _pav_isotonic,
    get_calibration_store,
    reset_calibration_store_for_tests,
)


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    """Each test starts with a fresh store and an isolated persist file."""
    monkeypatch.setattr("app.inference.calibration.settings.model_dir", str(tmp_path))
    reset_calibration_store_for_tests()
    yield
    reset_calibration_store_for_tests()


# ── PAV primitive ────────────────────────────────────────────────────────────


def test_pav_isotonic_already_monotone_is_identity():
    v = np.array([0.1, 0.2, 0.4, 0.7, 1.0])
    w = np.array([1, 1, 1, 1, 1], dtype=float)
    out = _pav_isotonic(v, w)
    assert np.allclose(out, v)


def test_pav_isotonic_pools_violators():
    # 0.4 → 0.2 violates; PAV pools to 0.3 each.
    v = np.array([0.1, 0.4, 0.2, 0.7])
    w = np.array([1, 1, 1, 1], dtype=float)
    out = _pav_isotonic(v, w)
    # Result must be monotone-non-decreasing.
    assert np.all(np.diff(out) >= -1e-9)
    # Pooled bin → average of the two violators (equal weights).
    assert np.isclose(out[1], 0.3)
    assert np.isclose(out[2], 0.3)


def test_pav_isotonic_respects_weights():
    v = np.array([0.5, 0.0])
    w = np.array([3.0, 1.0])
    out = _pav_isotonic(v, w)
    # Weighted mean → (0.5*3 + 0.0*1) / 4 = 0.375
    assert np.allclose(out, [0.375, 0.375])


# ── Histogram + bin construction ─────────────────────────────────────────────


def test_build_bins_handles_empty_input():
    bins, metrics = _build_bins([], n_bins=5, prior_n=4.0, prior_p=0.5)
    assert len(bins) == 5
    assert metrics["n_samples"] == 0
    # Empty bins fall back to the prior.
    assert all(abs(b.calibrated - 50.0) < 1e-6 for b in bins)


def test_build_bins_monotone_after_pav():
    # Crafted: high confidence loses a lot, low confidence wins a lot — anti-monotone raw.
    samples = [CalibrationSample(confidence=95.0, win=False) for _ in range(10)]
    samples += [CalibrationSample(confidence=10.0, win=True) for _ in range(10)]
    bins, metrics = _build_bins(samples, n_bins=10, prior_n=2.0, prior_p=0.5)
    cal = [b.calibrated for b in bins]
    # PAV must enforce monotone-non-decreasing across bins.
    diffs = np.diff(cal)
    assert np.all(diffs >= -1e-6), f"non-monotone after PAV: {cal}"


def test_lookup_calibrated_uses_correct_bin():
    samples = [CalibrationSample(confidence=10.0, win=False) for _ in range(20)]
    samples += [CalibrationSample(confidence=80.0, win=True) for _ in range(20)]
    bins, _ = _build_bins(samples, n_bins=10, prior_n=2.0, prior_p=0.5)

    low = _lookup_calibrated(bins, 12.0)
    high = _lookup_calibrated(bins, 82.0)
    assert high > low

    # Exact upper bound should still resolve (last bin is inclusive).
    assert _lookup_calibrated(bins, 100.0) == bins[-1].calibrated


# ── Store: cold start, fit/apply, persistence ────────────────────────────────


def _make_calibration_dataset(n=100, seed=42):
    """Roughly-monotone synthetic outcomes: higher confidence → higher win prob."""
    rng = np.random.default_rng(seed)
    confs = rng.uniform(40.0, 95.0, size=n)
    # True win probability scales with confidence (50% @ conf=40, 90% @ conf=90).
    p = 0.5 + (confs - 40.0) / 100.0 * 0.8
    wins = rng.uniform(0.0, 1.0, size=n) < p
    return [
        CalibrationSample(confidence=float(c), win=bool(w))
        for c, w in zip(confs, wins)
    ]


def test_cold_start_apply_is_identity():
    store = get_calibration_store()
    assert not store.is_active()
    assert store.apply(55.0) == 55.0
    assert store.apply(72.5) == 72.5


def test_fit_below_min_samples_stays_inactive(monkeypatch):
    monkeypatch.setattr("app.inference.calibration.settings.calibration_min_samples", 50)
    store = get_calibration_store()
    samples = _make_calibration_dataset(n=20)
    result = store.fit(samples)
    assert result["active"] is False
    assert result["reason"] == "insufficient_samples"
    assert result["n_samples"] == 20
    assert not store.is_active()
    # Apply still passes raw value through.
    assert store.apply(70.0) == 70.0


def test_fit_above_min_samples_activates_and_calibrates(monkeypatch):
    monkeypatch.setattr("app.inference.calibration.settings.calibration_min_samples", 30)
    store = get_calibration_store()
    samples = _make_calibration_dataset(n=200)
    result = store.fit(samples)

    assert result["active"] is True
    assert store.is_active()
    assert result["n_samples"] == 200
    metrics = result["metrics"]
    # Brier score should not get worse after calibration.
    assert metrics["brier_after"] <= metrics["brier_before"] + 1e-6


def test_fit_replaces_extreme_unrealistic_confidence_with_history(monkeypatch):
    """Gemini insists on conf=90 but historical wins are only ~50%.
    Calibrated value should drop sharply below 90."""
    monkeypatch.setattr("app.inference.calibration.settings.calibration_min_samples", 30)
    store = get_calibration_store()
    samples = [CalibrationSample(confidence=90.0, win=False) for _ in range(40)]
    samples += [CalibrationSample(confidence=90.0, win=True) for _ in range(40)]
    store.fit(samples)

    calibrated = store.apply(90.0)
    # Win rate ≈ 50%, calibrated value should be near 50, never near 90.
    assert 30.0 <= calibrated <= 65.0


def test_clear_disables_and_passes_through(monkeypatch):
    monkeypatch.setattr("app.inference.calibration.settings.calibration_min_samples", 30)
    store = get_calibration_store()
    store.fit(_make_calibration_dataset(n=200))
    assert store.is_active()
    store.clear()
    assert not store.is_active()
    assert store.apply(75.0) == 75.0


def test_persistence_roundtrip(tmp_path, monkeypatch):
    """A fresh store created after a fit should pick up the persisted bins."""
    monkeypatch.setattr("app.inference.calibration.settings.model_dir", str(tmp_path))
    monkeypatch.setattr("app.inference.calibration.settings.calibration_min_samples", 30)

    store_a = get_calibration_store()
    store_a.fit(_make_calibration_dataset(n=200))
    cal_a = store_a.apply(70.0)
    assert store_a.is_active()

    # New store instance — must read the file and remain active.
    reset_calibration_store_for_tests()
    store_b = get_calibration_store()
    assert store_b.is_active()
    assert store_b.apply(70.0) == pytest.approx(cal_a, abs=1e-6)

    # Persisted file is human-readable JSON.
    persisted = json.loads(Path(tmp_path, "calibration_state.json").read_text())
    assert persisted["meta"]["active"] is True
    assert len(persisted["bins"]) >= 1


def test_status_payload_shape(monkeypatch):
    monkeypatch.setattr("app.inference.calibration.settings.calibration_min_samples", 30)
    store = get_calibration_store()
    store.fit(_make_calibration_dataset(n=200))
    status = store.status()

    assert status["active"] is True
    assert status["n_samples"] == 200
    assert isinstance(status["bins"], list) and len(status["bins"]) >= 1
    bin0 = status["bins"][0]
    for key in ("lower", "upper", "calibrated", "raw_win_rate", "n_samples"):
        assert key in bin0


# ── Integration through _coerce_result ──────────────────────────────────────


def _basic_raw(direction: str = "BUY", confidence: float = 70.0) -> dict:
    return {
        "direction": direction,
        "confidence": confidence,
        "magnitude": 0.5,
        "entry_price": 50_000,
        "stop_loss": 49_900,
        "target_price": 50_200,
        "valid_minutes": 15,
        "reason": "stub",
    }


def test_coerce_uses_raw_confidence_when_calibration_inactive():
    from app.inference.gemini_predictor import _coerce_result

    out = _coerce_result(
        _basic_raw(confidence=72.0),
        50_000.0,
        indicators={"volume_ratio": 1.4},
        trend_context={"regime": "UPTREND", "primary_regime": "UPTREND", "higher_regime": "UPTREND"},
    )
    diag = out["_diagnostics"]
    assert diag["calibration_applied"] is False
    assert diag["raw_confidence"] == 72.0
    assert diag["calibrated_confidence"] == 72.0


def test_coerce_replaces_confidence_with_calibrated_when_active(monkeypatch):
    from app.inference.gemini_predictor import _coerce_result

    monkeypatch.setattr("app.inference.calibration.settings.calibration_min_samples", 30)
    store = get_calibration_store()
    # Train: at conf=70 the historical win rate is ~30%, so calibrated should drop.
    samples = [CalibrationSample(confidence=70.0, win=False) for _ in range(70)]
    samples += [CalibrationSample(confidence=70.0, win=True) for _ in range(30)]
    store.fit(samples)
    assert store.is_active()

    out = _coerce_result(
        _basic_raw(confidence=70.0),
        50_000.0,
        indicators={"volume_ratio": 1.4},
        trend_context={"regime": "UPTREND", "primary_regime": "UPTREND", "higher_regime": "UPTREND"},
    )
    diag = out["_diagnostics"]
    assert diag["calibration_applied"] is True
    assert diag["raw_confidence"] == 70.0
    # Calibrated must shift toward the empirical win rate (~30%).
    assert diag["calibrated_confidence"] < diag["raw_confidence"]
    # The headline `confidence` field is the calibrated value (used by the floor check).
    assert out["confidence"] == diag["calibrated_confidence"]


def test_coerce_calibration_drop_can_trigger_low_confidence_gate(monkeypatch):
    """Raw conf 75 (above floor) + calibration says 30% historical → HOLD via low-confidence gate."""
    from app.inference.gemini_predictor import _coerce_result

    monkeypatch.setattr("app.inference.calibration.settings.calibration_min_samples", 30)
    store = get_calibration_store()
    samples = [CalibrationSample(confidence=75.0, win=False) for _ in range(80)]
    samples += [CalibrationSample(confidence=75.0, win=True) for _ in range(20)]
    store.fit(samples)

    out = _coerce_result(
        _basic_raw(confidence=75.0),
        50_000.0,
        # SIDEWAYS — strict 65 floor applies.
        indicators={"volume_ratio": 1.4},
        trend_context={"regime": "SIDEWAYS", "primary_regime": "SIDEWAYS", "higher_regime": "SIDEWAYS"},
    )
    diag = out["_diagnostics"]
    assert diag["calibration_applied"] is True
    # Calibrated < strict floor → HOLD.
    assert out["direction"] == "HOLD"
    assert diag["gates"]["low_confidence"] is True


def test_coerce_does_not_calibrate_hold(monkeypatch):
    """HOLD predictions are direction-agnostic; no calibration should be applied."""
    from app.inference.gemini_predictor import _coerce_result

    monkeypatch.setattr("app.inference.calibration.settings.calibration_min_samples", 30)
    store = get_calibration_store()
    store.fit(_make_calibration_dataset(n=200))

    out = _coerce_result(
        _basic_raw(direction="HOLD", confidence=50.0),
        50_000.0,
        indicators={"volume_ratio": 1.4},
        trend_context={"regime": "SIDEWAYS", "primary_regime": "SIDEWAYS", "higher_regime": "SIDEWAYS"},
    )
    diag = out["_diagnostics"]
    assert diag["calibration_applied"] is False
    assert diag["raw_confidence"] == diag["calibrated_confidence"] == 50.0
