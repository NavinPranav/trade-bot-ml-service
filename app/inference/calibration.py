"""
Outcome-driven confidence calibration (Phase 4.3 + 4.5).

Maps Gemini's *raw* directional confidence to a probability backed by the
historical hit-rate of past directional predictions resolved by the backend's
``OutcomeResolutionService``. The mapping is fitted with **histogram binning +
Pool-Adjacent-Violators (PAV)** which gives an isotonic, monotone-non-decreasing
calibration curve without pulling in scikit-learn.

Why this matters
----------------
The downstream confidence floor (default 65) is meaningful only if a Gemini
``confidence=70`` actually corresponds to a ~70% historical win rate. In
practice LLMs are systematically over- or under-confident; calibration
recovers the operating curve from data so the floor doesn't gate trades on
arbitrary numbers.

Phase 4.5 — Per-(direction, horizon) maps
-----------------------------------------
Phase 4.3 fitted *one* global curve. In production BUY win-rates and SELL
win-rates often differ systematically (asymmetric volatility — markets fall
faster than they rise) and different horizons (15M vs 1H) sit on different
base rates. A single average curve produces calibrated confidences that are
wrong in *both* directions for any trade that doesn't fit the population mean.

Phase 4.5 stores a dictionary of bin maps keyed by ``(direction, horizon)``
where each component can be a concrete value (``"BUY"``, ``"15M"``) or the
wildcard ``"*"``. On ``apply()`` the store walks a deterministic fallback
chain so every trade gets the most-specific calibration that has enough
samples behind it:

    1. (direction, horizon)        — most specific
    2. (direction, "*")            — collapse horizons, keep direction
    3. ("*", horizon)              — collapse directions, keep horizon
    4. ("*", "*")                  — global; back-compat with Phase 4.3
    5. raw confidence passthrough  — cold start

Design
------
* **Pure numpy** — keeps the deploy small and avoids scipy/sklearn wheels on
  Python 3.14 (Render).
* **Histogram binning** — predictable, auditable bins (default 10 bins of
  width 10%) with Beta-style smoothing for sparse buckets.
* **PAV** — applied across bin centres so the resulting mapping is
  monotone-non-decreasing.
* **Cold-start safe** — if fewer than ``min_samples`` resolved predictions
  are provided per key, that key stays unfitted; ``apply()`` walks down the
  fallback chain and ultimately returns the raw confidence unchanged when no
  key qualifies. Inference falls back to the existing 65-floor.
* **Persistent** — fitted maps are written to ``settings.model_dir`` so a
  Render restart doesn't lose calibration. Legacy single-map state from
  Phase 4.3 is auto-migrated into the ``("*", "*")`` slot on load.
* **Back-compat** — ``CalibrationSample(confidence=..., win=...)``,
  ``store.fit(samples)`` and ``store.apply(raw_confidence)`` continue to work
  without direction / horizon arguments. All Phase 4.3 callers and tests
  remain unchanged.

Public surface
--------------
* ``CalibrationStore`` singleton via ``get_calibration_store()``.
* ``CalibrationStore.fit(samples)`` returns a status dict with metrics.
* ``CalibrationStore.apply(raw_confidence, direction=None, horizon=None)`` is
  what inference calls.
* ``CalibrationStore.status()`` powers the admin GET endpoint and now
  surfaces every fitted key.
* ``CalibrationStore.clear()`` powers the admin DELETE endpoint.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
from loguru import logger

from app.config import settings


# ── Public sample shape used by the admin endpoint and tests ────────────────


@dataclass(frozen=True)
class CalibrationSample:
    """One resolved prediction worth using for calibration.

    ``direction`` and ``horizon`` are optional: when both are missing every
    sample lands in the ``("*", "*")`` global bucket and the store behaves
    exactly like the Phase 4.3 single-map version. When supplied they enable
    the per-(direction, horizon) maps introduced in Phase 4.5.
    """

    confidence: float                # raw Gemini confidence on 0–100 scale
    win: bool                        # True if the prediction made money / hit target
    direction: Optional[str] = None  # "BUY" / "SELL" (HOLD samples should be filtered out before fitting)
    horizon: Optional[str] = None    # "5M" / "15M" / "1H" / "1D" etc.


# ── Helpers: key normalisation ──────────────────────────────────────────────


# A key is always a tuple of two strings, where "*" means "wildcard / collapse
# this dimension". This keeps lookup chains, persistence, and status output
# simple and Python-dict-friendly.
KeyType = Tuple[str, str]
WILDCARD: str = "*"


def _normalize_direction(value: Optional[str]) -> str:
    """Normalise a direction label to ``"BUY"`` / ``"SELL"`` / ``"*"``.

    Accepts a few common aliases (``"BULLISH"`` → ``"BUY"`` etc.) so backend
    exports written before the enum was tightened still slot into the right
    bucket. Anything else collapses to the wildcard.
    """
    if value is None:
        return WILDCARD
    s = str(value).strip().upper()
    if s in ("BUY", "BULLISH", "LONG"):
        return "BUY"
    if s in ("SELL", "BEARISH", "SHORT"):
        return "SELL"
    return WILDCARD


def _normalize_horizon(value: Optional[str]) -> str:
    """Normalise a horizon string. Empty / None → wildcard."""
    if value is None:
        return WILDCARD
    s = str(value).strip().upper()
    return s or WILDCARD


# ── Internal binning + PAV implementation ───────────────────────────────────


@dataclass
class _BinFit:
    """One histogram bin in the fitted calibration map."""

    lower: float           # raw confidence ≥ this (inclusive)
    upper: float           # raw confidence < this (exclusive); last bin is inclusive
    calibrated: float      # output on 0–100 scale
    raw_win_rate: float    # observed win rate before monotonization
    n_samples: int


def _pav_isotonic(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Pool-Adjacent-Violators isotonic regression. Given a sequence of values
    with weights, returns a monotone-non-decreasing sequence (same length)
    that minimises weighted squared error.
    """
    n = len(values)
    if n == 0:
        return np.array([], dtype=float)

    blocks_v = [float(v) for v in values]
    blocks_w = [max(1e-9, float(w)) for w in weights]
    blocks_idx: list[list[int]] = [[i] for i in range(n)]

    i = 0
    while i < len(blocks_v) - 1:
        if blocks_v[i] > blocks_v[i + 1] + 1e-12:
            new_w = blocks_w[i] + blocks_w[i + 1]
            new_v = (
                blocks_v[i] * blocks_w[i] + blocks_v[i + 1] * blocks_w[i + 1]
            ) / new_w
            blocks_v[i] = new_v
            blocks_w[i] = new_w
            blocks_idx[i] = blocks_idx[i] + blocks_idx[i + 1]
            del blocks_v[i + 1]
            del blocks_w[i + 1]
            del blocks_idx[i + 1]
            if i > 0:
                i -= 1
        else:
            i += 1

    out = np.zeros(n, dtype=float)
    for v, idxs in zip(blocks_v, blocks_idx):
        for idx in idxs:
            out[idx] = v
    return out


def _brier_score(confidences_pct: np.ndarray, wins: np.ndarray) -> float:
    """Mean squared error between confidence (as probability) and outcome."""
    p = np.clip(confidences_pct / 100.0, 0.0, 1.0)
    y = wins.astype(float)
    return float(np.mean((p - y) ** 2))


def _build_bins(
    samples: Iterable[CalibrationSample],
    n_bins: int,
    prior_n: float,
    prior_p: float,
) -> tuple[list[_BinFit], dict[str, float]]:
    """Bins + PAV; returns (bins, metrics) where metrics include sample counts and brier."""
    confs = np.array([float(s.confidence) for s in samples], dtype=float)
    wins = np.array([1.0 if s.win else 0.0 for s in samples], dtype=float)
    n = len(confs)

    edges = np.linspace(0.0, 100.0, n_bins + 1)

    # Pre-seed every bin with the prior so empty-input / sparse bins fall back
    # to the user-supplied default (0.5 by default) instead of a hard zero.
    bin_p_raw = np.full(n_bins, prior_p, dtype=float)
    bin_n = np.zeros(n_bins, dtype=float)

    if n > 0:
        bin_idx = np.digitize(confs, edges[1:-1])
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        for i in range(n_bins):
            mask = bin_idx == i
            count = int(mask.sum())
            bin_n[i] = count
            if count > 0:
                # Beta-style smoothing toward the prior so a bin with 1
                # sample doesn't claim 100% / 0% certainty.
                bin_p_raw[i] = (wins[mask].sum() + prior_n * prior_p) / (count + prior_n)
            else:
                bin_p_raw[i] = prior_p

    # Effective weights = N + prior so empty bins still influence neighbours
    bin_p_iso = _pav_isotonic(bin_p_raw, bin_n + prior_n)

    bins: list[_BinFit] = []
    for i in range(n_bins):
        bins.append(
            _BinFit(
                lower=float(edges[i]),
                upper=float(edges[i + 1]),
                calibrated=float(bin_p_iso[i] * 100.0),
                raw_win_rate=float(bin_p_raw[i] * 100.0),
                n_samples=int(bin_n[i]),
            )
        )

    metrics: dict[str, float] = {"n_samples": float(n)}
    if n > 0:
        # Apply the freshly-fit bins back to the training data to estimate
        # the post-calibration Brier improvement (cheap sanity metric).
        calibrated_pct = np.zeros(n, dtype=float)
        for j, c in enumerate(confs):
            calibrated_pct[j] = _lookup_calibrated(bins, float(c))
        metrics["brier_before"] = _brier_score(confs, wins)
        metrics["brier_after"] = _brier_score(calibrated_pct, wins)
        metrics["mean_raw_confidence"] = float(np.mean(confs))
        metrics["win_rate"] = float(np.mean(wins) * 100.0)
    return bins, metrics


def _lookup_calibrated(bins: list[_BinFit], raw_pct: float) -> float:
    """Piecewise-constant lookup. Returns raw_pct unchanged when no bin matches."""
    if not bins:
        return raw_pct
    raw = max(0.0, min(100.0, float(raw_pct)))
    for b in bins:
        if b.lower <= raw < b.upper:
            return b.calibrated
    # Last bin is inclusive on the upper edge so a raw==100 still maps.
    if abs(raw - bins[-1].upper) < 1e-9:
        return bins[-1].calibrated
    return raw_pct


def _key_str(key: KeyType) -> str:
    """JSON-friendly key for status payload and persistence."""
    return f"{key[0]}|{key[1]}"


def _key_from_str(s: str) -> KeyType:
    parts = s.split("|", 1)
    if len(parts) != 2:
        return (WILDCARD, WILDCARD)
    return (parts[0] or WILDCARD, parts[1] or WILDCARD)


# ── The store itself ────────────────────────────────────────────────────────


class CalibrationStore:
    """
    Thread-safe singleton holding the active per-(direction, horizon) maps.

    Construction is private — use :func:`get_calibration_store`.
    """

    _DEFAULT_FILENAME = "calibration_state.json"
    _PERSIST_VERSION = 2  # bumped from 1 (single-bins format) for Phase 4.5

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # key = (direction_norm, horizon_norm); value = list[_BinFit]
        self._maps: dict[KeyType, list[_BinFit]] = {}
        # key = same; value = per-key metadata (n_samples, brier_*, etc.)
        self._maps_meta: dict[KeyType, dict] = {}
        # store-wide metadata (active flag, fitted_at, etc.)
        self._meta: dict = {}
        self._persist_path: Optional[Path] = None
        self._configure_persistence()
        self._load_from_disk()

    # ── config ────────────────────────────────────────────────────────────

    def _configure_persistence(self) -> None:
        try:
            base = Path(getattr(settings, "model_dir", "./models"))
            base.mkdir(parents=True, exist_ok=True)
            self._persist_path = base / self._DEFAULT_FILENAME
        except Exception as e:  # noqa: BLE001 — best-effort
            logger.warning("Calibration: cannot prepare persist dir: {}", e)
            self._persist_path = None

    @staticmethod
    def _settings():
        n_bins = int(getattr(settings, "calibration_n_bins", 10) or 10)
        min_samples = int(getattr(settings, "calibration_min_samples", 50) or 50)
        prior_n = float(getattr(settings, "calibration_prior_strength", 4.0) or 4.0)
        prior_p = float(getattr(settings, "calibration_prior_win_rate", 0.5) or 0.5)
        return n_bins, min_samples, prior_n, prior_p

    # ── public API ────────────────────────────────────────────────────────

    def is_active(self) -> bool:
        """True if any (direction, horizon) map has been fitted."""
        with self._lock:
            return bool(self._maps) and bool(self._meta.get("active", False))

    def apply(
        self,
        raw_confidence: float,
        *,
        direction: Optional[str] = None,
        horizon: Optional[str] = None,
    ) -> float:
        """Returns calibrated confidence on a 0–100 scale.

        Walks the fallback chain (most specific → global). If no map qualifies
        the raw value passes through unchanged. Fully back-compat: omitting
        ``direction`` and ``horizon`` causes a direct lookup against the
        global ``("*", "*")`` map, identical to Phase 4.3 behaviour.
        """
        with self._lock:
            if not self._maps or not self._meta.get("active", False):
                return float(raw_confidence)
            for key in self._lookup_chain(direction, horizon):
                bins = self._maps.get(key)
                if bins:
                    return _lookup_calibrated(bins, float(raw_confidence))
        return float(raw_confidence)

    def apply_with_key(
        self,
        raw_confidence: float,
        *,
        direction: Optional[str] = None,
        horizon: Optional[str] = None,
    ) -> tuple[float, Optional[KeyType]]:
        """Like :meth:`apply` but also returns which key was used.

        Useful for logging / monitoring so the operator can see whether a
        given prediction fell back to the global map or hit the specific
        ``("BUY", "15M")`` calibration. Returns ``(value, None)`` when no
        map qualified and the raw value passed through.
        """
        with self._lock:
            if not self._maps or not self._meta.get("active", False):
                return float(raw_confidence), None
            for key in self._lookup_chain(direction, horizon):
                bins = self._maps.get(key)
                if bins:
                    return _lookup_calibrated(bins, float(raw_confidence)), key
        return float(raw_confidence), None

    @staticmethod
    def _lookup_chain(direction: Optional[str], horizon: Optional[str]) -> List[KeyType]:
        """Most-specific → global. Duplicates de-duped while preserving order."""
        d = _normalize_direction(direction)
        h = _normalize_horizon(horizon)
        chain: List[KeyType] = []
        seen: set[KeyType] = set()

        for k in [(d, h), (d, WILDCARD), (WILDCARD, h), (WILDCARD, WILDCARD)]:
            if k not in seen:
                chain.append(k)
                seen.add(k)
        return chain

    def status(self) -> dict:
        """Returns a JSON-serialisable snapshot of every fitted map."""
        with self._lock:
            maps_payload: dict[str, dict] = {}
            for key, bins in self._maps.items():
                meta = self._maps_meta.get(key, {})
                maps_payload[_key_str(key)] = {
                    "direction": key[0],
                    "horizon": key[1],
                    "n_samples": int(meta.get("n_samples", 0)),
                    "brier_before": meta.get("brier_before"),
                    "brier_after": meta.get("brier_after"),
                    "mean_raw_confidence": meta.get("mean_raw_confidence"),
                    "win_rate": meta.get("win_rate"),
                    "bins": [
                        {
                            "lower": b.lower,
                            "upper": b.upper,
                            "calibrated": round(b.calibrated, 2),
                            "raw_win_rate": round(b.raw_win_rate, 2),
                            "n_samples": b.n_samples,
                        }
                        for b in bins
                    ],
                }

            global_meta = self._maps_meta.get((WILDCARD, WILDCARD), {})
            return {
                "active": bool(self._meta.get("active", False)),
                "fitted_at_unix_ms": self._meta.get("fitted_at_unix_ms"),
                "n_samples": int(self._meta.get("n_samples", 0)),
                "n_maps": len(self._maps),
                "horizon_filter": self._meta.get("horizon_filter"),
                "maps": maps_payload,
                # Phase 4.3 back-compat: surface the global map's bins / metrics
                # at the top level so dashboards built before 4.5 keep working.
                "bins": [
                    {
                        "lower": b.lower,
                        "upper": b.upper,
                        "calibrated": round(b.calibrated, 2),
                        "raw_win_rate": round(b.raw_win_rate, 2),
                        "n_samples": b.n_samples,
                    }
                    for b in self._maps.get((WILDCARD, WILDCARD), [])
                ],
                "metrics": {
                    "brier_before": global_meta.get("brier_before"),
                    "brier_after": global_meta.get("brier_after"),
                    "mean_raw_confidence": global_meta.get("mean_raw_confidence"),
                    "win_rate": global_meta.get("win_rate"),
                },
            }

    def fit(
        self,
        samples: list[CalibrationSample],
        *,
        horizon_filter: Optional[str] = None,
    ) -> dict:
        """
        Fit calibration maps from resolved-prediction samples.

        Builds, in order:

        * Per-(direction, horizon) maps for every concrete ``(BUY/SELL, hh)``
          pair that has ≥ ``calibration_min_samples`` samples.
        * Per-direction maps (``(direction, "*")``) collapsing horizons.
        * Per-horizon maps (``("*", horizon)``) collapsing directions.
        * The global ``("*", "*")`` map (always built when total ≥ min_samples).

        ``samples`` should already be filtered to directional (BUY/SELL) rows
        whose outcome is known. Cold-start: if fewer than
        ``calibration_min_samples`` are provided in total, the store is left
        inactive and the call returns ``{"active": false, "reason": "insufficient samples"}``.
        """
        n_bins, min_samples, prior_n, prior_p = self._settings()

        cleaned: list[CalibrationSample] = []
        for s in samples:
            try:
                conf = float(s.confidence)
            except (TypeError, ValueError):
                continue
            if not (0.0 <= conf <= 100.0):
                continue
            cleaned.append(CalibrationSample(
                confidence=conf,
                win=bool(s.win),
                direction=s.direction,
                horizon=s.horizon,
            ))

        n = len(cleaned)
        if n < min_samples:
            with self._lock:
                self._meta = {
                    **self._meta,
                    "active": False,
                    "last_attempt_at_unix_ms": int(time.time() * 1000),
                    "last_attempt_n_samples": n,
                    "last_attempt_min_samples": min_samples,
                }
            logger.warning(
                "Calibration fit skipped: n={} < min_samples={}", n, min_samples
            )
            return {
                "active": False,
                "reason": "insufficient_samples",
                "n_samples": n,
                "min_samples": min_samples,
            }

        # Bucketise: group samples by all four key shapes so we can fit each
        # in one pass without re-walking the full sample list multiple times.
        # The "*" wildcard buckets receive every sample regardless of its
        # direction / horizon.
        buckets: dict[KeyType, list[CalibrationSample]] = {(WILDCARD, WILDCARD): list(cleaned)}
        for s in cleaned:
            d = _normalize_direction(s.direction)
            h = _normalize_horizon(s.horizon)
            if d != WILDCARD or h != WILDCARD:
                # Only build specific keys when the sample carries some labelling.
                buckets.setdefault((d, h), []).append(s)
            if d != WILDCARD:
                buckets.setdefault((d, WILDCARD), []).append(s)
            if h != WILDCARD:
                buckets.setdefault((WILDCARD, h), []).append(s)

        new_maps: dict[KeyType, list[_BinFit]] = {}
        new_meta: dict[KeyType, dict] = {}
        skipped: list[dict] = []
        for key, bucket in buckets.items():
            if len(bucket) < min_samples:
                skipped.append({"key": _key_str(key), "n_samples": len(bucket), "min_samples": min_samples})
                continue
            bins, metrics = _build_bins(bucket, n_bins, prior_n, prior_p)
            new_maps[key] = bins
            new_meta[key] = metrics

        # If even the global bucket failed, treat it as cold-start.
        if (WILDCARD, WILDCARD) not in new_maps:
            with self._lock:
                self._meta = {
                    **self._meta,
                    "active": False,
                    "last_attempt_at_unix_ms": int(time.time() * 1000),
                    "last_attempt_n_samples": n,
                    "last_attempt_min_samples": min_samples,
                }
            return {
                "active": False,
                "reason": "global_bucket_below_min_samples",
                "n_samples": n,
                "min_samples": min_samples,
                "skipped": skipped,
            }

        global_metrics = new_meta[(WILDCARD, WILDCARD)]

        with self._lock:
            self._maps = new_maps
            self._maps_meta = new_meta
            self._meta = {
                "active": True,
                "fitted_at_unix_ms": int(time.time() * 1000),
                "n_samples": n,
                "horizon_filter": horizon_filter,
                **global_metrics,  # surface global brier_* at top-level for back-compat
            }
        self._save_to_disk()

        improvement = global_metrics.get("brier_before", 0) - global_metrics.get("brier_after", 0)
        logger.info(
            "Calibration fitted: total_n={} maps={} brier_before={:.4f} brier_after={:.4f} "
            "improvement={:.4f} bins={}",
            n,
            len(new_maps),
            global_metrics.get("brier_before", 0),
            global_metrics.get("brier_after", 0),
            improvement,
            n_bins,
        )

        return {
            "active": True,
            "fitted_at_unix_ms": self._meta["fitted_at_unix_ms"],
            "n_samples": n,
            "n_bins": n_bins,
            "n_maps": len(new_maps),
            "maps": [
                {
                    "key": _key_str(k),
                    "direction": k[0],
                    "horizon": k[1],
                    "n_samples": int(new_meta[k].get("n_samples", 0)),
                    "brier_before": new_meta[k].get("brier_before"),
                    "brier_after": new_meta[k].get("brier_after"),
                }
                for k in new_maps
            ],
            "skipped": skipped,
            "metrics": global_metrics,  # Phase 4.3 back-compat
            "horizon_filter": horizon_filter,
        }

    def clear(self) -> None:
        with self._lock:
            self._maps = {}
            self._maps_meta = {}
            self._meta = {"active": False, "cleared_at_unix_ms": int(time.time() * 1000)}
        self._save_to_disk()
        logger.info("Calibration cleared; raw confidence will pass through.")

    # ── persistence ──────────────────────────────────────────────────────

    def _save_to_disk(self) -> None:
        if self._persist_path is None:
            return
        try:
            payload = {
                "version": self._PERSIST_VERSION,
                "meta": self._meta,
                "maps": [
                    {
                        "key": _key_str(key),
                        "direction": key[0],
                        "horizon": key[1],
                        "meta": self._maps_meta.get(key, {}),
                        "bins": [b.__dict__ for b in bins],
                    }
                    for key, bins in self._maps.items()
                ],
            }
            self._persist_path.write_text(json.dumps(payload, indent=2))
        except Exception as e:  # noqa: BLE001
            logger.warning("Calibration: persist failed: {}", e)

    def _load_from_disk(self) -> None:
        if self._persist_path is None or not self._persist_path.exists():
            return
        try:
            payload = json.loads(self._persist_path.read_text())
            with self._lock:
                self._meta = dict(payload.get("meta") or {})
                version = int(payload.get("version") or 1)
                if version >= 2 and isinstance(payload.get("maps"), list):
                    # New format: list of per-key entries.
                    self._maps = {}
                    self._maps_meta = {}
                    for entry in payload.get("maps", []):
                        key = _key_from_str(str(entry.get("key", f"{WILDCARD}|{WILDCARD}")))
                        self._maps[key] = [
                            _BinFit(
                                lower=float(b["lower"]),
                                upper=float(b["upper"]),
                                calibrated=float(b["calibrated"]),
                                raw_win_rate=float(b.get("raw_win_rate", b["calibrated"])),
                                n_samples=int(b.get("n_samples", 0)),
                            )
                            for b in (entry.get("bins") or [])
                        ]
                        self._maps_meta[key] = dict(entry.get("meta") or {})
                else:
                    # Phase 4.3 (v1) format: a flat ``bins`` list. Auto-migrate
                    # into the global ``("*", "*")`` slot so existing on-disk
                    # state survives the upgrade with zero operator action.
                    legacy_bins = [
                        _BinFit(
                            lower=float(b["lower"]),
                            upper=float(b["upper"]),
                            calibrated=float(b["calibrated"]),
                            raw_win_rate=float(b.get("raw_win_rate", b["calibrated"])),
                            n_samples=int(b.get("n_samples", 0)),
                        )
                        for b in (payload.get("bins") or [])
                    ]
                    if legacy_bins:
                        self._maps = {(WILDCARD, WILDCARD): legacy_bins}
                        self._maps_meta = {
                            (WILDCARD, WILDCARD): {
                                k: v
                                for k, v in self._meta.items()
                                if k.startswith("brier_") or k in ("mean_raw_confidence", "win_rate", "n_samples")
                            }
                        }
                    else:
                        self._maps = {}
                        self._maps_meta = {}

            if self._maps and self._meta.get("active"):
                logger.info(
                    "Calibration loaded from disk: maps={} n_samples={}",
                    len(self._maps),
                    self._meta.get("n_samples"),
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("Calibration: load failed: {}", e)


_INSTANCE: Optional[CalibrationStore] = None
_INSTANCE_LOCK = threading.Lock()


def get_calibration_store() -> CalibrationStore:
    """Return the process-wide :class:`CalibrationStore`. Creates it lazily."""
    global _INSTANCE
    if _INSTANCE is None:
        with _INSTANCE_LOCK:
            if _INSTANCE is None:
                _INSTANCE = CalibrationStore()
    return _INSTANCE


def reset_calibration_store_for_tests() -> None:
    """Test helper: drop the singleton so tests start with a clean state."""
    global _INSTANCE
    with _INSTANCE_LOCK:
        _INSTANCE = None
