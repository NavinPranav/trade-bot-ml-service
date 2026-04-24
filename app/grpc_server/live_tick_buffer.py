"""
Buffer for real-time ticks streamed from the Java backend.

Responsibilities:
  - Stores the latest tick per instrument (by symbol or token key).
  - Maintains a copy of the most recent OHLCV baseline from GetPrediction.
  - Merges live ticks into the last OHLCV bar (updates close/high/low/volume)
    so the predictor sees the freshest data.
  - Provides debounce-aware dirty tracking so StreamLiveTicks can trigger
    re-prediction only when enough time has passed since the last run.

Angel One may stream many instruments (each user's underlying, INDIA VIX, etc.).
Live re-prediction must only run for ticks that match the baseline underlying
from the last GetGeminiPrediction / GetPrediction — otherwise "current" price
would jump between unrelated instruments.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd
from loguru import logger

from app.config import settings


def live_tick_routing_key(tick) -> str:
    """Stable key for stream routing: prefer symbol, else token (matches Java LiveTick)."""
    sym = (getattr(tick, "symbol", "") or "").strip()
    tok = (getattr(tick, "token", "") or "").strip()
    return (sym or tok).upper()


@dataclass
class TickSnapshot:
    symbol: str
    exchange_type: int
    token: str
    ltp: float
    open: float
    high: float
    low: float
    close: float
    change: float
    change_pct: float
    volume: int
    timestamp_ms: int


class LiveTickBuffer:
    """Thread-safe tick buffer shared between StreamLiveTicks and GetPrediction."""

    def __init__(self, debounce_seconds: float = 5.0):
        self._lock = threading.Lock()
        self._ticks: Dict[str, TickSnapshot] = {}
        self._baseline_ohlcv: Optional[pd.DataFrame] = None
        self._baseline_vix: Optional[pd.DataFrame] = None
        self._baseline_horizon: str = ""
        self._baseline_engine: str = "ML"
        self._baseline_underlying: str = ""
        self._baseline_instrument_token: str = ""
        self._last_prediction_time: float = 0.0
        self._last_prediction_result: Optional[Dict[str, Any]] = None
        self._dirty = False
        self._debounce_seconds = debounce_seconds
        self._repredict_running = False

    def update_tick(self, tick) -> None:
        """Store the latest tick (from a gRPC LiveTick message).

        Field names match the 12-field LiveTick proto: open, high, low, close,
        change, change_pct, volume, timestamp_unix_ms."""
        key = live_tick_routing_key(tick)
        if not key:
            return

        ltp = float(getattr(tick, "last_traded_price", 0.0) or 0.0)
        open_p = float(getattr(tick, "open", 0.0) or 0.0)
        high_p = float(getattr(tick, "high", 0.0) or 0.0)
        low_p = float(getattr(tick, "low", 0.0) or 0.0)
        close_p = float(getattr(tick, "close", 0.0) or 0.0)
        chg = float(getattr(tick, "change", 0.0) or 0.0)
        chg_pct = float(getattr(tick, "change_pct", 0.0) or 0.0)
        vol = int(getattr(tick, "volume", 0) or 0)
        ts_ms = int(getattr(tick, "timestamp_unix_ms", 0) or 0)
        sym_raw = (getattr(tick, "symbol", "") or "").strip()
        tok_raw = (getattr(tick, "token", "") or "").strip()

        snap = TickSnapshot(
            symbol=sym_raw or key,
            exchange_type=int(getattr(tick, "exchange_type", 0) or 0),
            token=tok_raw,
            ltp=ltp,
            open=open_p,
            high=high_p,
            low=low_p,
            close=close_p,
            change=chg,
            change_pct=chg_pct,
            volume=vol,
            timestamp_ms=ts_ms,
        )
        keys: set[str] = {key}
        tok_u = tok_raw.upper()
        sym_u = sym_raw.upper()
        if tok_u:
            keys.add(tok_u)
        if sym_u:
            keys.add(sym_u)
        with self._lock:
            for k in keys:
                if k:
                    self._ticks[k] = snap
            self._dirty = True

    def get_latest_tick(self, symbol: str) -> Optional[TickSnapshot]:
        with self._lock:
            return self._ticks.get((symbol or "").strip().upper())

    def store_baseline(
        self,
        horizon: str,
        ohlcv: pd.DataFrame,
        vix: Optional[pd.DataFrame],
        engine: str = "ML",
        underlying_symbol: str = "",
        instrument_token: str = "",
    ) -> None:
        """Called from GetPrediction/GetGeminiPrediction to snapshot historical data for live merging."""
        with self._lock:
            self._baseline_ohlcv = ohlcv.copy() if ohlcv is not None else None
            self._baseline_vix = vix.copy() if vix is not None else None
            self._baseline_horizon = horizon
            self._baseline_engine = engine
            self._baseline_underlying = (underlying_symbol or "").strip()
            self._baseline_instrument_token = (instrument_token or "").strip()
            self._dirty = True
            logger.info(
                f"Baseline stored: engine={engine} horizon={horizon} "
                f"underlying={self._baseline_underlying!r} token={self._baseline_instrument_token!r} "
                f"bars={len(ohlcv) if ohlcv is not None else 0} "
                f"vix={'yes' if vix is not None and not vix.empty else 'no'}"
            )

    def tick_matches_baseline(self, route_key: str) -> bool:
        """True if this tick's routing key is the same instrument as the stored OHLCV baseline."""
        rk = (route_key or "").strip().upper()
        if not rk:
            return False
        sym = self._baseline_underlying.strip().upper()
        tok = self._baseline_instrument_token.strip().upper()
        if not sym and not tok:
            return False
        if tok and rk == tok:
            return True
        if sym and rk == sym:
            return True
        return False

    def get_baseline_underlying(self) -> str:
        with self._lock:
            return self._baseline_underlying

    def get_baseline_tick(self) -> Optional[TickSnapshot]:
        """Latest tick for the baseline instrument (token or name), if any."""
        with self._lock:
            tok = self._baseline_instrument_token.strip().upper()
            if tok:
                hit = self._ticks.get(tok)
                if hit is not None and hit.ltp > 0:
                    return hit
            sym = self._baseline_underlying.strip().upper()
            if sym:
                hit = self._ticks.get(sym)
                if hit is not None and hit.ltp > 0:
                    return hit
            return None

    def get_merged_ohlcv(self) -> Optional[pd.DataFrame]:
        """Returns the baseline OHLCV with the last row updated from the live tick for the baseline instrument."""
        with self._lock:
            if self._baseline_ohlcv is None or self._baseline_ohlcv.empty:
                return None
            tick = None
            tok = self._baseline_instrument_token.strip().upper()
            if tok:
                tick = self._ticks.get(tok)
            if tick is None or tick.ltp <= 0:
                sym = self._baseline_underlying.strip().upper()
                if sym:
                    tick = self._ticks.get(sym)
            if tick is None or tick.ltp <= 0:
                return self._baseline_ohlcv.copy()

            merged = self._baseline_ohlcv.copy()
            last_idx = merged.index[-1]
            merged.loc[last_idx, "close"] = tick.ltp
            merged.loc[last_idx, "high"] = max(merged.loc[last_idx, "high"], tick.high)
            merged.loc[last_idx, "low"] = min(merged.loc[last_idx, "low"], tick.low)
            if tick.volume > 0:
                merged.loc[last_idx, "volume"] = tick.volume
            return merged

    def get_baseline_vix(self) -> Optional[pd.DataFrame]:
        with self._lock:
            return self._baseline_vix.copy() if self._baseline_vix is not None else None

    def get_baseline_horizon(self) -> str:
        with self._lock:
            return self._baseline_horizon

    def get_baseline_engine(self) -> str:
        with self._lock:
            return self._baseline_engine

    def should_repredict(self) -> bool:
        """True if baseline is loaded, ticks changed since last prediction,
        the debounce window elapsed, and no re-prediction is currently running."""
        with self._lock:
            if not self._dirty:
                return False
            if self._baseline_ohlcv is None:
                return False
            if self._repredict_running:
                return False
            elapsed = time.monotonic() - self._last_prediction_time
            return elapsed >= self._debounce_seconds

    def start_repredict(self) -> bool:
        """Atomically claim the re-predict slot. Returns True if claimed."""
        with self._lock:
            if self._repredict_running:
                return False
            self._repredict_running = True
            return True

    def mark_predicted(self, result: Dict[str, Any]) -> None:
        with self._lock:
            self._last_prediction_time = time.monotonic()
            self._last_prediction_result = result
            self._dirty = False
            self._repredict_running = False

    def get_cached_live_prediction(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._last_prediction_result

    def has_baseline(self) -> bool:
        with self._lock:
            return self._baseline_ohlcv is not None and not self._baseline_ohlcv.empty


# Singleton shared across gRPC handlers
_buffer: Optional[LiveTickBuffer] = None
_buffer_lock = threading.Lock()


def get_live_tick_buffer() -> LiveTickBuffer:
    global _buffer
    if _buffer is None:
        with _buffer_lock:
            if _buffer is None:
                debounce = getattr(settings, "live_inference_interval_sec", 5.0)
                _buffer = LiveTickBuffer(debounce_seconds=debounce)
                logger.info(f"LiveTickBuffer initialized (debounce={debounce}s)")
    return _buffer
