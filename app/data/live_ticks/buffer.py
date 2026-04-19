"""LiveTickBuffer: stores streamed ticks (Angel One LiveTickData shape) and merges into baseline last bar."""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from app.config import settings


def _symbol_key(s: str) -> str:
    return "".join((s or "").split()).upper()


def _ascii_symbol(s: str) -> bool:
    """Reject mojibake / binary mistaken for symbol when proto fields misalign."""
    if not s or len(s) > 96:
        return False
    return all(32 <= ord(c) < 127 for c in s)


def _tick_routing_key(tick) -> str:
    """Route ticks to baseline: prefer Angel numeric token; symbol only if ASCII-safe.

    If Java/protobuf mis-decodes field 16, ``symbol`` can be garbage bytes — do not prefer it
    over ``token`` when token is a normal numeric instrument id.
    """
    sym = str(getattr(tick, "symbol", "") or "").strip()
    tok = str(getattr(tick, "token", "") or "").strip()
    if tok.isdigit():
        return _symbol_key(tok)
    if sym and _ascii_symbol(sym):
        return _symbol_key(sym)
    if tok:
        return _symbol_key(tok)
    return ""


def live_tick_proto_looks_valid(tick) -> bool:
    """Sanity check after proto matches Java LiveTickData."""
    if not str(getattr(tick, "token", "") or "").strip():
        return False
    try:
        ltp = float(getattr(tick, "last_traded_price", 0.0) or 0.0)
    except (TypeError, ValueError):
        return False
    if ltp < 0 or ltp > 1e8:
        return False
    return True


@dataclass
class TickSnapshot:
    token: str
    symbol: str
    subscription_mode: int
    exchange_type: int
    sequence_number: int
    exchange_timestamp_ms: int
    ltp: float
    last_traded_quantity: int
    average_traded_price: float
    volume_traded: int
    total_buy_quantity: float
    total_sell_quantity: float
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    change: float
    change_pct: float


def _snapshot_from_proto(tick) -> TickSnapshot:
    close_price = float(getattr(tick, "close_price", 0.0) or 0.0)
    ltp = float(getattr(tick, "last_traded_price", 0.0) or 0.0)
    ch = ltp - close_price if close_price > 0 else 0.0
    pct = (ltp - close_price) / close_price * 100.0 if close_price > 0 else 0.0
    return TickSnapshot(
        token=str(getattr(tick, "token", "") or ""),
        symbol=str(getattr(tick, "symbol", "") or ""),
        subscription_mode=int(getattr(tick, "subscription_mode", 0) or 0),
        exchange_type=int(getattr(tick, "exchange_type", 0) or 0),
        sequence_number=int(getattr(tick, "sequence_number", 0) or 0),
        exchange_timestamp_ms=int(getattr(tick, "exchange_timestamp_ms", 0) or 0),
        ltp=ltp,
        last_traded_quantity=int(getattr(tick, "last_traded_quantity", 0) or 0),
        average_traded_price=float(getattr(tick, "average_traded_price", 0.0) or 0.0),
        volume_traded=int(getattr(tick, "volume_traded", 0) or 0),
        total_buy_quantity=float(getattr(tick, "total_buy_quantity", 0.0) or 0.0),
        total_sell_quantity=float(getattr(tick, "total_sell_quantity", 0.0) or 0.0),
        open_price=float(getattr(tick, "open_price", 0.0) or 0.0),
        high_price=float(getattr(tick, "high_price", 0.0) or 0.0),
        low_price=float(getattr(tick, "low_price", 0.0) or 0.0),
        close_price=close_price,
        change=ch,
        change_pct=pct,
    )


class LiveTickBuffer:
    """
    - Ring buffer of ticks per routing key (token or symbol).
    - Baseline OHLCV from GetPrediction; lookup_key = instrument_token or underlying_symbol.
    """

    def __init__(self, debounce_seconds: float = 30.0, max_ticks_per_symbol: int = 5000) -> None:
        self._lock = threading.Lock()
        self._tick_history: Dict[str, deque] = {}
        self._latest: Dict[str, TickSnapshot] = {}
        self._baseline_ohlcv: Optional[pd.DataFrame] = None
        self._baseline_vix: Optional[pd.DataFrame] = None
        self._baseline_horizon: str = "1D"
        self._merge_lookup_key: str = ""
        self._debounce_seconds = debounce_seconds
        self._max_ticks = max_ticks_per_symbol
        self._dirty: Dict[str, bool] = {}
        self._last_predict_mono: Dict[str, float] = {}
        self._last_prediction_result: Optional[Dict[str, Any]] = None
        self._warned_stream_without_baseline = False

    def maybe_warn_stream_without_baseline(self) -> None:
        with self._lock:
            if self._baseline_ohlcv is not None:
                return
            if self._warned_stream_without_baseline:
                return
            self._warned_stream_without_baseline = True
        logger.warning(
            "StreamLiveTicks: ticks are arriving, but no baseline OHLCV is loaded yet — "
            "call GetPrediction (with enough sensex_ohlcv bars) first. "
            "Send instrument_token on GetPrediction matching LiveTick.token. "
            "Until then, [LIVE RE-PREDICT] will not run."
        )

    def record_tick(self, tick) -> str:
        snap = _snapshot_from_proto(tick)
        key = _tick_routing_key(tick)
        if not key:
            return ""
        with self._lock:
            dq = self._tick_history.setdefault(key, deque(maxlen=self._max_ticks))
            dq.append(snap)
            self._latest[key] = snap
            self._dirty[key] = True
            if self._baseline_ohlcv is not None and key not in self._last_predict_mono:
                self._last_predict_mono[key] = time.monotonic()
        return key

    def store_baseline(
        self,
        horizon: str,
        ohlcv: pd.DataFrame,
        vix: Optional[pd.DataFrame],
        lookup_key: str,
    ) -> None:
        """lookup_key = instrument_token or underlying_symbol (same convention as tick routing)."""
        with self._lock:
            self._baseline_ohlcv = ohlcv.copy() if ohlcv is not None and not ohlcv.empty else None
            self._baseline_vix = vix.copy() if vix is not None and not vix.empty else None
            self._baseline_horizon = horizon or "1D"
            self._merge_lookup_key = _symbol_key(lookup_key)
            self._tick_history.clear()
            self._latest.clear()
            self._dirty.clear()
            self._last_predict_mono.clear()
        logger.info("LiveTickBuffer: baseline replaced lookup_key={!r}", self._merge_lookup_key)

    def merge_ticks_into_ohlcv(self, ohlcv: pd.DataFrame, lookup_key: str) -> pd.DataFrame:
        """
        Merge buffered ticks for lookup_key into a copy of ohlcv's last row.
        lookup_key should match GetPrediction instrument_token or underlying_symbol (token preferred).
        """
        key = _symbol_key(lookup_key)
        if ohlcv is None or ohlcv.empty:
            return ohlcv.copy() if ohlcv is not None else pd.DataFrame()

        with self._lock:
            ticks = list(self._tick_history.get(key, ()))
            latest = self._latest.get(key)

        seq: List[TickSnapshot] = ticks if ticks else ([latest] if latest else [])
        if not seq:
            return ohlcv.copy()

        merged = ohlcv.copy()
        last_idx = merged.index[-1]
        prev_hi = float(merged.at[last_idx, "high"])
        prev_lo = float(merged.at[last_idx, "low"])
        o0 = float(merged.at[last_idx, "open"])
        vol0 = float(merged.at[last_idx, "volume"])

        ltps = [s.ltp for s in seq if s.ltp > 0]
        highs = [s.high_price for s in seq if s.high_price > 0]
        lows = [s.low_price for s in seq if s.low_price > 0]
        close = ltps[-1] if ltps else float(merged.at[last_idx, "close"])
        hi = max([prev_hi] + highs + ltps)
        if prev_lo > 0:
            lo = min([prev_lo] + lows + ltps) if (lows or ltps) else min(prev_lo, close)
        else:
            lo = min(lows + ltps) if (lows or ltps) else float(merged.at[last_idx, "low"])
        vol_extra = sum(float(s.volume_traded) for s in seq if s.volume_traded and s.volume_traded > 0)

        merged.at[last_idx, "open"] = o0
        merged.at[last_idx, "high"] = hi
        merged.at[last_idx, "low"] = lo
        merged.at[last_idx, "close"] = close
        merged.at[last_idx, "volume"] = vol0 + vol_extra
        return merged

    def get_merged_ohlcv(self, tick_routing_key: str) -> Optional[pd.DataFrame]:
        with self._lock:
            if self._baseline_ohlcv is None or self._baseline_ohlcv.empty:
                return None
            base = self._baseline_ohlcv
        return self.merge_ticks_into_ohlcv(base, tick_routing_key)

    def get_baseline_vix(self) -> Optional[pd.DataFrame]:
        with self._lock:
            return self._baseline_vix.copy() if self._baseline_vix is not None else None

    def get_baseline_horizon(self) -> str:
        with self._lock:
            return self._baseline_horizon

    def get_latest_tick(self, tick_routing_key: str) -> Optional[TickSnapshot]:
        key = _symbol_key(tick_routing_key)
        with self._lock:
            return self._latest.get(key)

    def should_repredict_symbol(self) -> Optional[str]:
        if not getattr(settings, "live_inference_enabled", True):
            return None
        now = time.monotonic()
        with self._lock:
            if self._baseline_ohlcv is None or self._baseline_ohlcv.empty:
                return None
            for key, is_dirty in list(self._dirty.items()):
                if not is_dirty:
                    continue
                last = self._last_predict_mono.get(key, 0.0)
                if now - last >= self._debounce_seconds:
                    return key
        return None

    def mark_predicted(self, symbol_key: str, result: Dict[str, Any]) -> None:
        with self._lock:
            self._last_predict_mono[symbol_key] = time.monotonic()
            self._dirty[symbol_key] = False
            self._last_prediction_result = result

    def get_cached_live_prediction(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._last_prediction_result

    def has_baseline(self) -> bool:
        with self._lock:
            return self._baseline_ohlcv is not None and not self._baseline_ohlcv.empty

    def has_buffered_ticks(self, lookup_key: str) -> bool:
        key = _symbol_key(lookup_key)
        with self._lock:
            if key in self._tick_history and len(self._tick_history[key]) > 0:
                return True
            return key in self._latest


_buffer: Optional[LiveTickBuffer] = None
_buffer_lock = threading.Lock()


def get_live_tick_buffer() -> LiveTickBuffer:
    global _buffer
    if _buffer is None:
        with _buffer_lock:
            if _buffer is None:
                debounce = float(getattr(settings, "live_inference_interval_sec", 30.0))
                _buffer = LiveTickBuffer(debounce_seconds=debounce)
                logger.info("LiveTickBuffer ready (debounce={}s)", debounce)
    return _buffer
