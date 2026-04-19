"""Stable fingerprint for OHLCV windows (used by Redis prediction cache)."""
import hashlib

import pandas as pd


def ohlcv_cache_fingerprint(ohlcv: pd.DataFrame) -> str:
    if ohlcv.empty:
        return "empty"
    row = ohlcv.iloc[-1]
    row0 = ohlcv.iloc[0]
    idx = ohlcv.index[-1]
    idx0 = ohlcv.index[0]
    o0 = float(row0["open"]) if "open" in ohlcv.columns else float(row0["close"])
    payload = (f"{idx0}|{o0}|{idx}|{float(row['close'])}|{len(ohlcv)}").encode()
    return hashlib.sha256(payload).hexdigest()[:16]
