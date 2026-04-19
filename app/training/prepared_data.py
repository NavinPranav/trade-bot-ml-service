"""
Download and persist OHLCV + India VIX for ModelTrainer.train_all.

Uses yfinance for local/offline dataset builds only (not used in production gRPC path).
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from loguru import logger

from app.config import settings

DEFAULT_DATA_DIR = Path("data/training")

_YF_HINT = (
    "Yahoo/yfinance often fails with JSON/SSL on macOS (LibreSSL + urllib3 v2). "
    "Try: pip install 'urllib3>=1.26.18,<2' && pip install -U yfinance certifi "
    "— or place sensex_ohlcv.csv / india_vix.csv in data/training/ from your broker."
)


def _flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or not isinstance(df.columns, pd.MultiIndex):
        return df
    out = df.copy()
    out.columns = out.columns.get_level_values(0)
    return out


def _configure_ssl_bundle() -> None:
    try:
        import os

        import certifi

        bundle = certifi.where()
        os.environ.setdefault("SSL_CERT_FILE", bundle)
        os.environ.setdefault("REQUESTS_CA_BUNDLE", bundle)
    except ImportError:
        pass


def _yf_history_raw(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """Try yf.download (more reliable than Ticker.history for some symbols), then Ticker.history."""
    _configure_ssl_bundle()
    import yfinance as yf

    last_err: Exception | None = None
    for attempt in range(3):
        for use_download in (True, False):
            try:
                if use_download:
                    raw = yf.download(
                        symbol,
                        period=period,
                        interval=interval,
                        progress=False,
                        auto_adjust=False,
                        threads=False,
                    )
                    raw = _flatten_yfinance_columns(raw)
                else:
                    raw = yf.Ticker(symbol).history(
                        period=period, interval=interval, auto_adjust=False
                    )
                if raw is not None and not raw.empty:
                    return raw
            except Exception as e:
                last_err = e
                logger.debug(f"yfinance attempt={attempt} download={use_download} {symbol!r}: {e}")
        time.sleep(0.8 * (attempt + 1))

    msg = f"No OHLCV returned for {symbol!r} (period={period}). {_YF_HINT}"
    if last_err is not None:
        raise RuntimeError(msg) from last_err
    raise RuntimeError(msg)


def _ohlcv_from_raw(raw: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    df = raw.rename(columns={k: v for k, v in rename.items() if k in raw.columns})
    need = ["open", "high", "low", "close"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise RuntimeError(f"OHLCV frame missing columns {missing}; got {list(df.columns)}")
    if "volume" not in df.columns:
        df = df.copy()
        df["volume"] = 0.0
    df = df[["open", "high", "low", "close", "volume"]]
    return _normalize_ohlcv_index(df)


def _normalize_ohlcv_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_convert(settings.timezone)
    out.index = pd.DatetimeIndex(out.index).normalize()
    out.index.name = "timestamp"
    return out.sort_index()


def fetch_ohlcv_yfinance(
    symbol: str,
    period: str = "10y",
    interval: str = "1d",
    fallback_symbols: List[str] | None = None,
) -> pd.DataFrame:
    candidates = [symbol]
    if fallback_symbols:
        for s in fallback_symbols:
            if s and s not in candidates:
                candidates.append(s)

    last_err: Exception | None = None
    for sym in candidates:
        try:
            raw = _yf_history_raw(sym, period, interval)
            df = _ohlcv_from_raw(raw)
            if sym != symbol:
                logger.info(f"Used fallback Yahoo symbol {sym!r} (primary {symbol!r} had no rows)")
            return df
        except Exception as e:
            last_err = e
            logger.warning(f"OHLCV fetch failed for {sym!r}: {e}")

    raise RuntimeError(
        f"Could not load OHLCV for any of {candidates!r}. {_YF_HINT}"
    ) from last_err


def fetch_india_vix_yfinance(symbol: str = "^INDIAVIX", period: str = "10y", interval: str = "1d") -> pd.DataFrame:
    raw = _yf_history_raw(symbol, period, interval)
    if "Close" not in raw.columns:
        raise RuntimeError(f"VIX frame has no Close column: {list(raw.columns)}")
    df = pd.DataFrame({"vix": raw["Close"].astype(float)})
    return _normalize_ohlcv_index(df)


def save_training_bundle(
    ohlcv: pd.DataFrame,
    vix: pd.DataFrame,
    out_dir: Path,
    meta: Dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ohlcv_path = out_dir / "sensex_ohlcv.csv"
    vix_path = out_dir / "india_vix.csv"
    ohlcv.to_csv(ohlcv_path, index_label="timestamp")
    vix.to_csv(vix_path, index_label="timestamp")
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    logger.info(f"Wrote {ohlcv_path} ({len(ohlcv)} rows), {vix_path} ({len(vix)} rows), {manifest_path}")


def load_training_bundle(data_dir: Path | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = data_dir or DEFAULT_DATA_DIR
    ohlcv_path = data_dir / "sensex_ohlcv.csv"
    vix_path = data_dir / "india_vix.csv"
    if not ohlcv_path.is_file() or not vix_path.is_file():
        raise FileNotFoundError(
            f"Missing {ohlcv_path} or {vix_path}. Run: python scripts/prepare_training_data.py"
        )
    ohlcv = pd.read_csv(ohlcv_path, index_col="timestamp", parse_dates=True)
    vix = pd.read_csv(vix_path, index_col="timestamp", parse_dates=True)
    return _normalize_ohlcv_index(ohlcv), _normalize_ohlcv_index(vix)


def build_manifest(
    *,
    ohlcv_symbol: str,
    vix_symbol: str,
    period: str,
    interval: str,
    rows_ohlcv: int,
    rows_vix: int,
) -> Dict[str, Any]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "ohlcv_symbol": ohlcv_symbol,
        "vix_symbol": vix_symbol,
        "period": period,
        "interval": interval,
        "rows_ohlcv": rows_ohlcv,
        "rows_vix": rows_vix,
        "source": "yfinance (offline prep only)",
    }
