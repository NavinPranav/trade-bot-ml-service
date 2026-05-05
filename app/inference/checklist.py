"""8-step Bank Nifty Pro Trading Checklist — mirrors the frontend JS logic.

Called from GeminiPredictor.predict() so every live-tick re-prediction carries
the same structured signal context as the frontend checklist panel.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict
from zoneinfo import ZoneInfo

import pandas as pd
from loguru import logger

IST = ZoneInfo("Asia/Kolkata")


def _ema(closes: list[float], period: int) -> float | None:
    if len(closes) < period:
        return None
    k = 2 / (period + 1)
    ema = sum(closes[:period]) / period
    for p in closes[period:]:
        ema = p * k + ema * (1 - k)
    return ema


def _rsi(closes: list[float], period: int = 14) -> float | None:
    if len(closes) <= period:
        return None
    gains = losses = 0.0
    for i in range(1, period + 1):
        d = closes[i] - closes[i - 1]
        if d > 0:
            gains += d
        else:
            losses -= d
    avg_gain = gains / period
    avg_loss = losses / period
    for i in range(period + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        avg_gain = (avg_gain * (period - 1) + (d if d > 0 else 0)) / period
        avg_loss = (avg_loss * (period - 1) + (-d if d < 0 else 0)) / period
    return 100.0 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)


def _attach_ist_dates(ohlcv: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Return (df_with_date_col, today_str) where date is IST YYYY-MM-DD."""
    df = ohlcv.copy()
    idx = df.index
    try:
        if getattr(idx, "tz", None) is None:
            idx = idx.tz_localize("UTC")
        idx_ist = idx.tz_convert(IST)
    except Exception:
        idx_ist = idx
    df["_date"] = [d.strftime("%Y-%m-%d") for d in idx_ist]
    today = datetime.now(IST).strftime("%Y-%m-%d")
    return df, today


def run_checklist(ohlcv: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute all 8 active checklist steps from an intraday OHLCV DataFrame.

    Expects columns: open, high, low, close, volume with a DatetimeIndex.
    Returns a dict with per-step signals plus an 'overall' verdict.
    """
    if ohlcv.empty:
        return {"overall": "NO_TRADE", "error": "empty_ohlcv"}

    try:
        df, today = _attach_ist_dates(ohlcv)
    except Exception as e:
        logger.warning(f"Checklist: could not attach IST dates: {e}")
        return {"overall": "NO_TRADE", "error": str(e)}

    closes = df["close"].astype(float).tolist()
    today_df = df[df["_date"] == today]

    # ── Step 1: TREND (EMA 20 vs EMA 50) ──────────────────────────────
    ema20, ema50 = _ema(closes, 20), _ema(closes, 50)
    if ema20 is None or ema50 is None:
        step1 = {"signal": "INSUFFICIENT_DATA", "trade_type": "NONE"}
    else:
        diff = abs(ema20 - ema50) / ema50
        if diff <= 0.001:
            sig, trade = "MIXED", "NONE"
        elif ema20 > ema50:
            sig, trade = "BULLISH", "CE"
        else:
            sig, trade = "BEARISH", "PE"
        step1 = {"signal": sig, "trade_type": trade,
                 "ema20": round(ema20, 2), "ema50": round(ema50, 2)}

    # ── Step 2: VWAP ──────────────────────────────────────────────────
    if today_df.empty:
        step2 = {"signal": "INSUFFICIENT_DATA"}
    else:
        tp = (today_df["high"].astype(float)
              + today_df["low"].astype(float)
              + today_df["close"].astype(float)) / 3
        vol = today_df["volume"].astype(float).replace(0, 1)
        vwap = float((tp * vol).sum() / vol.sum())
        cp = float(today_df["close"].iloc[-1])
        diff_pct = abs(cp - vwap) / vwap if vwap else 0
        sig = "AVOID" if diff_pct <= 0.0005 else ("BULLISH" if cp > vwap else "BEARISH")
        step2 = {"signal": sig, "vwap": round(vwap, 2),
                 "current_price": cp, "diff_pct": round(diff_pct * 100, 3)}

    # ── Step 3: MOMENTUM RSI(14) ───────────────────────────────────────
    rsi_val = _rsi(closes)
    if rsi_val is None:
        step3 = {"signal": "INSUFFICIENT_DATA"}
    else:
        sig = "BUY" if rsi_val > 55 else ("SELL" if rsi_val < 45 else "SIDEWAYS")
        step3 = {"signal": sig, "rsi": round(rsi_val, 2)}

    # ── Step 5: LEVELS (Pivot Points) ─────────────────────────────────
    dates_sorted = sorted(df["_date"].unique())
    today_idx = dates_sorted.index(today) if today in dates_sorted else -1
    if today_idx < 1:
        step5 = {"signal": "INSUFFICIENT_DATA"}
    else:
        prev = df[df["_date"] == dates_sorted[today_idx - 1]]
        H = float(prev["high"].max())
        L = float(prev["low"].min())
        C = float(prev["close"].iloc[-1])
        pivot = (H + L + C) / 3
        r1, r2 = 2 * pivot - L, pivot + (H - L)
        s1, s2 = 2 * pivot - H, pivot - (H - L)
        if today_df.empty:
            step5 = {"signal": "INSUFFICIENT_DATA", "pivot": round(pivot, 2)}
        else:
            cp = float(today_df["close"].iloc[-1])
            PROX = 0.003
            above_r1 = cp > r1 * (1 + PROX)
            below_s1 = cp < s1 * (1 - PROX)
            near_r = (abs(cp - r1) / r1 <= PROX or abs(cp - r2) / r2 <= PROX)
            near_s = (abs(cp - s1) / s1 <= PROX or abs(cp - s2) / s2 <= PROX)
            sig = ("BREAKOUT_UP" if above_r1 else "BREAKOUT_DOWN" if below_s1
                   else "NEAR_RESISTANCE" if near_r else "NEAR_SUPPORT" if near_s
                   else "NO_LEVEL")
            step5 = {"signal": sig, "pivot": round(pivot, 2),
                     "r1": round(r1, 2), "r2": round(r2, 2),
                     "s1": round(s1, 2), "s2": round(s2, 2)}

    # ── Step 6: ENTRY CANDLE ──────────────────────────────────────────
    valid = df[df["high"].astype(float) > 0]
    if len(valid) < 2:
        step6 = {"signal": "INSUFFICIENT_DATA"}
    else:
        last = valid.iloc[-2]
        o, cl = float(last["open"]), float(last["close"])
        h, l = float(last["high"]), float(last["low"])
        rng = h - l
        if rng == 0:
            step6 = {"signal": "WEAK_CANDLE", "body_ratio": 0}
        else:
            br = abs(cl - o) / rng
            is_bull = cl > o
            sig = ("BULLISH_CANDLE" if br > 0.6 and is_bull
                   else "BEARISH_CANDLE" if br > 0.6 else "WEAK_CANDLE")
            step6 = {"signal": sig, "body_ratio": round(br, 3), "is_bullish": is_bull}

    # ── Step 7: STRIKE SELECTION ──────────────────────────────────────
    if today_df.empty:
        step7 = {"signal": "INSUFFICIENT_DATA"}
    else:
        cp = float(today_df["close"].iloc[-1])
        atm = round(cp / 100) * 100
        b = sum([step2.get("signal") == "BULLISH", step3.get("signal") == "BUY",
                 step5.get("signal") in ("NEAR_SUPPORT", "BREAKOUT_UP"),
                 step6.get("signal") == "BULLISH_CANDLE"])
        s = sum([step2.get("signal") == "BEARISH", step3.get("signal") == "SELL",
                 step5.get("signal") in ("NEAR_RESISTANCE", "BREAKOUT_DOWN"),
                 step6.get("signal") == "BEARISH_CANDLE"])
        conf = max(b, s) / 4 * 100
        is_bull = step1.get("signal") == "BULLISH"
        is_bkout = step5.get("signal") in ("BREAKOUT_UP", "BREAKOUT_DOWN")
        st = "OTM" if is_bkout else ("ATM" if conf >= 75 else "ITM")
        rec = (atm + 100 if (st == "OTM" and is_bull) else
               atm - 100 if st == "OTM" else
               atm if st == "ATM" else
               atm - 100 if is_bull else atm + 100)
        step7 = {"signal": st, "recommended_strike": rec, "atm": atm,
                 "confidence": round(conf, 1)}

    # ── Step 8: RISK MANAGEMENT ───────────────────────────────────────
    step8 = {"signal": "VALID", "sl_pct": 12, "target_pct": 25,
             "risk_reward": 2.08, "max_trades": 2}

    # ── Step 9: NO TRADE CONDITIONS ───────────────────────────────────
    now_ist = datetime.now(IST)
    mins = now_ist.hour * 60 + now_ist.minute
    too_early = 555 <= mins < 570   # 9:15–9:30 AM
    too_late  = 915 <= mins <= 930  # 3:15–3:30 PM

    date_vols: dict[str, float] = {}
    for d, v in zip(df["_date"], df["volume"].astype(float)):
        date_vols[d] = date_vols.get(d, 0) + float(v)
    past_vols = [date_vols[d] for d in sorted(date_vols) if d < today][-20:]
    avg_vol = sum(past_vols) / len(past_vols) if past_vols else 0
    low_vol = avg_vol > 0 and date_vols.get(today, 0) < avg_vol * 0.8

    b_c = sum([step2.get("signal") == "BULLISH", step3.get("signal") == "BUY",
               step5.get("signal") in ("NEAR_SUPPORT", "BREAKOUT_UP"),
               step6.get("signal") == "BULLISH_CANDLE"])
    s_c = sum([step2.get("signal") == "BEARISH", step3.get("signal") == "SELL",
               step5.get("signal") in ("NEAR_RESISTANCE", "BREAKOUT_DOWN"),
               step6.get("signal") == "BEARISH_CANDLE"])
    conflict = min(b_c, s_c) >= 2

    no_trade = (step2.get("signal") == "AVOID" or step3.get("signal") == "SIDEWAYS"
                or low_vol or conflict or too_early or too_late)
    step9 = {
        "signal": "NO_TRADE" if no_trade else "GO",
        "reasons": {
            "vwap_avoid":         step2.get("signal") == "AVOID",
            "rsi_sideways":       step3.get("signal") == "SIDEWAYS",
            "low_volume":         low_vol,
            "conflicting_signals": conflict,
            "market_timing":      too_early or too_late,
        },
    }

    overall = "NO_TRADE" if no_trade else (step1.get("trade_type") or "HOLD")

    return {
        "overall": overall,
        "step1_trend":    step1,
        "step2_vwap":     step2,
        "step3_rsi":      step3,
        "step5_levels":   step5,
        "step6_candle":   step6,
        "step7_strike":   step7,
        "step8_risk":     step8,
        "step9_no_trade": step9,
    }
