"""Options-specific features: PCR, IV rank, max pain."""
import pandas as pd
import numpy as np
from loguru import logger


def compute_pcr(options_chain: pd.DataFrame) -> dict:
    if options_chain.empty:
        return {"pcr_oi": 0, "pcr_volume": 0}

    calls = options_chain[options_chain["option_type"] == "CALL"]
    puts = options_chain[options_chain["option_type"] == "PUT"]

    call_oi = calls["oi"].sum()
    put_oi = puts["oi"].sum()
    call_vol = calls["volume"].sum()
    put_vol = puts["volume"].sum()

    return {
        "pcr_oi": put_oi / call_oi if call_oi > 0 else 0,
        "pcr_volume": put_vol / call_vol if call_vol > 0 else 0,
    }


def compute_iv_rank(current_iv: float, iv_history: pd.Series) -> dict:
    if iv_history.empty or len(iv_history) < 20:
        return {"iv_rank": 50, "iv_percentile": 50}

    iv_min = iv_history.min()
    iv_max = iv_history.max()
    iv_rank = ((current_iv - iv_min) / (iv_max - iv_min)) * 100 if iv_max > iv_min else 50
    iv_percentile = (iv_history < current_iv).sum() / len(iv_history) * 100

    return {"iv_rank": round(iv_rank, 2), "iv_percentile": round(iv_percentile, 2)}


def compute_max_pain(options_chain: pd.DataFrame) -> float:
    if options_chain.empty:
        return 0

    strikes = options_chain["strike"].unique()
    min_pain = float("inf")
    max_pain_strike = 0

    for strike in strikes:
        calls = options_chain[(options_chain["option_type"] == "CALL") & (options_chain["strike"] < strike)]
        puts = options_chain[(options_chain["option_type"] == "PUT") & (options_chain["strike"] > strike)]

        call_pain = ((strike - calls["strike"]) * calls["oi"]).sum()
        put_pain = ((puts["strike"] - strike) * puts["oi"]).sum()
        total_pain = call_pain + put_pain

        if total_pain < min_pain:
            min_pain = total_pain
            max_pain_strike = strike

    return max_pain_strike