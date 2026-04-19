#!/usr/bin/env python3
"""
Build CSV training bundles under data/training/ for ModelTrainer.

Requires: yfinance (project dependency). Network access.

Example:
  python scripts/prepare_training_data.py --period 10y
  python scripts/prepare_training_data.py --ohlcv-symbol RELIANCE.BSE
  python scripts/train_models.py

If ^BSESN fails (Yahoo JSON/SSL on macOS): pip install 'urllib3>=1.26.18,<2' && pip install -U yfinance certifi
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    from loguru import logger

    from app.training.prepared_data import (
        DEFAULT_DATA_DIR,
        build_manifest,
        fetch_india_vix_yfinance,
        fetch_ohlcv_yfinance,
        save_training_bundle,
    )

    p = argparse.ArgumentParser(description="Download OHLCV + India VIX and save to data/training/")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--ohlcv-symbol", default="^BSESN", help="Yahoo symbol for index/equity OHLCV")
    p.add_argument("--vix-symbol", default="^INDIAVIX", help="Yahoo symbol for India VIX")
    p.add_argument("--period", default="10y", help="yfinance period, e.g. 5y, 10y, max")
    p.add_argument("--interval", default="1d", help="yfinance interval (default daily)")
    p.add_argument(
        "--ohlcv-fallback",
        action="append",
        default=[],
        metavar="SYMBOL",
        help="Extra Yahoo symbols to try if the primary OHLCV symbol returns no rows (repeatable)",
    )
    args = p.parse_args()

    fallbacks: list[str] = list(args.ohlcv_fallback)
    if args.ohlcv_symbol == "^BSESN":
        for alt in ("SENSEX.BO",):
            if alt not in fallbacks:
                fallbacks.insert(0, alt)

    logger.info(f"Fetching OHLCV {args.ohlcv_symbol} period={args.period} ...")
    ohlcv = fetch_ohlcv_yfinance(
        args.ohlcv_symbol,
        period=args.period,
        interval=args.interval,
        fallback_symbols=fallbacks or None,
    )
    logger.info(f"Fetching VIX {args.vix_symbol} ...")
    vix = fetch_india_vix_yfinance(args.vix_symbol, period=args.period, interval=args.interval)

    meta = build_manifest(
        ohlcv_symbol=args.ohlcv_symbol,
        vix_symbol=args.vix_symbol,
        period=args.period,
        interval=args.interval,
        rows_ohlcv=len(ohlcv),
        rows_vix=len(vix),
    )
    save_training_bundle(ohlcv, vix, args.out_dir, meta)
    logger.info("Done. Train with: python scripts/train_models.py")


if __name__ == "__main__":
    main()
