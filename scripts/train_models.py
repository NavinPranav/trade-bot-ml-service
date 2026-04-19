#!/usr/bin/env python3
"""
Train all models using CSVs produced by scripts/prepare_training_data.py.

Example:
  python scripts/prepare_training_data.py
  python scripts/train_models.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    from loguru import logger

    from app.training.prepared_data import DEFAULT_DATA_DIR
    from app.training.trainer import ModelTrainer

    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = p.parse_args()

    trainer = ModelTrainer()
    results = trainer.train_from_prepared(args.data_dir)
    if results.get("error"):
        logger.error(results)
        raise SystemExit(1)
    print(json.dumps(results, default=str, indent=2))


if __name__ == "__main__":
    main()
