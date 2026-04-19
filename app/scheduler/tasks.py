"""Celery tasks for scheduled operations."""
from loguru import logger
from app.scheduler.celery_config import celery_app


@celery_app.task
def fetch_market_data():
    # Market data is pushed by the Java backend — nothing to fetch here
    logger.info("Task: fetch_market_data is a no-op; data is supplied via gRPC from the backend")
    return {"status": "no-op"}


@celery_app.task
def run_daily_prediction():
    # Predictions are triggered via gRPC from the Java backend, not scheduled here
    logger.info("Task: run_daily_prediction is a no-op; predictions are triggered via gRPC")
    return {"status": "no-op"}


@celery_app.task
def evaluate_predictions():
    logger.info("Task: Evaluating past predictions against actual market close...")
    # TODO: Compare stored predictions with actual Sensex close
    return {"status": "stub"}


@celery_app.task
def retrain_models(ohlcv_records: list, vix_records: list):
    """Retrain all models. Caller must supply ohlcv_records and vix_records as lists of dicts."""
    import pandas as pd
    from app.training.trainer import ModelTrainer
    logger.info("Task: Retraining all models...")
    ohlcv = pd.DataFrame(ohlcv_records)
    vix = pd.DataFrame(vix_records)
    trainer = ModelTrainer()
    results = trainer.train_all(ohlcv, vix)
    return results


@celery_app.task
def fetch_news_sentiment():
    logger.info("Task: Fetching news sentiment...")
    from app.features.sentiment_features import sentiment_analyzer
    from app.data.ingestion.news_fetcher import fetch_news_headlines
    headlines = fetch_news_headlines()
    if headlines:
        score = sentiment_analyzer.get_aggregate_sentiment([h["title"] for h in headlines])
        logger.info(f"Aggregate sentiment score: {score}")
        return {"sentiment_score": score}
    return {"sentiment_score": 0}