from celery import Celery
from app.config import settings

celery_app = Celery(
    "sensex_ml",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    timezone="Asia/Kolkata",
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    task_track_started=True,
    beat_schedule={
        "fetch-market-data": {
            "task": "app.scheduler.tasks.fetch_market_data",
            "schedule": 180.0,  # Every 3 minutes during market hours
        },
        "run-daily-prediction": {
            "task": "app.scheduler.tasks.run_daily_prediction",
            "schedule": {"hour": 8, "minute": 45},  # 8:45 AM IST
        },
        "evaluate-predictions": {
            "task": "app.scheduler.tasks.evaluate_predictions",
            "schedule": {"hour": 16, "minute": 0},  # 4:00 PM IST
        },
        "retrain-models": {
            "task": "app.scheduler.tasks.retrain_models",
            "schedule": {"day_of_week": 0, "hour": 22},  # Sunday 10 PM
        },
    },
)