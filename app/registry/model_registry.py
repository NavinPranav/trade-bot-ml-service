"""MLflow model registry integration."""
from loguru import logger


def log_model(model_name: str, metrics: dict, model_path: str):
    try:
        import mlflow
        with mlflow.start_run(run_name=model_name):
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)
            mlflow.log_artifact(model_path)
            logger.info(f"Model {model_name} logged to MLflow")
    except Exception as e:
        logger.warning(f"MLflow logging failed (non-critical): {e}")