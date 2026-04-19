from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        protected_namespaces=("settings_",),
        extra="ignore",
    )

    # Database
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "sensex_trader"
    db_user: str = "postgres"
    db_password: str = "postgres"

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379

    # gRPC
    grpc_port: int = 50051

    # Models
    model_dir: Path = Path("./models")
    mlflow_tracking_uri: str = "http://localhost:5000"

    # Market
    market_open: str = "09:15"
    market_close: str = "15:30"
    timezone: str = "Asia/Kolkata"
    # Minimum rows after aggregating gRPC OhlcvBar → one row per calendar day (not raw bar count).
    min_ohlcv_bars_grpc: int = 100

    # Live tick stream → debounced inference (uses baseline OHLCV from last GetPrediction for that symbol).
    live_inference_enabled: bool = True
    live_inference_interval_sec: float = 5.0

    # Google Gemini (used by GetGeminiPrediction RPC when prediction_engine=AI)
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"

    # Logging
    log_level: str = "DEBUG"
    # Print GetPrediction lines to stderr (works when Loguru output is not visible). Env: SENSEX_PRINT_RPC=true
    sensex_print_rpc: bool = False

    @property
    def db_url(self) -> str:
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    @property
    def async_db_url(self) -> str:
        return f"postgresql+asyncpg://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/0"


settings = Settings()