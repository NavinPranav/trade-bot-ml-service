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
    # Minimum bars required. Intra-day 1M/5M feeds accumulate quickly; 30 bars covers ~30 min of 1M data.
    min_ohlcv_bars_grpc: int = 30

    # Live tick stream → debounced inference (uses baseline OHLCV from last GetPrediction for that symbol).
    live_inference_enabled: bool = True
    live_inference_interval_sec: float = 300.0

    # Google Gemini (used by GetGeminiPrediction RPC when prediction_engine=AI)
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"

    # Post-coercion policy (tune without code changes — env / .env)
    gemini_min_confidence: float = 65.0
    gemini_min_risk_reward: float = 1.5
    # When EMA9 vs EMA21 separation is strong, allow lower confidence for BUY/SELL (less excessive HOLD).
    gemini_strong_trend_min_ema_gap_pct: float = 0.10
    gemini_relaxed_confidence_floor_strong_trend: float = 58.0
    # SELL + checklist near support (S1/S2): require higher confidence to cut false SELLs into support.
    gemini_sell_near_support_min_confidence: float = 72.0
    # If > 0: force HOLD when ATR(14)/price*100 is below this (very dead tape). 0 = disabled.
    gemini_min_atr_pct_of_price: float = 0.0

    # Live predict: retries on HTTP 429 before HOLD placeholder (same idea as /admin/analyse).
    gemini_429_max_retries: int = 4
    gemini_429_retry_base_delay_sec: float = 8.0

    # News API (newsapi.org) — financial sentiment enrichment
    news_api_key: str = ""
    news_fetch_interval_sec: int = 900   # 15-minute cache TTL (free tier: 100 req/day)

    # Logging
    log_level: str = "DEBUG"
    # Print GetPrediction lines to stderr (works when Loguru output is not visible). Env: SENSEX_PRINT_RPC=true
    sensex_print_rpc: bool = False

    @property
    def db_url(self) -> str:
        return f"postgresql+psycopg://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    @property
    def async_db_url(self) -> str:
        return f"postgresql+psycopg_async://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/0"


settings = Settings()