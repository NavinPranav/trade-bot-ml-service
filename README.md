# Sensex ML Service

Python ML prediction service for Sensex options trading.

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate gRPC stubs
make proto

# Train models (fetches data + trains all)
make train

# Run service (FastAPI + gRPC)
make run
```

## Architecture

Three model categories:
- **Statistical** (ARIMA, GARCH) — not ML, math-based
- **Classical ML** (XGBoost, LightGBM) — feature-engineered
- **Deep Learning** (LSTM, TFT) — learns from raw sequences

All feed into a stacking ensemble that outputs:
direction, magnitude, volatility, confidence.

## gRPC port: 50051 | Health API: http://localhost:8000/health
