"""
Main entry point — starts both FastAPI (health/admin) and gRPC server.
"""
import asyncio
import threading

import uvicorn
from fastapi import FastAPI
from loguru import logger

from app.config import settings
from app.logging_setup import configure_logging
from app.grpc_server.server import serve_grpc
from app.data.storage.db import engine, Base

configure_logging(settings.log_level)

app = FastAPI(title="Sensex ML Service", version="0.1.0")


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "grpc_listen_port": settings.grpc_port,
        "http_listen_port": 8000,
        "grpc_note": (
            "Use gRPC (plaintext or matching TLS) on grpc_listen_port. "
            "Port 8000 is HTTP/FastAPI only; pointing a gRPC client at 8000 causes protobuf DecodeError."
        ),
        "debug_hint": "GET /debug/ml for gRPC prediction counters (if GetPrediction is never called, counters stay 0).",
    }


@app.get("/debug/ml")
async def debug_ml():
    """In-process stats: use this if Loguru logs are missing (Docker/IDE)."""
    from app.grpc_diagnostics import snapshot

    return snapshot()


@app.get("/models/status")
async def model_status():
    from app.inference.predictor import Predictor
    predictor = Predictor()
    return predictor.get_model_health()


def start_grpc():
    """Run gRPC server in a background thread."""
    logger.info(f"Starting gRPC server on port {settings.grpc_port}")
    asyncio.run(serve_grpc())


if __name__ == "__main__":
    logger.info("Sensex ML Service starting...")

    # Start gRPC in background thread
    grpc_thread = threading.Thread(target=start_grpc, daemon=True)
    grpc_thread.start()

    # Start FastAPI (health checks + admin)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")