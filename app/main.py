"""
Main entry point — starts both FastAPI (health/admin) and gRPC server.

On Render (or any single-port host), gRPC is the primary service on the
externally-routed port (GRPC_PORT / PORT).  FastAPI runs on a secondary
internal port (HTTP_PORT, default 8081) for health-checks and debugging.
"""
import asyncio
import os
import threading

import uvicorn
from fastapi import FastAPI
from loguru import logger

from app.config import settings
from app.logging_setup import configure_logging
from app.grpc_server.server import serve_grpc
from app.data.storage.db import engine, Base

configure_logging(settings.log_level)

RENDER_PORT = int(os.environ.get("PORT", "0"))
GRPC_PORT = RENDER_PORT if RENDER_PORT else settings.grpc_port
HTTP_PORT = int(os.environ.get("HTTP_PORT", "8081" if RENDER_PORT else "8000"))

app = FastAPI(title="Sensex ML Service", version="0.1.0")


@app.get("/")
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "grpc_listen_port": GRPC_PORT,
        "http_listen_port": HTTP_PORT,
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
    logger.info(f"Starting gRPC server on port {GRPC_PORT}")
    asyncio.run(serve_grpc(GRPC_PORT))


if __name__ == "__main__":
    logger.info("Sensex ML Service starting...")
    logger.info(f"gRPC port: {GRPC_PORT}, HTTP port: {HTTP_PORT}")

    grpc_thread = threading.Thread(target=start_grpc, daemon=True)
    grpc_thread.start()

    uvicorn.run(app, host="0.0.0.0", port=HTTP_PORT, log_level="info")