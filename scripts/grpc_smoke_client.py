#!/usr/bin/env python3
"""
Local check: can the ML server decode GetPrediction? Run from repo root:

  python scripts/grpc_smoke_client.py --host localhost --port 50051

If this works but Java fails, compare .proto field numbers/types (especially double vs float on OHLC).
If this fails with DecodeError too, the running server likely uses stale *_pb2.py or wrong port.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _build_request(*, n_bars: int, n_vix: int) -> "object":
    from app.grpc_server.generated import prediction_service_pb2 as pb2

    base_ms = 1_700_000_000_000  # arbitrary epoch ms
    day_ms = 86_400_000
    req = pb2.PredictionRequest(
        horizon="1D",
        underlying_symbol="RELIANCE.BSE",
        instrument_token="",  # set to Angel One token string to match LiveTick.token
    )
    for i in range(n_bars):
        ms = base_ms + i * day_ms
        b = req.sensex_ohlcv.add()
        b.timestamp_unix_ms = ms
        b.open = 100.0 + i * 0.01
        b.high = b.open + 1.0
        b.low = b.open - 1.0
        b.close = b.open + 0.5
        b.volume = 1_000_000
    for j in range(n_vix):
        ms = base_ms + j * day_ms
        p = req.india_vix.add()
        p.timestamp_unix_ms = ms
        p.vix = 15.0 + j * 0.01
    return req


async def _run(host: str, port: int, n_bars: int) -> None:
    import grpc.aio
    from app.grpc_server.generated import prediction_service_pb2_grpc as pbg

    target = f"{host}:{port}"
    channel = grpc.aio.insecure_channel(target)
    stub = pbg.PredictionServiceStub(channel)
    req = _build_request(n_bars=n_bars, n_vix=max(30, n_bars // 4))
    try:
        resp = await stub.GetPrediction(req, timeout=120.0)
        print("OK", resp)
    except grpc.aio.AioRpcError as e:
        print("RPC failed:", e.code(), e.details())
        raise SystemExit(1)
    finally:
        await channel.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=50051)
    p.add_argument("--bars", type=int, default=120, help="Must be >= min_ohlcv_bars_grpc on server (default 100)")
    args = p.parse_args()
    asyncio.run(_run(args.host, args.port, args.bars))


if __name__ == "__main__":
    main()
