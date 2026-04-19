"""In-process counters for gRPC / prediction visibility (works even when Loguru output is lost)."""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

_lock = threading.Lock()
_state: Dict[str, Any] = {
    "grpc_listen_addr": None,
    "grpc_stubs_ok": None,
    "get_prediction_started": 0,
    "get_prediction_completed": 0,
    "get_prediction_aborted_invalid": 0,
    "get_prediction_aborted_other": 0,
    "last_get_prediction_at": None,
    "last_get_prediction_error": None,
    "last_prediction_direction": None,
    "last_prediction_confidence": None,
    "last_predictor_error": None,
    "stream_live_ticks_received": 0,
    "stream_live_batches": 0,
}


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def set_grpc_listen(addr: str, stubs_ok: bool) -> None:
    with _lock:
        _state["grpc_listen_addr"] = addr
        _state["grpc_stubs_ok"] = stubs_ok


def record_get_prediction_start(n_bars: int) -> None:
    with _lock:
        _state["get_prediction_started"] = int(_state["get_prediction_started"]) + 1
        _state["last_get_prediction_at"] = _now_iso()
        _state["last_get_prediction_error"] = None
        _state["last_predictor_error"] = None
        _state["_last_n_bars"] = n_bars


def record_get_prediction_abort(invalid: bool, detail: str) -> None:
    with _lock:
        if invalid:
            _state["get_prediction_aborted_invalid"] = int(_state["get_prediction_aborted_invalid"]) + 1
        else:
            _state["get_prediction_aborted_other"] = int(_state["get_prediction_aborted_other"]) + 1
        _state["last_get_prediction_error"] = detail[:500]


def record_get_prediction_success(result: Dict[str, Any]) -> None:
    with _lock:
        _state["get_prediction_completed"] = int(_state["get_prediction_completed"]) + 1
        _state["last_prediction_direction"] = result.get("direction")
        _state["last_prediction_confidence"] = result.get("confidence")
        if result.get("error"):
            _state["last_predictor_error"] = str(result.get("error"))


def record_stream_tick() -> None:
    with _lock:
        _state["stream_live_ticks_received"] = int(_state["stream_live_ticks_received"]) + 1


def record_stream_batch_closed(n: int) -> None:
    with _lock:
        _state["stream_live_batches"] = int(_state["stream_live_batches"]) + 1
        _state["_last_stream_batch_ticks"] = n


def snapshot() -> Dict[str, Any]:
    with _lock:
        return {k: v for k, v in _state.items() if not str(k).startswith("_")}
