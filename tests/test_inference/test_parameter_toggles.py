"""Tests for the AI parameter toggle store (Phase 4.4).

Covers three layers:

1. ``_ParameterTogglesStore`` — pure-Python singleton. Verifies defaults,
   that ``update_many`` flips optional keys, ignores unknown keys, and
   refuses to disable required ones (defensive second layer).
2. The FastAPI admin endpoints ``GET`` / ``PUT /admin/parameters`` — verifies
   the round-trip and that required keys cannot be turned off via HTTP.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.inference.gemini_predictor import _ParameterTogglesStore
from app.main import app


@pytest.fixture(autouse=True)
def _reset_store():
    _ParameterTogglesStore.reset()
    yield
    _ParameterTogglesStore.reset()


# ── Store: pure unit tests ──────────────────────────────────────────────


def test_defaults_all_enabled_initially():
    state = _ParameterTogglesStore.to_dict()
    # All known keys present and on by default
    assert set(state.keys()) == set(_ParameterTogglesStore._DEFAULTS.keys())
    assert all(state.values())


def test_required_keys_cannot_be_disabled():
    # Try to flip a required key OFF — must remain ON.
    _ParameterTogglesStore.update_many({"ohlcv_bars": False, "trend_context": False})
    assert _ParameterTogglesStore.is_enabled("ohlcv_bars") is True
    assert _ParameterTogglesStore.is_enabled("trend_context") is True


def test_optional_keys_can_be_toggled():
    _ParameterTogglesStore.update_many({"news_sentiment": False, "india_vix": False})
    assert _ParameterTogglesStore.is_enabled("news_sentiment") is False
    assert _ParameterTogglesStore.is_enabled("india_vix") is False

    _ParameterTogglesStore.update_many({"news_sentiment": True})
    assert _ParameterTogglesStore.is_enabled("news_sentiment") is True


def test_unknown_keys_silently_ignored():
    _ParameterTogglesStore.update_many({"this_key_does_not_exist": False, "checklist_signal": False})
    assert _ParameterTogglesStore.is_enabled("checklist_signal") is False
    assert "this_key_does_not_exist" not in _ParameterTogglesStore.to_dict()


def test_truthy_values_are_coerced_to_bool():
    _ParameterTogglesStore.update_many({"news_sentiment": 0, "india_vix": "yes"})
    # Optional keys are bool-coerced (0 → False, non-empty str → True)
    assert _ParameterTogglesStore.is_enabled("news_sentiment") is False
    assert _ParameterTogglesStore.is_enabled("india_vix") is True


def test_reset_restores_defaults():
    _ParameterTogglesStore.update_many({"news_sentiment": False, "checklist_signal": False})
    _ParameterTogglesStore.reset()
    state = _ParameterTogglesStore.to_dict()
    assert state == _ParameterTogglesStore._DEFAULTS


# ── HTTP: admin endpoints ───────────────────────────────────────────────


def _client() -> TestClient:
    return TestClient(app)


def test_get_parameters_returns_current_map():
    r = _client().get("/admin/parameters")
    assert r.status_code == 200
    body = r.json()
    assert "toggles" in body and isinstance(body["toggles"], dict)
    # Required keys should also be reported as required by the endpoint.
    assert set(body["required"]) >= {"ohlcv_bars", "trend_context", "current_price", "target_minutes", "technical_indicators"}


def test_put_parameters_disables_optional_only():
    r = _client().put("/admin/parameters", json={"toggles": {
        "news_sentiment": False,
        "ohlcv_bars": False,         # required, must stay True
        "checklist_signal": False,
    }})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["toggles"]["news_sentiment"] is False
    assert body["toggles"]["checklist_signal"] is False
    assert body["toggles"]["ohlcv_bars"] is True   # forced back ON by the store


def test_put_parameters_ignores_unknown_keys():
    r = _client().put("/admin/parameters", json={"toggles": {
        "totally_made_up": True,
        "news_sentiment": False,
    }})
    assert r.status_code == 200
    body = r.json()
    assert "totally_made_up" not in body["toggles"]
    assert body["toggles"]["news_sentiment"] is False
