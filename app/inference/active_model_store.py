"""Centralized store for the active AI tool (GEMINI / OPENAI) and its model_id.

This is the single routing authority: main.py reads get_tool() to decide which
predictor class to instantiate, and each predictor calls get_model_id() for the
actual model string sent to the upstream API.

_ModelStore in gemini_predictor.py is kept for backward-compat (GeminiPredictor
still uses it internally); this store is updated in sync with it whenever the
active tool is GEMINI.
"""
from __future__ import annotations


class _ActiveModelStore:
    _tool: str = "GEMINI"
    _model_id: str = ""

    @classmethod
    def set(cls, tool: str, model_id: str) -> None:
        cls._tool = tool.strip().upper()
        cls._model_id = model_id.strip()

    @classmethod
    def get_tool(cls) -> str:
        return cls._tool or "GEMINI"

    @classmethod
    def get_model_id(cls) -> str:
        if cls._model_id:
            return cls._model_id
        from app.config import settings
        return settings.openai_model if cls.get_tool() == "OPENAI" else settings.gemini_model

    @classmethod
    def clear(cls) -> None:
        cls._tool = "GEMINI"
        cls._model_id = ""
