"""Configure loguru once: stderr, respect settings.log_level, thread-safe for gRPC + asyncio."""
from __future__ import annotations

import sys

from loguru import logger

_configured = False


def configure_logging(level: str = "INFO") -> None:
    global _configured
    if _configured:
        return
    _configured = True

    lvl = (level or "INFO").upper()
    logger.remove()
    logger.add(
        sys.stderr,
        level=lvl,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        enqueue=True,
        colorize=sys.stderr.isatty(),
    )
    # One line always visible if stderr works (even when level filters INFO)
    if lvl in ("WARNING", "ERROR", "CRITICAL"):
        print(
            f"sensex-ml-service: log_level={lvl} — prediction details use INFO; "
            f"set LOG_LEVEL=INFO or DEBUG in .env to see them.",
            file=sys.stderr,
        )
    else:
        logger.info("Logging configured (level={})", lvl)
