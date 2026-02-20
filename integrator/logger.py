"""
Entropy Engine — Structured Logger
====================================
Per-tick structured logging for the orchestration layer.
Provides human-readable console output and machine-parseable fields.
"""

from __future__ import annotations

import logging
from config import LOG_LEVEL


def setup_logger(name: str = "orchestrator") -> logging.Logger:
    """Create and configure a structured logger."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # already configured

    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)-15s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


# ── Module-level loggers ──
log = setup_logger("orchestrator")
log_safety = setup_logger("safety")
log_ai = setup_logger("ai-mode")
log_confidence = setup_logger("confidence")


def log_tick(tick_num: int, metrics: dict, decision: dict, safety: dict) -> None:
    """Log a single orchestration tick."""
    log.info(
        "Tick #%d | T=%.0f°C P=%.1fbar V=%.0f→%.0f%% W=%.0fkW | "
        "mode=%s safety=%s conf=%.3f",
        tick_num,
        metrics.get("temperature", 0),
        metrics.get("pressure", 0),
        metrics.get("valve_position", 0),
        decision.get("valve", 0),
        metrics.get("power_output", 0),
        decision.get("mode", "?"),
        safety.get("level", "?"),
        decision.get("confidence", 0),
    )


def log_safety_event(level: str, message: str) -> None:
    """Log a safety-related event."""
    if level == "CRITICAL":
        log_safety.critical("🚨 %s", message)
    elif level == "WARNING":
        log_safety.warning("⚠️  %s", message)
    else:
        log_safety.info("✅ %s", message)


def log_ai_event(event: str, message: str) -> None:
    """Log an AI mode transition."""
    log_ai.info("[%s] %s", event, message)


def log_confidence_event(confidence: float, threshold: float, action: str) -> None:
    """Log a confidence-related event."""
    log_confidence.info(
        "Confidence=%.3f threshold=%.3f → %s",
        confidence, threshold, action,
    )
