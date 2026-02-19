"""
Entropy Engine — Safety Enforcement Layer
===========================================
Absolute last line of defense.
Runs AFTER the AI decision, BEFORE sending the control command.

Even if the model is wrong, these hard-coded physical limits
prevent the plant from reaching dangerous operating conditions.
"""

from __future__ import annotations

import logging
import sys

sys.path.insert(0, ".")
from config import (
    MAX_VALVE,
    MIN_VALVE,
    PRESSURE_CRITICAL,
    PRESSURE_HARD_LIMIT,
    PRESSURE_SAFETY_LIMIT,
    TEMPERATURE_CRITICAL,
)

logger = logging.getLogger("safety")


# ──────────────────────────────────────────────
#  Hard Safety Override
# ──────────────────────────────────────────────

def enforce_safety(metrics: dict, proposed_valve: float) -> float:
    """
    Override AI decision if safety is at risk.

    Rules (ordered by priority):
      1. Pressure > 7.8  → force reduce valve by 10 %
      2. Pressure > 7.5  → block any valve increase
      3. Temperature > 590 → reduce valve by 5 %
      4. Always clamp valve to [0, 100]

    Args:
        metrics:        Current plant state from GET /metrics.
        proposed_valve: Valve position the AI wants to set.

    Returns:
        Safe valve position (may differ from proposed).
    """
    pressure = metrics["pressure"]
    temperature = metrics["temperature"]
    current_valve = metrics["valve_position"]
    valve = proposed_valve

    # ── RULE 1: Emergency pressure ──
    if pressure > PRESSURE_CRITICAL:
        valve = current_valve - 10.0
        logger.warning(
            "🚨 SAFETY OVERRIDE: pressure %.2f > %.1f → forcing valve %+.0f%%",
            pressure, PRESSURE_CRITICAL, valve - current_valve,
        )

    # ── RULE 2: High pressure — don't increase ──
    elif pressure > PRESSURE_SAFETY_LIMIT:
        if valve > current_valve:
            valve = current_valve
            logger.warning(
                "⚠️  SAFETY: pressure %.2f > %.1f → blocking increase",
                pressure, PRESSURE_SAFETY_LIMIT,
            )

    # ── RULE 3: Temperature too high ──
    if temperature > TEMPERATURE_CRITICAL:
        valve = min(valve, current_valve - 5.0)
        logger.warning(
            "⚠️  SAFETY: temp %.1f°C > %.0f → reducing valve",
            temperature, TEMPERATURE_CRITICAL,
        )

    # ── RULE 4: Hard clamp ──
    valve = max(MIN_VALVE, min(MAX_VALVE, valve))

    return round(valve, 2)


# ──────────────────────────────────────────────
#  Safety Status (for dashboard / frontend)
# ──────────────────────────────────────────────

def get_safety_status(metrics: dict) -> dict:
    """
    Return a safety assessment suitable for frontend display.

    Returns:
        {
            "safety_level": "NORMAL" | "WARNING" | "CRITICAL",
            "color":        "green"  | "orange"  | "red",
            "pressure_headroom": float (bar until 8.0),
            "temp_headroom":     float (°C until 600),
        }
    """
    pressure = metrics["pressure"]
    temp = metrics["temperature"]

    if pressure > PRESSURE_CRITICAL or temp > TEMPERATURE_CRITICAL:
        level, color = "CRITICAL", "red"
    elif pressure > PRESSURE_SAFETY_LIMIT or temp > 570:
        level, color = "WARNING", "orange"
    else:
        level, color = "NORMAL", "green"

    return {
        "safety_level": level,
        "color": color,
        "pressure_headroom": round(PRESSURE_HARD_LIMIT - pressure, 2),
        "temp_headroom": round(600.0 - temp, 1),
    }
