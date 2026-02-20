"""
Entropy Engine — Enhanced Safety Fallback System
==================================================
Production-grade failsafe layer sitting between AI and simulation.
Cannot be bypassed. Tracks all overrides for dashboard display.

Rules (priority order):
  1. Pressure > 7.8 bar  → force reduce valve by 10%  (CRITICAL)
  2. Pressure > 7.5 bar  → block any valve increase    (WARNING)
  3. Temperature > 590°C → reduce valve by 5%          (WARNING)
  4. Always clamp valve to [0, 100]
"""

from __future__ import annotations

from config import (
    MAX_VALVE,
    MIN_VALVE,
    PRESSURE_CRITICAL,
    PRESSURE_HARD_LIMIT,
    PRESSURE_WARNING,
    TEMPERATURE_CRITICAL,
)
from logger import log_safety_event


class SafetyFallback:
    """
    Production safety layer.
    Sits between AI decision and simulation — cannot be bypassed.
    Tracks override statistics for frontend display.
    """

    def __init__(self) -> None:
        self.total_overrides: int = 0
        self.consecutive_overrides: int = 0
        self.last_override_reason: str | None = None
        self._override_counts: dict[str, int] = {
            "PRESSURE_CRITICAL": 0,
            "PRESSURE_HIGH": 0,
            "TEMP_CRITICAL": 0,
        }

    def check(
        self,
        metrics: dict,
        proposed_valve: float,
    ) -> tuple[float, dict]:
        """
        Validate proposed valve against safety constraints.

        Args:
            metrics:        Current plant state from GET /metrics.
            proposed_valve: Valve position the AI wants to set.

        Returns:
            (safe_valve, safety_report)
        """
        pressure = metrics.get("pressure", 0)
        temp = metrics.get("temperature", 0)
        current = metrics.get("valve_position", 50)
        valve = proposed_valve
        overridden = False
        reason = None
        level = "NORMAL"

        # ── RULE 1: Pressure emergency ──
        if pressure > PRESSURE_CRITICAL:
            valve = max(0, current - 10)
            overridden = True
            reason = "PRESSURE_CRITICAL"
            level = "CRITICAL"
            log_safety_event(
                "CRITICAL",
                f"Pressure {pressure:.2f}bar > {PRESSURE_CRITICAL} "
                f"→ forcing valve {current:.0f}→{valve:.0f}%",
            )

        # ── RULE 2: Pressure high — block increase ──
        elif pressure > PRESSURE_WARNING:
            if valve > current:
                valve = current
                overridden = True
                reason = "PRESSURE_HIGH"
                level = "WARNING"
                log_safety_event(
                    "WARNING",
                    f"Pressure {pressure:.2f}bar > {PRESSURE_WARNING} "
                    f"→ blocking valve increase",
                )

        # ── RULE 3: Temperature critical ──
        if temp > TEMPERATURE_CRITICAL:
            valve = min(valve, current - 5)
            if not overridden:
                overridden = True
                reason = "TEMP_CRITICAL"
                level = "WARNING"
            log_safety_event(
                "WARNING",
                f"Temp {temp:.1f}°C > {TEMPERATURE_CRITICAL} → reducing valve",
            )

        # ── RULE 4: Hard clamp ──
        valve = max(MIN_VALVE, min(MAX_VALVE, round(valve, 2)))

        # ── Track stats ──
        if overridden:
            self.total_overrides += 1
            self.consecutive_overrides += 1
            self.last_override_reason = reason
            if reason in self._override_counts:
                self._override_counts[reason] += 1
        else:
            self.consecutive_overrides = 0

        report = {
            "original_valve": round(proposed_valve, 2),
            "final_valve": valve,
            "overridden": overridden,
            "reason": reason,
            "level": level,
            "pressure_headroom": round(PRESSURE_HARD_LIMIT - pressure, 2),
            "temp_headroom": round(600.0 - temp, 1),
        }
        return valve, report

    def get_stats(self) -> dict:
        """Return override statistics for dashboard."""
        return {
            "total_overrides": self.total_overrides,
            "consecutive_overrides": self.consecutive_overrides,
            "last_reason": self.last_override_reason,
            "breakdown": dict(self._override_counts),
        }

    def get_safety_status(self, metrics: dict) -> dict:
        """
        Quick safety assessment for frontend display.

        Returns:
            {"safety_level": ..., "color": ..., "pressure_headroom": ..., "temp_headroom": ...}
        """
        pressure = metrics.get("pressure", 0)
        temp = metrics.get("temperature", 0)

        if pressure > PRESSURE_CRITICAL or temp > TEMPERATURE_CRITICAL:
            level, color = "CRITICAL", "red"
        elif pressure > PRESSURE_WARNING or temp > 570:
            level, color = "WARNING", "orange"
        else:
            level, color = "NORMAL", "green"

        return {
            "safety_level": level,
            "color": color,
            "pressure_headroom": round(PRESSURE_HARD_LIMIT - pressure, 2),
            "temp_headroom": round(600.0 - temp, 1),
        }
