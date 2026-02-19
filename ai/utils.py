"""
Entropy Engine — AI Metrics & Confidence Reporting
====================================================
Tracks AI decision quality, power improvement, prediction accuracy,
and safety violations.  Exposes a JSON-ready summary for
the frontend dashboard.

Usage:
    from utils import AIReport
    report = AIReport()
    report.record(metrics_dict, decision_dict)
    summary = report.get_summary()
"""

from __future__ import annotations

import logging
import sys

import numpy as np

sys.path.insert(0, ".")
from safety import get_safety_status

logger = logging.getLogger("ai-report")


class AIReport:
    """
    Accumulates per-tick records and provides aggregate metrics
    for the frontend/dashboard.
    """

    def __init__(self, power_before_ai: float | None = None):
        self.history: list[dict] = []
        self.power_before_ai = power_before_ai  # baseline average kW

    # ── Recording ────────────────────────────

    def record(self, metrics: dict, ai_decision: dict) -> None:
        """
        Append one tick's worth of data.

        Args:
            metrics:     Raw plant state from GET /metrics.
            ai_decision: Dict from MPC or heuristic with at least "optimal_valve".
        """
        safety = get_safety_status(metrics)
        self.history.append({
            "timestamp": metrics.get("timestamp"),
            "actual_power": metrics["power_output"],
            "predicted_power": ai_decision.get("predicted_power"),
            "valve_sent": ai_decision.get("optimal_valve"),
            "confidence": ai_decision.get("confidence"),
            "mode": ai_decision.get("mode", "unknown"),
            "safety_level": safety["safety_level"],
        })

    # ── Summary ──────────────────────────────

    def get_summary(self, window: int = 30) -> dict:
        """
        Return dashboard-ready JSON summary.

        Args:
            window: Number of most-recent ticks to average over.

        Returns:
            {
                "avg_power_kw":           float,
                "prediction_error_kw":    float | None,
                "power_improvement_pct":  float | None,
                "total_decisions":        int,
                "safety_violations":      int,
                "avg_confidence":         float | None,
                "status":                 str,
            }
        """
        if len(self.history) < 5:
            return {"status": "collecting_baseline", "total_decisions": len(self.history)}

        recent = self.history[-window:]

        # ── Average power ──
        actual_powers = [r["actual_power"] for r in recent]
        avg_power = float(np.mean(actual_powers))

        # ── Prediction accuracy ──
        pred_errors = [
            abs(r["predicted_power"] - r["actual_power"])
            for r in recent
            if r["predicted_power"] is not None
        ]
        avg_error = float(np.mean(pred_errors)) if pred_errors else None

        # ── Improvement over baseline ──
        improvement = None
        if self.power_before_ai and self.power_before_ai > 0:
            improvement = ((avg_power - self.power_before_ai)
                           / self.power_before_ai * 100)

        # ── Confidence ──
        confs = [r["confidence"] for r in recent if r["confidence"] is not None]
        avg_conf = float(np.mean(confs)) if confs else None

        # ── Safety ──
        safety_violations = sum(
            1 for r in self.history if r["safety_level"] == "CRITICAL"
        )

        return {
            "avg_power_kw": round(avg_power, 1),
            "prediction_error_kw": round(avg_error, 2) if avg_error is not None else None,
            "power_improvement_pct": round(improvement, 1) if improvement is not None else None,
            "total_decisions": len(self.history),
            "safety_violations": safety_violations,
            "avg_confidence": round(avg_conf, 4) if avg_conf is not None else None,
            "status": "active",
        }

    # ── Convenience ──────────────────────────

    def __repr__(self) -> str:
        summary = self.get_summary()
        return f"AIReport({summary})"
