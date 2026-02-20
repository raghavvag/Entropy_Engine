"""
Entropy Engine — Model Confidence Monitor
===========================================
Tracks AI prediction quality in real-time using rolling error analysis.
Auto-disables AI when predictions become unreliable.
"""

from __future__ import annotations

from config import AI_CONFIDENCE_THRESHOLD
from logger import log_confidence_event


class ConfidenceMonitor:
    """
    Monitors AI prediction accuracy via rolling comparison
    of predicted vs actual power output.

    Confidence = 1 - (relative prediction error), averaged over a window.
    When rolling confidence drops below threshold → auto-disable AI.
    """

    WINDOW: int = 30           # rolling average size
    RECOVERY_WINDOW: int = 10  # consecutive good predictions to re-enable

    def __init__(self, threshold: float = AI_CONFIDENCE_THRESHOLD) -> None:
        self.threshold = threshold
        self.errors: list[float] = []
        self.confidence_history: list[float] = []
        self.disabled: bool = False
        self.disable_reason: str | None = None
        self._good_streak: int = 0

    def update(
        self,
        predicted_power: float | None,
        actual_power: float,
    ) -> float:
        """
        Record one prediction vs actual and return current confidence.

        Args:
            predicted_power: AI's power prediction (None if AI didn't predict).
            actual_power:    Actual observed power output.

        Returns:
            Rolling average confidence (0–1).
        """
        if predicted_power is None:
            return self._avg_confidence()

        error = abs(predicted_power - actual_power)
        relative_error = error / max(actual_power, 1.0)
        confidence = max(0.0, 1.0 - relative_error)

        self.errors.append(error)
        self.confidence_history.append(confidence)

        # Trim to window
        if len(self.confidence_history) > self.WINDOW:
            self.confidence_history = self.confidence_history[-self.WINDOW:]
            self.errors = self.errors[-self.WINDOW:]

        avg = self._avg_confidence()

        # ── Auto-disable check ──
        if avg < self.threshold and not self.disabled:
            self.disabled = True
            self.disable_reason = (
                f"Rolling confidence {avg:.3f} < {self.threshold}"
            )
            self._good_streak = 0
            log_confidence_event(avg, self.threshold, "AI AUTO-DISABLED")

        # ── Recovery check ──
        if self.disabled:
            if confidence >= self.threshold:
                self._good_streak += 1
                if self._good_streak >= self.RECOVERY_WINDOW:
                    self.disabled = False
                    self.disable_reason = None
                    self._good_streak = 0
                    log_confidence_event(avg, self.threshold, "AI RECOVERED")
            else:
                self._good_streak = 0

        return avg

    def should_use_ai(self) -> bool:
        """Return False if confidence is too low."""
        return not self.disabled

    def get_report(self) -> dict:
        """Return confidence stats for dashboard."""
        avg_conf = self._avg_confidence()
        avg_err = (
            sum(self.errors) / len(self.errors) if self.errors else 0.0
        )
        return {
            "confidence": round(float(avg_conf), 4),
            "avg_error_kw": round(float(avg_err), 2),
            "disabled": self.disabled,
            "disable_reason": self.disable_reason,
            "samples": len(self.confidence_history),
            "threshold": float(self.threshold),
        }

    def _avg_confidence(self) -> float:
        if not self.confidence_history:
            return 0.0
        return sum(self.confidence_history) / len(self.confidence_history)
