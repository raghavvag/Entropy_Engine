"""
Entropy Engine — AI Bridge
============================
Connects the orchestrator to Person 2's trained PINN + MPC controller.
Provides a clean interface that the orchestrator calls without
needing to know AI internals.

Also provides a heuristic fallback when MPC is unavailable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from config import AI_MODEL_PATH, MAX_VALVE_CHANGE_PER_TICK, MIN_VALVE, MAX_VALVE
from logger import log_ai


def _to_python(val):
    """Convert numpy scalars to native Python types for JSON serialisation."""
    if isinstance(val, (np.floating, np.integer)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


def _sanitize(d: dict) -> dict:
    """Recursively convert numpy types inside a dict."""
    return {k: _to_python(v) for k, v in d.items()}

# ── Path to AI module — added lazily to avoid import collisions ──
AI_DIR = Path(__file__).resolve().parent.parent / "ai"

_mpc = None
_mpc_load_error: str | None = None


def _load_mpc():
    """Lazy-load the MPC controller from Person 2's checkpoint."""
    global _mpc, _mpc_load_error
    if _mpc is not None:
        return _mpc
    try:
        # Add AI dir to path so Python can find mpc_controller, model, etc.
        ai_dir_str = str(AI_DIR)
        if ai_dir_str not in sys.path:
            sys.path.insert(0, ai_dir_str)

        from mpc_controller import ModelPredictiveController

        model_path = AI_MODEL_PATH
        _mpc = ModelPredictiveController.from_checkpoint(model_path)
        log_ai.info("✅ MPC loaded from %s", model_path)
        return _mpc
    except Exception as e:
        _mpc_load_error = str(e)
        log_ai.error("❌ Failed to load MPC: %s", e)
        return None


def get_ai_decision(metrics: dict) -> dict:
    """
    Get AI's optimal valve decision for the current plant state.

    Returns:
        {
            "valve":            float,  # recommended valve position
            "predicted_power":  float | None,
            "confidence":       float,
            "mode":             str,    # "mpc" | "heuristic"
            "fallback":         bool,
        }
    """
    mpc = _load_mpc()

    if mpc is None:
        # MPC unavailable — use heuristic
        valve = _heuristic_fallback(metrics)
        return _sanitize({
            "valve": valve,
            "predicted_power": None,
            "confidence": 0.0,
            "mode": "heuristic",
            "fallback": True,
        })

    try:
        result = mpc.find_optimal_valve(metrics)

        if result.get("fallback", False):
            # MPC low confidence — use heuristic
            valve = _heuristic_fallback(metrics)
            return _sanitize({
                "valve": valve,
                "predicted_power": result.get("predicted_power"),
                "confidence": result.get("confidence", 0),
                "mode": "hybrid→heuristic",
                "fallback": True,
            })

        return _sanitize({
            "valve": result["optimal_valve"],
            "predicted_power": result.get("predicted_power"),
            "confidence": result.get("confidence", 0),
            "mode": "mpc",
            "fallback": False,
        })

    except Exception as e:
        log_ai.error("MPC error during decision: %s", e)
        valve = _heuristic_fallback(metrics)
        return _sanitize({
            "valve": valve,
            "predicted_power": None,
            "confidence": 0.0,
            "mode": "heuristic",
            "fallback": True,
        })


def is_ai_loaded() -> bool:
    """Check whether the MPC model is loaded."""
    return _mpc is not None


def get_load_error() -> str | None:
    """Return the last MPC load error, if any."""
    return _mpc_load_error


def _heuristic_fallback(metrics: dict) -> float:
    """
    Simple rule-based valve controller (mirrors Person 2's baseline).
    Used when MPC is unavailable or has low confidence.
    """
    temp = metrics.get("temperature", 500)
    pressure = metrics.get("pressure", 5)
    current_valve = metrics.get("valve_position", 50)
    valve = current_valve

    # Pressure safety
    if pressure > 7.8:
        valve -= 8.0
    elif pressure > 7.5:
        valve -= 3.0
    # Temperature optimization
    elif temp > 590:
        valve -= 5.0
    elif temp > 560:
        valve -= 3.0
    elif temp > 520:
        valve += 1.0
    elif temp > 480:
        valve += 3.0
    elif temp < 440:
        valve -= 3.0

    # Anti-oscillation clamp
    delta = valve - current_valve
    delta = max(-MAX_VALVE_CHANGE_PER_TICK, min(MAX_VALVE_CHANGE_PER_TICK, delta))
    valve = current_valve + delta
    valve = max(MIN_VALVE, min(MAX_VALVE, round(valve, 2)))

    return valve
