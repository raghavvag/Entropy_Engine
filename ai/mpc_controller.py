"""
Entropy Engine — Model Predictive Controller (MPC)
===================================================
Uses the trained PINN model to evaluate N candidate valve positions
and selects the one that maximises predicted power while respecting
pressure safety constraints.

Usage:
    from mpc_controller import ModelPredictiveController
    mpc = ModelPredictiveController.from_checkpoint()
    result = mpc.find_optimal_valve(metrics_dict)
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from config import (
    MAX_VALVE,
    MAX_VALVE_CHANGE_PER_TICK,
    MIN_VALVE,
    MODEL_SAVE_PATH,
    MPC_CONFIDENCE_THRESHOLD,
    MPC_NUM_CANDIDATES,
    MPC_SEARCH_RANGE,
    PRESSURE_COEFFICIENT,
    PRESSURE_SAFETY_LIMIT,
)
from model import FEATURE_COLS, PlantDynamicsModel, load_trained_model

logger = logging.getLogger("mpc")


class ModelPredictiveController:
    """
    Model Predictive Control — searches candidate valve positions
    and picks the one that maximises predicted power while staying safe.
    """

    def __init__(
        self,
        model: PlantDynamicsModel,
        norm: dict,
        num_candidates: int = MPC_NUM_CANDIDATES,
        search_range: float = MPC_SEARCH_RANGE,
    ):
        self.model = model
        self.model.eval()
        self.x_mean = norm["x_mean"]
        self.x_std = norm["x_std"]
        self.y_mean = norm["y_mean"]
        self.y_std = norm["y_std"]
        self.num_candidates = num_candidates
        self.search_range = search_range

    # ── Factory ──────────────────────────────

    @classmethod
    def from_checkpoint(cls, path: str = MODEL_SAVE_PATH, **kwargs) -> "ModelPredictiveController":
        """Load model from disk and return a ready-to-use MPC."""
        model, norm = load_trained_model(path)
        return cls(model, norm, **kwargs)

    # ── Core prediction ──────────────────────

    def predict_power(self, state: dict, candidate_valve: float) -> float:
        """
        Predict next-step power if we hypothetically set valve to *candidate_valve*.
        """
        x = np.array(
            [
                state["temperature"],
                state["pressure"],
                state["flow_rate"],
                candidate_valve,
                state["power_output"],
            ],
            dtype=np.float32,
        )
        x_norm = (x - self.x_mean) / self.x_std
        x_tensor = torch.from_numpy(x_norm).unsqueeze(0)

        with torch.no_grad():
            y_norm = self.model(x_tensor).item()

        return y_norm * self.y_std + self.y_mean

    # ── Candidate search ─────────────────────

    def find_optimal_valve(self, state: dict) -> dict:
        """
        Evaluate N candidates around the current valve and select the best.

        Returns:
            {
                "optimal_valve":       float,
                "predicted_power":     float,
                "confidence":          float,   # 0–1, improvement fraction
                "safe":                bool,
                "candidates_evaluated": int,
                "fallback":            bool,    # True if confidence too low
            }
        """
        current_valve = state["valve_position"]
        current_pressure = state["pressure"]

        # ── Generate candidates around current position ──
        lo = max(MIN_VALVE, current_valve - self.search_range)
        hi = min(MAX_VALVE, current_valve + self.search_range)
        candidates = np.linspace(lo, hi, self.num_candidates)

        best_valve = current_valve
        best_power = -float("inf")

        for valve in candidates:
            pred = self.predict_power(state, valve)

            # Estimate pressure at this valve setting
            eff_flow = state["flow_rate"] * (valve / 100.0)
            est_pressure = PRESSURE_COEFFICIENT * state["temperature"] * eff_flow

            # Skip unsafe candidates
            if est_pressure > PRESSURE_SAFETY_LIMIT:
                continue

            if pred > best_power:
                best_power = pred
                best_valve = float(valve)

        # ── Anti-oscillation: clamp delta ──
        delta = best_valve - current_valve
        delta = max(-MAX_VALVE_CHANGE_PER_TICK, min(MAX_VALVE_CHANGE_PER_TICK, delta))
        best_valve = current_valve + delta
        best_valve = max(MIN_VALVE, min(MAX_VALVE, best_valve))

        # ── Confidence ──
        current_power = state["power_output"]
        improvement = (best_power - current_power) / max(current_power, 1.0)
        confidence = round(min(abs(improvement), 1.0), 4)
        fallback = confidence < MPC_CONFIDENCE_THRESHOLD

        return {
            "optimal_valve": round(best_valve, 2),
            "predicted_power": round(best_power, 2),
            "confidence": confidence,
            "safe": current_pressure < PRESSURE_SAFETY_LIMIT,
            "candidates_evaluated": self.num_candidates,
            "fallback": fallback,
        }


# ──────────────────────────────────────────────
#  Quick smoke test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    mpc = ModelPredictiveController.from_checkpoint()
    logger.info("MPC loaded from %s", MODEL_SAVE_PATH)

    sample = {
        "temperature": 500.0,
        "pressure": 5.0,
        "flow_rate": 3.0,
        "valve_position": 50.0,
        "power_output": 200.0,
    }
    result = mpc.find_optimal_valve(sample)
    print(f"\nSample state: {sample}")
    print(f"MPC result:   {result}")
