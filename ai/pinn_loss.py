"""
Entropy Engine — Physics-Informed Neural Network Loss
======================================================
Adds physics constraints and safety penalties to the standard MSE loss,
making the model respect known thermodynamic laws.

PINN total loss:
    L = λ_data  × MSE(pred, target)
      + λ_physics × physics_residual²
      + λ_safety  × safety_penalty
"""

from __future__ import annotations

import sys

import torch
import torch.nn as nn

sys.path.insert(0, ".")
from config import (
    DATA_LOSS_WEIGHT,
    PHYSICS_LOSS_WEIGHT,
    PRESSURE_COEFFICIENT,
    PRESSURE_SAFETY_LIMIT,
    SAFETY_LOSS_WEIGHT,
    TURBINE_EFFICIENCY,
)


# ──────────────────────────────────────────────
#  Physics Residual
# ──────────────────────────────────────────────

def physics_residual(
    state: torch.Tensor,
    predicted_power: torch.Tensor,
    pressure_coeff: float = PRESSURE_COEFFICIENT,
    turbine_eff: float = TURBINE_EFFICIENCY,
) -> torch.Tensor:
    """
    Measure how far the predicted power deviates from the physics equations.

    From Person 1's simulation:
        effective_flow = flow_rate × (valve / 100)
        pressure       = c1 × temperature × effective_flow
        power          = η  × pressure × effective_flow

    Args:
        state:  (B, 5) normalised tensor — but we need *un-normalised* values,
                so the caller must pass the raw (denormalised) state.
        predicted_power: (B, 1) denormalised predicted power.

    Returns:
        Scalar mean squared residual.
    """
    temp  = state[:, 0]   # temperature  (°C)
    # pressure - state[:, 1] is not used in forward physics
    flow  = state[:, 2]   # flow_rate    (kg/s)
    valve = state[:, 3]   # valve_position (%)

    effective_flow = flow * (valve / 100.0)
    expected_pressure = pressure_coeff * temp * effective_flow
    expected_power = turbine_eff * expected_pressure * effective_flow

    residual = (predicted_power.squeeze() - expected_power) ** 2
    return residual.mean()


# ──────────────────────────────────────────────
#  Safety Penalty
# ──────────────────────────────────────────────

def safety_penalty(
    state: torch.Tensor,
    limit: float = PRESSURE_SAFETY_LIMIT,
) -> torch.Tensor:
    """
    Penalise states where pressure exceeds the safety limit.

    Uses soft ReLU so the gradient is smooth and differentiable.

    Args:
        state: (B, 5) **raw** state tensor (denormalised).
        limit: Pressure bar threshold.

    Returns:
        Scalar mean squared violation.
    """
    pressure = state[:, 1]
    violation = torch.relu(pressure - limit)
    return (violation ** 2).mean()


# ──────────────────────────────────────────────
#  Combined PINN Loss
# ──────────────────────────────────────────────

class PINNLoss(nn.Module):
    """
    Composite loss that blends data-fit, physics-residual, and safety-penalty.

    The model receives **normalised** inputs during forward pass, so we must
    denormalise before computing physics terms.  The caller should pass
    normalisation stats (x_mean, x_std, y_mean, y_std) at construction.
    """

    def __init__(
        self,
        x_mean: torch.Tensor,
        x_std: torch.Tensor,
        y_mean: float,
        y_std: float,
        lambda_data: float = DATA_LOSS_WEIGHT,
        lambda_physics: float = PHYSICS_LOSS_WEIGHT,
        lambda_safety: float = SAFETY_LOSS_WEIGHT,
    ):
        super().__init__()
        self.register_buffer("x_mean", x_mean)
        self.register_buffer("x_std", x_std)
        self.y_mean = y_mean
        self.y_std = y_std

        self.lambda_data = lambda_data
        self.lambda_physics = lambda_physics
        self.lambda_safety = lambda_safety
        self.mse = nn.MSELoss()

    def forward(
        self,
        pred_norm: torch.Tensor,
        target_norm: torch.Tensor,
        state_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            pred_norm:   (B, 1) normalised predicted power.
            target_norm: (B, 1) normalised actual power.
            state_norm:  (B, 5) normalised state features.

        Returns:
            (total_loss, breakdown_dict)
        """
        # ── Data loss (in normalised space) ──
        data_loss = self.mse(pred_norm, target_norm)

        # ── Denormalise for physics/safety ──
        state_raw = state_norm * self.x_std + self.x_mean
        pred_raw = pred_norm * self.y_std + self.y_mean

        # ── Physics residual ──
        phys_loss = physics_residual(state_raw, pred_raw)
        # Scale physics loss to be comparable magnitude to normalised MSE
        phys_loss_scaled = phys_loss / (self.y_std ** 2 + 1e-8)

        # ── Safety penalty ──
        safe_loss = safety_penalty(state_raw)

        # ── Combined ──
        total = (
            self.lambda_data * data_loss
            + self.lambda_physics * phys_loss_scaled
            + self.lambda_safety * safe_loss
        )

        breakdown = {
            "data_loss": data_loss.item(),
            "physics_loss": phys_loss_scaled.item(),
            "safety_loss": safe_loss.item(),
            "total_loss": total.item(),
        }
        return total, breakdown
