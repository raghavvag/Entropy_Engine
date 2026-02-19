"""
Entropy Engine — PyTorch Plant Dynamics Model
===============================================
Feedforward neural network that predicts next-step power output
given the current plant state vector.

Architecture:  [temp, pressure, flow, valve, power] → 64 → 64 → 32 → 1
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

sys.path.insert(0, ".")
from config import (
    BATCH_SIZE,
    DATA_FILE,
    DROPOUT,
    HIDDEN_DIM,
    INPUT_DIM,
    MODEL_SAVE_PATH,
    OUTPUT_DIM,
)

# ──────────────────────────────────────────────
#  Dataset
# ──────────────────────────────────────────────

FEATURE_COLS = ["temperature", "pressure", "flow_rate", "valve_position", "power_output"]


class PlantDataset(Dataset):
    """
    Supervised dataset: (state_t) → power_{t+1}.

    Normalises using per-feature mean/std so the network trains stably.
    Stores normalisation stats so they can be saved alongside the model.
    """

    def __init__(self, csv_path: str = DATA_FILE, lookback: int = 1):
        df = pd.read_csv(csv_path)

        X_raw = df[FEATURE_COLS].values[:-lookback].astype(np.float32)
        y_raw = df["power_output"].values[lookback:].astype(np.float32)

        # ── Normalisation stats ──
        self.x_mean = X_raw.mean(axis=0)
        self.x_std = X_raw.std(axis=0) + 1e-8
        self.y_mean = y_raw.mean()
        self.y_std = float(y_raw.std() + 1e-8)

        # ── Normalise ──
        self.X = (X_raw - self.x_mean) / self.x_std
        self.y = (y_raw - self.y_mean) / self.y_std

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.X[idx]),
            torch.tensor([self.y[idx]], dtype=torch.float32),
        )


# ──────────────────────────────────────────────
#  Network
# ──────────────────────────────────────────────

class PlantDynamicsModel(nn.Module):
    """
    Simple feed-forward predictor.

    Input  (5): [temperature, pressure, flow_rate, valve_position, power_output]
    Output (1): predicted power at next time-step
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        output_dim: int = OUTPUT_DIM,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),

            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────
#  Helper: load a trained checkpoint
# ──────────────────────────────────────────────

def load_trained_model(path: str = MODEL_SAVE_PATH) -> tuple[PlantDynamicsModel, dict]:
    """
    Load model weights + normalisation stats from a saved checkpoint.

    Returns:
        (model, norm_stats) where norm_stats contains x_mean, x_std, y_mean, y_std.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model checkpoint at {path}")

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = PlantDynamicsModel()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    norm = {
        "x_mean": ckpt["x_mean"],
        "x_std": ckpt["x_std"],
        "y_mean": ckpt["y_mean"],
        "y_std": ckpt["y_std"],
    }
    return model, norm


def predict_power(
    model: PlantDynamicsModel,
    norm: dict,
    state: dict,
) -> float:
    """
    Predict next-step power from a raw (un-normalised) plant state dict.
    """
    x = np.array(
        [state[c] for c in FEATURE_COLS],
        dtype=np.float32,
    )
    x_norm = (x - norm["x_mean"]) / norm["x_std"]
    x_t = torch.from_numpy(x_norm).unsqueeze(0)  # (1, 5)

    model.eval()
    with torch.no_grad():
        y_norm = model(x_t).item()

    return y_norm * norm["y_std"] + norm["y_mean"]
