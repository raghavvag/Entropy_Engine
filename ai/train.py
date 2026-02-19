"""
Entropy Engine — Model Training Script
========================================
Trains the PlantDynamicsModel on collected CSV data and saves the best
checkpoint (by validation loss) to disk.

Usage:
    python train.py                     # train with defaults (200 epochs)
    python train.py --epochs 50         # quick sanity run
    python train.py --plot              # save loss curves after training
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, ".")
from config import (
    BATCH_SIZE,
    DATA_FILE,
    DATA_LOSS_WEIGHT,
    EPOCHS,
    LEARNING_RATE,
    MODEL_SAVE_PATH,
    PHYSICS_LOSS_WEIGHT,
    SAFETY_LOSS_WEIGHT,
    TRAIN_SPLIT,
)
from model import PlantDataset, PlantDynamicsModel, load_trained_model
from pinn_loss import PINNLoss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


# ──────────────────────────────────────────────
#  Training loop
# ──────────────────────────────────────────────

def train_model(
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    save_plot: bool = False,
    use_pinn: bool = True,
) -> dict:
    """
    Train-validate loop.  Saves best model by val-loss.

    Returns:
        dict with training history and final metrics.
    """

    # ── Dataset ──
    dataset = PlantDataset(DATA_FILE)
    logger.info("Dataset: %d samples, features=%d", len(dataset), dataset.X.shape[1])

    train_size = int(TRAIN_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    logger.info("Split: %d train / %d val", train_size, val_size)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    # ── Model ──
    model = PlantDynamicsModel()
    optimiser = optim.Adam(model.parameters(), lr=lr)

    # ── Loss function (PINN or plain MSE) ──
    if use_pinn:
        criterion = PINNLoss(
            x_mean=torch.from_numpy(dataset.x_mean).float(),
            x_std=torch.from_numpy(dataset.x_std).float(),
            y_mean=dataset.y_mean,
            y_std=dataset.y_std,
            lambda_data=DATA_LOSS_WEIGHT,
            lambda_physics=PHYSICS_LOSS_WEIGHT,
            lambda_safety=SAFETY_LOSS_WEIGHT,
        )
        logger.info("Using PINN loss (λ_data=%.1f, λ_phys=%.2f, λ_safe=%.1f)",
                     DATA_LOSS_WEIGHT, PHYSICS_LOSS_WEIGHT, SAFETY_LOSS_WEIGHT)
    else:
        criterion = None  # will use plain MSE below
        mse_fn = nn.MSELoss()
        logger.info("Using plain MSE loss")
    logger.info("Model:\n%s", model)

    # ── Tracking ──
    best_val = float("inf")
    patience_counter = 0
    patience_limit = 30  # early-stopping patience
    train_losses: list[float] = []
    val_losses: list[float] = []

    start = time.time()

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        total_train = 0.0
        for xb, yb in train_loader:
            pred = model(xb)
            if use_pinn:
                loss, _ = criterion(pred, yb, xb)
            else:
                loss = mse_fn(pred, yb)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total_train += loss.item()
        avg_train = total_train / len(train_loader)

        # ── Validate ──
        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                if use_pinn:
                    loss, breakdown = criterion(pred, yb, xb)
                else:
                    loss = mse_fn(pred, yb)
                total_val += loss.item()
        avg_val = total_val / len(val_loader)

        train_losses.append(avg_train)
        val_losses.append(avg_val)

        # ── Logging ──
        if epoch % 20 == 0 or epoch == 1:
            logger.info(
                "Epoch %3d/%d  |  Train %.6f  |  Val %.6f  %s",
                epoch, epochs, avg_train, avg_val,
                "★" if avg_val < best_val else "",
            )

        # ── Checkpointing ──
        if avg_val < best_val:
            best_val = avg_val
            patience_counter = 0
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "x_mean": dataset.x_mean,
                    "x_std": dataset.x_std,
                    "y_mean": dataset.y_mean,
                    "y_std": dataset.y_std,
                    "epoch": epoch,
                    "val_loss": best_val,
                },
                MODEL_SAVE_PATH,
            )
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch, patience_limit)
                break

    elapsed = time.time() - start
    logger.info(
        "✅ Training complete in %.1f s — best val loss: %.6f",
        elapsed, best_val,
    )
    logger.info("💾 Model saved to %s", MODEL_SAVE_PATH)

    # ── Optional plot ──
    if save_plot:
        _save_loss_plot(train_losses, val_losses)

    # ── Quick inference test ──
    _inference_sanity_check()

    return {
        "best_val_loss": best_val,
        "final_train_loss": train_losses[-1],
        "epochs_trained": len(train_losses),
        "elapsed_s": elapsed,
    }


# ──────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────

def _save_loss_plot(train_losses: list, val_losses: list) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(train_losses, label="Train Loss", linewidth=0.8)
        ax.plot(val_losses, label="Val Loss", linewidth=0.8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss (normalised)")
        ax.set_title("Training Progress")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_path = "ai/models/loss_curve.png"
        plt.savefig(plot_path, dpi=120)
        plt.close()
        logger.info("📈 Loss curve saved → %s", plot_path)
    except ImportError:
        logger.warning("matplotlib not installed — skipping plot")


def _inference_sanity_check() -> None:
    """Load best checkpoint and run one prediction to verify."""
    try:
        model, norm = load_trained_model()
        sample_state = {
            "temperature": 500.0,
            "pressure": 5.0,
            "flow_rate": 3.0,
            "valve_position": 60.0,
            "power_output": 200.0,
        }
        from model import predict_power
        pred = predict_power(model, norm, sample_state)
        logger.info(
            "🧪 Sanity check — sample state → predicted power: %.2f kW", pred
        )
    except Exception as exc:
        logger.error("Sanity check failed: %s", exc)


# ──────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train plant dynamics model")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Training epochs")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--plot", action="store_true", help="Save loss curve plot")
    parser.add_argument("--no-pinn", action="store_true", help="Use plain MSE instead of PINN loss")
    args = parser.parse_args()

    result = train_model(epochs=args.epochs, lr=args.lr, save_plot=args.plot, use_pinn=not args.no_pinn)
    print(f"\n{'='*50}")
    print(f"  RESULT: {result}")
    print(f"{'='*50}")
