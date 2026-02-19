"""
Entropy Engine — Data Collection Pipeline
==========================================
Polls /metrics with random valve perturbations to explore the
operating envelope.  Saves diverse training data for the PyTorch model.

Usage:
    python data_collector.py                        # 1500 samples (~25 min)
    python data_collector.py --samples 200          # quick run (~3.3 min)
    python data_collector.py --samples 200 --fast   # turbo (~40 s)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import random
import sys
import time

import httpx
import pandas as pd

# ── Resolve imports whether run from ai/ or project root ──
sys.path.insert(0, ".")
from config import (
    COLLECTION_INTERVAL,
    CONTROL_ENDPOINT,
    DATA_DIR,
    DATA_FILE,
    METRICS_ENDPOINT,
    MIN_TRAINING_SAMPLES,
    VALVE_PERTURBATION_INTERVAL,
    VALVE_PERTURBATION_MAX,
    VALVE_PERTURBATION_MIN,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("data-collector")


# ──────────────────────────────────────────────
#  Core collection routine
# ──────────────────────────────────────────────

async def collect_data(
    num_samples: int = 1500,
    interval: float = COLLECTION_INTERVAL,
    fast: bool = False,
) -> pd.DataFrame:
    """
    Poll GET /metrics, apply random valve perturbations, and save rows.

    Args:
        num_samples: Total rows to collect.
        interval:    Seconds between reads (ignored if *fast*).
        fast:        If True, use 0.2 s interval instead of 1.0 s.

    Returns:
        DataFrame with all collected samples.
    """
    effective_interval = 0.2 if fast else interval
    est_minutes = (num_samples * effective_interval) / 60
    logger.info(
        "📡 Starting data collection: %d samples, %.1fs interval (~%.1f min)",
        num_samples, effective_interval, est_minutes,
    )

    data: list[dict] = []
    start = time.time()

    async with httpx.AsyncClient(timeout=5.0) as client:
        for i in range(num_samples):
            try:
                # ── Read plant state ──
                resp = await client.get(METRICS_ENDPOINT)
                resp.raise_for_status()
                row = resp.json()
                data.append(row)

                # ── Random valve perturbation to explore state space ──
                if i % VALVE_PERTURBATION_INTERVAL == 0:
                    random_valve = round(
                        random.uniform(VALVE_PERTURBATION_MIN, VALVE_PERTURBATION_MAX),
                        2,
                    )
                    await client.post(
                        CONTROL_ENDPOINT,
                        json={"valve_position": random_valve},
                    )
                    logger.info(
                        "[%d/%d] 🎲 Valve perturbation → %.1f%%",
                        i, num_samples, random_valve,
                    )

                # ── Progress log every 100 samples ──
                if i > 0 and i % 100 == 0:
                    elapsed = time.time() - start
                    remaining = elapsed / i * (num_samples - i)
                    logger.info(
                        "[%d/%d] collected — %.0fs elapsed, ~%.0fs remaining",
                        i, num_samples, elapsed, remaining,
                    )

                await asyncio.sleep(effective_interval)

            except httpx.ConnectError:
                logger.error("Backend unreachable — retrying in 3 s ...")
                await asyncio.sleep(3)
            except Exception as exc:
                logger.error("Sample %d error: %s", i, exc, exc_info=True)
                await asyncio.sleep(effective_interval)

    elapsed = time.time() - start
    logger.info(
        "✅ Collection finished — %d samples in %.1f s",
        len(data), elapsed,
    )

    # ── Build DataFrame and save ──
    df = pd.DataFrame(data)
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(DATA_FILE, index=False)
    logger.info("💾 Saved to %s (%d rows × %d cols)", DATA_FILE, len(df), len(df.columns))

    return df


# ──────────────────────────────────────────────
#  Data exploration (post-collection)
# ──────────────────────────────────────────────

def explore_data(df: pd.DataFrame) -> None:
    """Print stats and save exploration plots."""
    print("\n" + "=" * 60)
    print("  DATA EXPLORATION")
    print("=" * 60)

    print("\n📊 Descriptive statistics:")
    print(df.describe().to_string())

    # ── Correlations with power ──
    numeric = df.select_dtypes(include="number")
    if "power_output" in numeric.columns:
        print("\n🔗 Correlation with power_output:")
        corr = numeric.corr()["power_output"].sort_values(ascending=False)
        for col, val in corr.items():
            arrow = "↑" if val > 0 else "↓"
            print(f"  {arrow}  {col:20s}  {val:+.4f}")

    # ── Valve coverage ──
    valve_min = df["valve_position"].min()
    valve_max = df["valve_position"].max()
    print(f"\n🔧 Valve coverage: {valve_min:.1f}% – {valve_max:.1f}%")

    # ── NaN check ──
    nans = df.isna().sum().sum()
    print(f"🕳  Missing values: {nans}")

    # ── Plots ──
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt

        cols = ["temperature", "pressure", "flow_rate", "valve_position", "power_output"]
        fig, axes = plt.subplots(len(cols), 1, figsize=(14, 12), sharex=True)
        for ax, col in zip(axes, cols):
            ax.plot(df[col], linewidth=0.7)
            ax.set_ylabel(col, fontsize=9)
            ax.grid(True, alpha=0.3)
        axes[0].set_title("Training Data — Time Series", fontsize=12)
        axes[-1].set_xlabel("Sample #")
        plt.tight_layout()

        plot_path = os.path.join(DATA_DIR, "exploration.png")
        plt.savefig(plot_path, dpi=120)
        logger.info("📈 Saved exploration plot → %s", plot_path)
        plt.close()

    except ImportError:
        logger.warning("matplotlib not available — skipping plots")

    print("=" * 60 + "\n")


# ──────────────────────────────────────────────
#  CLI entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect training data from plant simulation")
    parser.add_argument("--samples", type=int, default=1500, help="Number of samples to collect")
    parser.add_argument("--fast", action="store_true", help="Use 0.2s interval instead of 1.0s")
    parser.add_argument("--explore-only", action="store_true", help="Just run exploration on existing CSV")
    args = parser.parse_args()

    if args.explore_only:
        if not os.path.exists(DATA_FILE):
            logger.error("No data file found at %s", DATA_FILE)
            sys.exit(1)
        df = pd.read_csv(DATA_FILE)
        explore_data(df)
    else:
        df = asyncio.run(collect_data(num_samples=args.samples, fast=args.fast))
        explore_data(df)
