"""
Entropy Engine — AI Control Loop
==================================
Main entry point that ties everything together:
  1. Reads plant state (/metrics)
  2. Decides valve position (heuristic / MPC / hybrid)
  3. Applies safety enforcement
  4. Anti-oscillation clamp
  5. Sends command (/control)

Usage:
    python control_loop.py --mode heuristic            # rule-based only
    python control_loop.py --mode mpc                  # model-predictive
    python control_loop.py --mode hybrid               # MPC + fallback
    python control_loop.py --mode hybrid --duration 60  # run for 60 s
    python control_loop.py --collect                   # collect data first
    python control_loop.py --train                     # train model first
    python control_loop.py --collect --train --mode mpc # full pipeline
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time

import httpx

sys.path.insert(0, ".")
from config import (
    CONTROL_ENDPOINT,
    CONTROL_INTERVAL,
    MAX_VALVE,
    MAX_VALVE_CHANGE_PER_TICK,
    METRICS_ENDPOINT,
    MIN_VALVE,
    MODEL_SAVE_PATH,
)
from baseline_controller import compute_valve_heuristic
from safety import enforce_safety, get_safety_status

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ai-loop")


# ──────────────────────────────────────────────
#  Load MPC controller (lazy)
# ──────────────────────────────────────────────

def _load_mpc():
    """Import and instantiate MPC (only when needed)."""
    from mpc_controller import ModelPredictiveController
    try:
        mpc = ModelPredictiveController.from_checkpoint()
        logger.info("✅ MPC loaded from %s", MODEL_SAVE_PATH)
        return mpc
    except FileNotFoundError:
        logger.error("❌ No trained model at %s — run --train first", MODEL_SAVE_PATH)
        sys.exit(1)


# ──────────────────────────────────────────────
#  Main AI Control Loop
# ──────────────────────────────────────────────

async def run_ai_control(
    mode: str = "heuristic",
    duration: float | None = None,
) -> list[dict]:
    """
    Continuous AI control loop.

    Args:
        mode:     "heuristic" | "mpc" | "hybrid"
        duration: Seconds to run (None = forever).

    Returns:
        History of decision dicts.
    """
    logger.info("🧠 AI Control Loop starting — mode='%s'", mode)

    controller = None
    if mode in ("mpc", "hybrid"):
        controller = _load_mpc()

    history: list[dict] = []
    start = time.time()

    async with httpx.AsyncClient(timeout=5.0) as client:
        while True:
            try:
                # ── Step 1: Read plant state ──
                resp = await client.get(METRICS_ENDPOINT)
                resp.raise_for_status()
                metrics = resp.json()

                current_valve = metrics["valve_position"]
                decision_info: dict = {}

                # ── Step 2: Decide valve position ──
                if mode == "heuristic":
                    new_valve = compute_valve_heuristic(metrics, current_valve)
                    decision_info = {"optimal_valve": new_valve, "mode": "heuristic"}

                elif mode == "mpc" and controller:
                    decision_info = controller.find_optimal_valve(metrics)
                    decision_info["mode"] = "mpc"
                    new_valve = decision_info["optimal_valve"]

                elif mode == "hybrid" and controller:
                    decision_info = controller.find_optimal_valve(metrics)
                    new_valve = decision_info["optimal_valve"]

                    if decision_info.get("fallback", False):
                        new_valve = compute_valve_heuristic(metrics, current_valve)
                        decision_info["mode"] = "hybrid→heuristic"
                        logger.debug("Low confidence → falling back to heuristic")
                    else:
                        decision_info["mode"] = "hybrid→mpc"
                else:
                    new_valve = current_valve
                    decision_info = {"mode": "passthrough"}

                # ── Step 3: Safety enforcement ──
                new_valve = enforce_safety(metrics, new_valve)

                # ── Step 4: Anti-oscillation ──
                delta = new_valve - current_valve
                delta = max(-MAX_VALVE_CHANGE_PER_TICK, min(MAX_VALVE_CHANGE_PER_TICK, delta))
                new_valve = current_valve + delta
                new_valve = round(max(MIN_VALVE, min(MAX_VALVE, new_valve)), 2)

                # ── Step 5: Send control command ──
                await client.post(
                    CONTROL_ENDPOINT,
                    json={"valve_position": new_valve},
                )

                # ── Step 6: Log ──
                safety = get_safety_status(metrics)
                entry = {
                    **metrics,
                    "ai_valve": new_valve,
                    "mode": decision_info.get("mode", mode),
                    "predicted_power": decision_info.get("predicted_power"),
                    "confidence": decision_info.get("confidence"),
                    "safety_level": safety["safety_level"],
                }
                history.append(entry)

                logger.info(
                    "T=%6.1f°C  P=%5.2fbar  V=%5.1f→%5.1f%%  W=%6.1fkW  [%s] safety=%s",
                    metrics["temperature"],
                    metrics["pressure"],
                    current_valve,
                    new_valve,
                    metrics["power_output"],
                    decision_info.get("mode", mode),
                    safety["safety_level"],
                )

                # ── Duration check ──
                if duration and (time.time() - start) >= duration:
                    logger.info("⏱  Duration %.0fs reached — stopping.", duration)
                    break

                await asyncio.sleep(CONTROL_INTERVAL)

            except httpx.ConnectError:
                logger.error("Backend unreachable. Retrying in 3 s ...")
                await asyncio.sleep(3)
            except KeyboardInterrupt:
                logger.info("Interrupted by user.")
                break
            except Exception as exc:
                logger.error("Tick error: %s", exc, exc_info=True)
                await asyncio.sleep(CONTROL_INTERVAL)

    logger.info("🧠 AI Control Loop STOPPED — %d decisions made.", len(history))
    return history


# ──────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entropy Engine AI Control")
    parser.add_argument(
        "--mode",
        choices=["heuristic", "mpc", "hybrid"],
        default="heuristic",
        help="Control strategy",
    )
    parser.add_argument("--duration", type=float, default=None, help="Run for N seconds")
    parser.add_argument("--collect", action="store_true", help="Collect training data first")
    parser.add_argument("--train", action="store_true", help="Train model before running")
    args = parser.parse_args()

    # ── Optional pipeline steps ──
    if args.collect:
        from data_collector import collect_data
        asyncio.run(collect_data())

    if args.train:
        from train import train_model
        train_model(save_plot=True)

    # ── Run control loop ──
    asyncio.run(run_ai_control(mode=args.mode, duration=args.duration))
