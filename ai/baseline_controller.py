"""
Entropy Engine — Heuristic Baseline Controller
================================================
Rule-based valve controller that requires NO machine learning.
This is the fallback and the "before AI" benchmark.

Strategy:
    - High temperature → open valve (extract more power)
    - Pressure near limit → close valve (safety)
    - Low temperature → close valve (let heat build up)
    - Anti-oscillation: max ±5% change per tick
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time

import httpx

# ── Resolve imports whether run from ai/ or project root ──
sys.path.insert(0, ".")
from config import (
    CONTROL_ENDPOINT,
    CONTROL_INTERVAL,
    HEURISTIC_COOL_TEMP,
    HEURISTIC_HIGH_TEMP,
    HEURISTIC_LOW_TEMP,
    HEURISTIC_VALVE_SMALL_STEP,
    HEURISTIC_VALVE_STEP,
    HEURISTIC_WARM_TEMP,
    MAX_VALVE,
    MAX_VALVE_CHANGE_PER_TICK,
    METRICS_ENDPOINT,
    MIN_VALVE,
    PRESSURE_CRITICAL,
    PRESSURE_SAFETY_LIMIT,
    TEMPERATURE_CRITICAL,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ai-heuristic")


# ──────────────────────────────────────────────
# Core heuristic logic
# ──────────────────────────────────────────────

def compute_valve_heuristic(metrics: dict, current_valve: float) -> float:
    """
    Determine the next valve position using hand-crafted rules.

    Priority order:
        1. Pressure safety (override everything)
        2. Temperature-based optimization
        3. Anti-oscillation clamp

    Args:
        metrics: Current plant state from GET /metrics.
        current_valve: Current valve opening (%).

    Returns:
        Recommended valve position (0–100 %).
    """
    temp = metrics["temperature"]
    pressure = metrics["pressure"]
    valve = current_valve

    # ── PRIORITY 1: Pressure safety ──────────────────
    if pressure > PRESSURE_CRITICAL:
        # EMERGENCY — reduce aggressively
        valve -= 8.0
        logger.warning(
            "🚨 EMERGENCY pressure %.2f bar > %.1f → valve -8%%",
            pressure,
            PRESSURE_CRITICAL,
        )
    elif pressure > PRESSURE_SAFETY_LIMIT:
        # CAUTION — reduce gently
        valve -= HEURISTIC_VALVE_STEP
        logger.warning(
            "⚠️  High pressure %.2f bar > %.1f → valve -%.0f%%",
            pressure,
            PRESSURE_SAFETY_LIMIT,
            HEURISTIC_VALVE_STEP,
        )

    # ── PRIORITY 2: Temperature-based optimization ───
    elif temp > TEMPERATURE_CRITICAL:
        # Way too hot — reduce valve
        valve -= 5.0
        logger.warning(
            "⚠️  Critical temp %.1f°C > %.0f → valve -5%%",
            temp,
            TEMPERATURE_CRITICAL,
        )
    elif temp > HEURISTIC_HIGH_TEMP:
        # Hot — open valve to extract more power
        valve += HEURISTIC_VALVE_STEP
    elif temp > HEURISTIC_WARM_TEMP:
        # Warm — slightly open
        valve += HEURISTIC_VALVE_SMALL_STEP
    elif temp < HEURISTIC_LOW_TEMP:
        # Cold — close valve to let heat build up
        valve -= HEURISTIC_VALVE_STEP
    elif temp < HEURISTIC_COOL_TEMP:
        # Cool — slightly close
        valve -= HEURISTIC_VALVE_SMALL_STEP

    # ── PRIORITY 3: Anti-oscillation ─────────────────
    delta = valve - current_valve
    delta = max(-MAX_VALVE_CHANGE_PER_TICK, min(MAX_VALVE_CHANGE_PER_TICK, delta))
    valve = current_valve + delta

    # ── Hard clamp ──
    valve = max(MIN_VALVE, min(MAX_VALVE, valve))

    return round(valve, 2)


# ──────────────────────────────────────────────
# Async control loop
# ──────────────────────────────────────────────

async def run_baseline(duration: float | None = None) -> list[dict]:
    """
    Run the heuristic controller in a continuous loop.

    Args:
        duration: If set, run for this many seconds then stop.
                  If None, run forever.

    Returns:
        History of (metrics + ai_valve) dicts.
    """
    logger.info("🤖 Heuristic baseline controller STARTING")
    history: list[dict] = []
    start = time.time()

    async with httpx.AsyncClient(timeout=5.0) as client:
        while True:
            try:
                # ── Read plant state ──
                resp = await client.get(METRICS_ENDPOINT)
                metrics = resp.json()

                # ── Compute optimal valve ──
                current_valve = metrics["valve_position"]
                new_valve = compute_valve_heuristic(metrics, current_valve)

                # ── Send control command ──
                await client.post(
                    CONTROL_ENDPOINT,
                    json={"valve_position": new_valve},
                )

                # ── Log ──
                entry = {
                    **metrics,
                    "ai_valve": new_valve,
                    "mode": "heuristic",
                }
                history.append(entry)

                logger.info(
                    "T=%6.1f°C  P=%5.2fbar  V=%5.1f→%5.1f%%  W=%6.1fkW",
                    metrics["temperature"],
                    metrics["pressure"],
                    current_valve,
                    new_valve,
                    metrics["power_output"],
                )

                # ── Duration check ──
                if duration and (time.time() - start) >= duration:
                    logger.info(
                        "⏱  Duration %.0fs reached — stopping baseline.",
                        duration,
                    )
                    break

                await asyncio.sleep(CONTROL_INTERVAL)

            except httpx.ConnectError:
                logger.error("Backend not reachable. Retrying in 3s...")
                await asyncio.sleep(3)
            except KeyboardInterrupt:
                logger.info("Interrupted by user.")
                break
            except Exception as exc:
                logger.error("Tick error: %s", exc, exc_info=True)
                await asyncio.sleep(CONTROL_INTERVAL)

    logger.info(
        "🤖 Heuristic baseline STOPPED — %d decisions made.", len(history)
    )
    return history


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Heuristic baseline controller")
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Run for N seconds then stop (default: run forever)",
    )
    args = parser.parse_args()

    asyncio.run(run_baseline(duration=args.duration))
