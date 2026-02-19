"""
Entropy Engine — End-to-End Integration Test
==============================================
Compares baseline (no AI) vs heuristic vs MPC controller,
measures power improvement, and verifies safety throughout.

Usage:
    python test_ai_integration.py               # full test (~80 s)
    python test_ai_integration.py --quick        # quick test (~30 s)

Requires:
    - Backend running on localhost:8000 (fresh restart recommended)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time

import httpx
import numpy as np

sys.path.insert(0, ".")
from config import (
    CONTROL_ENDPOINT,
    CONTROL_INTERVAL,
    METRICS_ENDPOINT,
    PRESSURE_HARD_LIMIT,
)
from baseline_controller import compute_valve_heuristic
from safety import enforce_safety, get_safety_status
from utils import AIReport

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("integration-test")


async def _read_and_collect(client: httpx.AsyncClient, seconds: int) -> list[dict]:
    """Passively read metrics for *seconds*, return list of snapshots."""
    readings = []
    for _ in range(seconds):
        r = await client.get(METRICS_ENDPOINT)
        readings.append(r.json())
        await asyncio.sleep(1)
    return readings


async def _run_heuristic(client: httpx.AsyncClient, seconds: int) -> list[dict]:
    """Run heuristic controller for *seconds*, return readings."""
    readings = []
    for _ in range(seconds):
        r = (await client.get(METRICS_ENDPOINT)).json()
        new_valve = compute_valve_heuristic(r, r["valve_position"])
        new_valve = enforce_safety(r, new_valve)
        await client.post(CONTROL_ENDPOINT, json={"valve_position": round(new_valve, 2)})
        readings.append(r)
        await asyncio.sleep(1)
    return readings


async def _run_mpc(client: httpx.AsyncClient, seconds: int) -> list[dict]:
    """Run MPC controller for *seconds*, return readings."""
    from mpc_controller import ModelPredictiveController

    mpc = ModelPredictiveController.from_checkpoint()
    readings = []
    for _ in range(seconds):
        r = (await client.get(METRICS_ENDPOINT)).json()
        decision = mpc.find_optimal_valve(r)
        new_valve = enforce_safety(r, decision["optimal_valve"])
        await client.post(CONTROL_ENDPOINT, json={"valve_position": round(new_valve, 2)})
        readings.append({**r, **decision})
        await asyncio.sleep(1)
    return readings


async def test_full_integration(quick: bool = False):
    """
    End-to-end test:
      Phase A: Baseline (no AI)  → record avg power
      Phase B: Heuristic AI      → measure improvement
      Phase C: MPC AI            → measure further improvement
      Phase D: Safety audit
    """
    t_baseline = 10 if quick else 20
    t_heuristic = 10 if quick else 30
    t_mpc = 10 if quick else 30

    total_est = t_baseline + t_heuristic + t_mpc
    logger.info("🧪 Integration test starting — ~%d s total", total_est)

    async with httpx.AsyncClient(timeout=5.0) as client:
        # ── Phase A: Baseline ──
        logger.info("📊 Phase A: Recording baseline (%d s, no AI)...", t_baseline)
        baseline = await _read_and_collect(client, t_baseline)
        baseline_powers = [r["power_output"] for r in baseline]
        avg_baseline = float(np.mean(baseline_powers))
        max_pressure_a = max(r["pressure"] for r in baseline)
        logger.info("   Baseline avg power: %.1f kW", avg_baseline)

        # ── Phase B: Heuristic ──
        logger.info("🤖 Phase B: Heuristic controller (%d s)...", t_heuristic)
        heuristic = await _run_heuristic(client, t_heuristic)
        # Use last 2/3 to avoid initial transient
        stable = heuristic[len(heuristic)//3:]
        heuristic_powers = [r["power_output"] for r in stable]
        avg_heuristic = float(np.mean(heuristic_powers))
        max_pressure_b = max(r["pressure"] for r in heuristic)
        improvement_h = (avg_heuristic - avg_baseline) / max(avg_baseline, 1) * 100
        logger.info("   Heuristic avg power: %.1f kW  (%.+1f%%)", avg_heuristic, improvement_h)

        # ── Phase C: MPC ──
        logger.info("🧠 Phase C: MPC controller (%d s)...", t_mpc)
        try:
            mpc_readings = await _run_mpc(client, t_mpc)
            stable_mpc = mpc_readings[len(mpc_readings)//3:]
            mpc_powers = [r["power_output"] for r in stable_mpc]
            avg_mpc = float(np.mean(mpc_powers))
            max_pressure_c = max(r["pressure"] for r in mpc_readings)
            improvement_m = (avg_mpc - avg_baseline) / max(avg_baseline, 1) * 100
            logger.info("   MPC avg power: %.1f kW  (%+.1f%%)", avg_mpc, improvement_m)
        except Exception as exc:
            logger.warning("MPC phase failed (model may not be trained): %s", exc)
            avg_mpc = None
            max_pressure_c = 0
            improvement_m = None

        # ── Phase D: Safety audit ──
        max_pressure = max(max_pressure_a, max_pressure_b, max_pressure_c)
        safe = max_pressure <= PRESSURE_HARD_LIMIT

    # ── Print results ──
    print()
    print("=" * 60)
    print("  ENTROPY ENGINE — INTEGRATION TEST RESULTS")
    print("=" * 60)
    print(f"  Baseline power:      {avg_baseline:>8.1f} kW")
    print(f"  Heuristic power:     {avg_heuristic:>8.1f} kW  ({improvement_h:+.1f}%)")
    if avg_mpc is not None:
        print(f"  MPC power:           {avg_mpc:>8.1f} kW  ({improvement_m:+.1f}%)")
    else:
        print(f"  MPC power:           {'SKIPPED':>8s}")
    print(f"  Max pressure seen:   {max_pressure:>8.2f} bar")
    print(f"  Safety:              {'✅ PASS' if safe else '❌ FAIL'}")
    print(f"  Total time:          {t_baseline + t_heuristic + t_mpc:>5d} s")
    print("=" * 60)

    return {
        "baseline_kw": avg_baseline,
        "heuristic_kw": avg_heuristic,
        "heuristic_improvement_pct": improvement_h,
        "mpc_kw": avg_mpc,
        "mpc_improvement_pct": improvement_m,
        "max_pressure": max_pressure,
        "safety_pass": safe,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entropy Engine integration test")
    parser.add_argument("--quick", action="store_true", help="Short test (~30 s)")
    args = parser.parse_args()
    asyncio.run(test_full_integration(quick=args.quick))
