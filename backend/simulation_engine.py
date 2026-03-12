"""
Entropy Engine — Physics Simulation Core
==========================================
Continuous 1 Hz loop that models a waste heat recovery plant.

Components modelled:
    Furnace → Heat Exchanger → Steam Drum → Turbine Generator

Physics (simplified thermodynamics):
    1. Heat decay:   dT/dt = -k * (T - T_ambient)
    2. Effective flow: F_eff = flow_rate * (valve_position / 100)
    3. Pressure:     P = c1 * T * F_eff
    4. Power output: W = η  * P * F_eff
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Dict

from config import (
    AMBIENT_TEMPERATURE,
    FLOW_DRIFT_MAGNITUDE,
    FURNACE_HEAT_RATE,
    HEAT_DECAY_CONSTANT,
    HEAT_SPIKE_MAGNITUDE,
    HEAT_SPIKE_PROBABILITY,
    INERTIA_FACTOR,
    INITIAL_FLOW_RATE,
    INITIAL_POWER_OUTPUT,
    INITIAL_PRESSURE,
    INITIAL_TEMPERATURE,
    INITIAL_VALVE_POSITION,
    LOG_INTERVAL,
    MAX_FLOW_RATE,
    MAX_POWER,
    MAX_PRESSURE,
    MAX_TEMPERATURE,
    MIN_FLOW_RATE,
    MIN_POWER,
    MIN_PRESSURE,
    MIN_TEMPERATURE,
    POWER_MULTIPLIER,
    PRESSURE_COEFFICIENT,
    SENSOR_NOISE_PERCENT,
    TICK_INTERVAL,
    TURBINE_EFFICIENCY,
    VALVE_RESPONSE_RATE,
)

logger = logging.getLogger("entropy-engine")


# ──────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────

def clamp(value: float, lo: float, hi: float) -> float:
    """Restrict *value* to the closed interval [lo, hi]."""
    return max(lo, min(hi, value))


def add_noise(value: float, noise_pct: float = SENSOR_NOISE_PERCENT) -> float:
    """Return *value* with uniform sensor noise applied."""
    return value * (1.0 + random.uniform(-noise_pct, noise_pct))


# ──────────────────────────────────────────────
# Simulation Engine
# ──────────────────────────────────────────────

class SimulationEngine:
    """
    Continuous physics loop for the waste heat recovery plant.

    Usage::

        engine = SimulationEngine()
        asyncio.create_task(engine.run())   # start in background
        state  = await engine.get_metrics() # read current state
        await engine.set_valve(75.0)        # send control command
    """

    def __init__(self) -> None:
        # ── True internal state (noise-free) ──
        self._temperature: float = INITIAL_TEMPERATURE
        self._flow_rate: float = INITIAL_FLOW_RATE
        self._pressure: float = INITIAL_PRESSURE
        self._valve_position: float = INITIAL_VALVE_POSITION
        self._power_output: float = INITIAL_POWER_OUTPUT

        # ── Valve target (for inertia / delay) ──
        self._target_valve_position: float = INITIAL_VALVE_POSITION

        # ── Timing ──
        self._tick_count: int = 0
        self._start_time: float = time.time()
        self._running: bool = False

        # ── Concurrency ──
        self._lock: asyncio.Lock = asyncio.Lock()

        # ── AI auto-control mode ──
        self._ai_mode: bool = False

        # ── Public snapshot (with noise — served by API) ──
        self.current_state: Dict[str, float] = self._build_snapshot()

        logger.info(
            "Engine initialised | T=%.1f°C  P=%.1fbar  V=%.0f%%",
            self._temperature,
            self._pressure,
            self._valve_position,
        )

    # ──────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────

    @property
    def uptime(self) -> float:
        return time.time() - self._start_time

    @property
    def tick_count(self) -> int:
        return self._tick_count

    @property
    def ai_mode(self) -> bool:
        return self._ai_mode

    # ──────────────────────────────────────────
    # Main loop
    # ──────────────────────────────────────────

    async def run(self) -> None:
        """Start the infinite simulation loop (1 Hz)."""
        self._running = True
        logger.info("Simulation loop STARTED  (tick interval=%.1fs)", TICK_INTERVAL)

        while self._running:
            try:
                async with self._lock:
                    self._update_physics()
                    self._tick_count += 1

                    # Periodic console log
                    if self._tick_count % LOG_INTERVAL == 0:
                        self._log_state()

                    # Simple AI auto-control (rule-based placeholder)
                    if self._ai_mode:
                        self._auto_control()

                await asyncio.sleep(TICK_INTERVAL)

            except asyncio.CancelledError:
                logger.info("Simulation loop CANCELLED")
                break
            except Exception as exc:  # noqa: BLE001
                logger.error("Simulation tick error: %s", exc, exc_info=True)
                await asyncio.sleep(TICK_INTERVAL)

    def stop(self) -> None:
        """Signal the loop to exit gracefully."""
        self._running = False
        logger.info("Simulation loop STOPPED  (ticks=%d)", self._tick_count)

    # ──────────────────────────────────────────
    # Physics update (called every tick)
    # ──────────────────────────────────────────

    def _update_physics(self) -> None:
        """Advance the plant model by one time-step (dt = TICK_INTERVAL)."""
        dt = TICK_INTERVAL

        # ── Step 1: Heat input from furnace + decay ──
        dT = (FURNACE_HEAT_RATE - HEAT_DECAY_CONSTANT * (self._temperature - AMBIENT_TEMPERATURE)) * dt
        new_temperature = self._temperature + dT

        # ── Step 2: Random heat spike (furnace burst) ──
        if random.random() < HEAT_SPIKE_PROBABILITY:
            new_temperature += HEAT_SPIKE_MAGNITUDE
            logger.warning(
                "🔥 Heat spike! +%.0f°C  (T now %.1f°C)",
                HEAT_SPIKE_MAGNITUDE,
                new_temperature,
            )

        # ── Step 3: Valve inertia (gradual movement) ──
        valve_delta = VALVE_RESPONSE_RATE * (
            self._target_valve_position - self._valve_position
        )
        new_valve = self._valve_position + valve_delta

        # ── Step 4: Flow-rate drift (small random walk) ──
        flow_drift = random.uniform(-FLOW_DRIFT_MAGNITUDE, FLOW_DRIFT_MAGNITUDE)
        new_flow = self._flow_rate + flow_drift

        # ── Step 5: Effective flow ──
        effective_flow = new_flow * (new_valve / 100.0)

        # ── Step 6: Pressure ──
        new_pressure = PRESSURE_COEFFICIENT * new_temperature * effective_flow

        # ── Step 7: Power output ──
        new_power = POWER_MULTIPLIER * TURBINE_EFFICIENCY * new_pressure * effective_flow

        # ── Step 8: Inertia smoothing (no instant jumps) ──
        self._temperature = (
            INERTIA_FACTOR * self._temperature
            + (1 - INERTIA_FACTOR) * new_temperature
        )
        self._flow_rate = (
            INERTIA_FACTOR * self._flow_rate
            + (1 - INERTIA_FACTOR) * new_flow
        )
        self._pressure = (
            INERTIA_FACTOR * self._pressure
            + (1 - INERTIA_FACTOR) * new_pressure
        )
        self._power_output = (
            INERTIA_FACTOR * self._power_output
            + (1 - INERTIA_FACTOR) * new_power
        )
        self._valve_position = new_valve  # valve uses its own inertia above

        # ── Step 9: Clamp to safe operating ranges ──
        self._temperature = clamp(self._temperature, MIN_TEMPERATURE, MAX_TEMPERATURE)
        self._flow_rate = clamp(self._flow_rate, MIN_FLOW_RATE, MAX_FLOW_RATE)
        self._valve_position = clamp(self._valve_position, 0.0, 100.0)

        # Safety relief valve
        if self._pressure > MAX_PRESSURE:
            logger.warning(
                "⚠️  Pressure capped at %.1f bar (was %.2f)",
                MAX_PRESSURE,
                self._pressure,
            )
        self._pressure = clamp(self._pressure, MIN_PRESSURE, MAX_PRESSURE)
        self._power_output = clamp(self._power_output, MIN_POWER, MAX_POWER)

        # ── Step 10: Build noisy public snapshot ──
        self.current_state = self._build_snapshot()

    # ──────────────────────────────────────────
    # Snapshot builder
    # ──────────────────────────────────────────

    def _build_snapshot(self) -> Dict[str, float]:
        """Return a sensor-reading dict with noise applied and clamped."""
        return {
            "temperature": round(
                clamp(add_noise(self._temperature), MIN_TEMPERATURE, MAX_TEMPERATURE), 2
            ),
            "pressure": round(
                clamp(add_noise(self._pressure), MIN_PRESSURE, MAX_PRESSURE), 2
            ),
            "flow_rate": round(
                clamp(add_noise(self._flow_rate), MIN_FLOW_RATE, MAX_FLOW_RATE), 2
            ),
            "valve_position": round(
                clamp(self._valve_position, 0.0, 100.0), 2
            ),
            "power_output": round(
                clamp(add_noise(self._power_output), MIN_POWER, MAX_POWER), 2
            ),
            "timestamp": round(time.time(), 3),
        }

    # ──────────────────────────────────────────
    # Control interface (called by API)
    # ──────────────────────────────────────────

    async def set_valve(self, position: float) -> None:
        """Set the *target* valve position.  Actual movement is gradual."""
        async with self._lock:
            old = self._target_valve_position
            self._target_valve_position = clamp(position, 0.0, 100.0)
            logger.info(
                "Valve target changed: %.1f%% → %.1f%%",
                old,
                self._target_valve_position,
            )

    async def get_metrics(self) -> Dict[str, float]:
        """Return the latest noisy sensor snapshot (thread-safe copy)."""
        async with self._lock:
            return dict(self.current_state)

    async def set_ai_mode(self, enabled: bool) -> None:
        """Toggle the built-in rule-based auto-controller."""
        async with self._lock:
            self._ai_mode = enabled
            logger.info("AI auto-control mode: %s", "ON" if enabled else "OFF")

    # ──────────────────────────────────────────
    # Rule-based AI placeholder
    # ──────────────────────────────────────────

    def _auto_control(self) -> None:
        """Simple proportional controller — keeps temperature near 500 °C."""
        target_temp = 500.0
        error = self._temperature - target_temp

        # Open valve when too hot, close when too cold
        adjustment = error * 0.5  # proportional gain
        new_target = clamp(self._valve_position + adjustment, 0.0, 100.0)
        self._target_valve_position = new_target

    # ──────────────────────────────────────────
    # Logging
    # ──────────────────────────────────────────

    def _log_state(self) -> None:
        """Print a concise state summary to the console."""
        logger.info(
            "Tick %5d | T=%6.1f°C  P=%5.2fbar  F=%4.2fkg/s  V=%5.1f%%  W=%6.1fkW",
            self._tick_count,
            self._temperature,
            self._pressure,
            self._flow_rate,
            self._valve_position,
            self._power_output,
        )
