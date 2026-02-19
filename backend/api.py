"""
Entropy Engine — FastAPI Service Layer
=======================================
Exposes the simulation state over HTTP and accepts control commands.

Endpoints:
    GET  /metrics  → current plant sensor readings
    POST /control  → update valve position
    GET  /status   → engine health & uptime
    GET  /health   → simple liveness probe
"""

from __future__ import annotations

import asyncio
import logging

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import API_HOST, API_PORT, DEBUG_MODE
from models import ControlInput, PlantMetrics, SystemStatus
from simulation_engine import SimulationEngine

# ──────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("entropy-engine")

# ──────────────────────────────────────────────
# Singleton simulation engine
# ──────────────────────────────────────────────
engine = SimulationEngine()


# ──────────────────────────────────────────────
# Application lifespan (startup / shutdown)
# ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the simulation loop as a background task on server boot."""
    logger.info("🚀 Entropy Engine starting up …")
    sim_task = asyncio.create_task(engine.run())
    yield
    logger.info("🛑 Shutting down simulation …")
    engine.stop()
    sim_task.cancel()
    try:
        await sim_task
    except asyncio.CancelledError:
        pass


# ──────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────
app = FastAPI(
    title="Entropy Engine",
    description=(
        "Industrial waste heat recovery plant simulator. "
        "Provides real-time sensor metrics and accepts control commands."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow all origins during hackathon development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get(
    "/metrics",
    response_model=PlantMetrics,
    summary="Get current plant metrics",
    tags=["Plant"],
)
async def get_metrics():
    """Return the latest sensor readings from the plant simulation."""
    return await engine.get_metrics()


@app.post(
    "/control",
    summary="Send a control command",
    tags=["Plant"],
)
async def set_control(command: ControlInput):
    """Update the target valve position. The valve moves gradually due to inertia."""
    await engine.set_valve(command.valve_position)
    return {
        "status": "accepted",
        "target_valve_position": command.valve_position,
    }


@app.get(
    "/status",
    response_model=SystemStatus,
    summary="Engine health & uptime",
    tags=["System"],
)
async def get_status():
    """Return operational information about the simulation engine."""
    return SystemStatus(
        status="running",
        uptime_seconds=round(engine.uptime, 2),
        tick_count=engine.tick_count,
        ai_mode=engine.ai_mode,
    )


@app.post(
    "/ai-mode",
    summary="Toggle AI auto-control",
    tags=["System"],
)
async def toggle_ai_mode(enabled: bool = True):
    """Enable or disable the built-in rule-based auto-controller."""
    await engine.set_ai_mode(enabled)
    return {"ai_mode": enabled}


@app.get(
    "/health",
    summary="Liveness probe",
    tags=["System"],
)
async def health_check():
    """Simple health check for monitoring."""
    return {"status": "ok"}


# ──────────────────────────────────────────────
# Direct execution entry-point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
    )
