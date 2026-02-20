"""
Entropy Engine — Orchestrator Configuration
=============================================
Env-driven configuration for the integration layer.
All values fall back to sensible defaults if .env is missing.

IMPORTANT: This config is a SUPERSET of the AI config (ai/config.py).
It includes all AI constants so that Person 2's modules (mpc_controller,
model, safety, etc.) can import from 'config' without collision.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Load .env from project root
ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT_DIR / ".env"

try:
    from dotenv import load_dotenv
    load_dotenv(ENV_FILE)
except ImportError:
    pass  # dotenv is optional — use defaults or system env

# ─────────────────────────────────────────────
# Simulation Backend (Person 1)
# ─────────────────────────────────────────────
SIM_API_URL: str = os.getenv("SIMULATION_API_URL", "http://localhost:8000")
SIM_METRICS_URL: str = f"{SIM_API_URL}{os.getenv('SIMULATION_METRICS_PATH', '/metrics')}"
SIM_CONTROL_URL: str = f"{SIM_API_URL}{os.getenv('SIMULATION_CONTROL_PATH', '/control')}"
SIM_STATUS_URL: str = f"{SIM_API_URL}{os.getenv('SIMULATION_STATUS_PATH', '/status')}"
SIM_HEALTH_URL: str = f"{SIM_API_URL}{os.getenv('SIMULATION_HEALTH_PATH', '/health')}"

# ── These aliases are used by Person 2's AI modules ──
BACKEND_URL: str = SIM_API_URL
METRICS_ENDPOINT: str = SIM_METRICS_URL
CONTROL_ENDPOINT: str = SIM_CONTROL_URL
STATUS_ENDPOINT: str = f"{SIM_API_URL}/status"

# ─────────────────────────────────────────────
# AI Model (Person 2)
# ─────────────────────────────────────────────
AI_MODEL_PATH: str = os.getenv(
    "AI_MODEL_PATH",
    str(ROOT_DIR / "ai" / "ai" / "models" / "pinn_model.pt"),
)
AI_MODE_DEFAULT: bool = os.getenv("AI_MODE_DEFAULT", "false").lower() == "true"
AI_CONFIDENCE_THRESHOLD: float = float(os.getenv("AI_CONFIDENCE_THRESHOLD", "0.3"))

# ── Alias used by Person 2's model.py / train.py ──
MODEL_SAVE_PATH: str = AI_MODEL_PATH

# ─────────────────────────────────────────────
# Safety Thresholds
# ─────────────────────────────────────────────
PRESSURE_HARD_LIMIT: float = float(os.getenv("PRESSURE_LIMIT", "8.0"))
PRESSURE_WARNING: float = float(os.getenv("PRESSURE_WARNING", "7.5"))
PRESSURE_CRITICAL: float = float(os.getenv("PRESSURE_CRITICAL", "7.8"))
TEMPERATURE_CRITICAL: float = float(os.getenv("TEMPERATURE_CRITICAL", "590"))
SAFE_VALVE_POSITION: float = float(os.getenv("SAFE_VALVE_POSITION", "50.0"))

# ── Aliases used by Person 2 ──
PRESSURE_SAFETY_LIMIT: float = PRESSURE_WARNING   # ai/safety.py uses this name

# ─────────────────────────────────────────────
# Valve Constraints
# ─────────────────────────────────────────────
MIN_VALVE: float = 0.0
MAX_VALVE: float = 100.0
MAX_VALVE_CHANGE_PER_TICK: float = 5.0

# ─────────────────────────────────────────────
# Orchestrator Server
# ─────────────────────────────────────────────
ORCHESTRATOR_HOST: str = os.getenv("ORCHESTRATOR_HOST", "0.0.0.0")
ORCHESTRATOR_PORT: int = int(os.getenv("ORCHESTRATOR_PORT", "8001"))
CONTROL_INTERVAL: float = float(os.getenv("CONTROL_INTERVAL", "1.0"))
MAX_HISTORY: int = int(os.getenv("MAX_HISTORY_LENGTH", "300"))
MAX_RECOVERY_ATTEMPTS: int = int(os.getenv("MAX_RECOVERY_ATTEMPTS", "3"))
MAX_CONSECUTIVE_ERRORS: int = int(os.getenv("MAX_CONSECUTIVE_ERRORS", "5"))

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# ─────────────────────────────────────────────
# MPC Controller
# ─────────────────────────────────────────────
MPC_NUM_CANDIDATES: int = 50
MPC_SEARCH_RANGE: float = 15.0
MPC_CONFIDENCE_THRESHOLD: float = float(os.getenv("MPC_CONFIDENCE_THRESHOLD", "0.01"))

# ─────────────────────────────────────────────
# Physics Constants (must match Person 1)
# ─────────────────────────────────────────────
PRESSURE_COEFFICIENT: float = 0.004
TURBINE_EFFICIENCY: float = 0.35

# ─────────────────────────────────────────────
# Optimization Targets (Person 2)
# ─────────────────────────────────────────────
TARGET_POWER: float = 300.0
TARGET_TEMPERATURE: float = 520.0

# ─────────────────────────────────────────────
# Heuristic Controller Params (Person 2)
# ─────────────────────────────────────────────
HEURISTIC_HIGH_TEMP: float = 560.0
HEURISTIC_LOW_TEMP: float = 440.0
HEURISTIC_WARM_TEMP: float = 520.0
HEURISTIC_COOL_TEMP: float = 480.0
HEURISTIC_VALVE_STEP: float = 3.0
HEURISTIC_VALVE_SMALL_STEP: float = 1.0

# ─────────────────────────────────────────────
# Data Collection (Person 2)
# ─────────────────────────────────────────────
DATA_DIR: str = str(ROOT_DIR / "ai" / "ai" / "data")
DATA_FILE: str = str(ROOT_DIR / "ai" / "ai" / "data" / "training_data.csv")
MIN_TRAINING_SAMPLES: int = 1000
COLLECTION_INTERVAL: float = 1.0
VALVE_PERTURBATION_INTERVAL: int = 50
VALVE_PERTURBATION_MIN: float = 20.0
VALVE_PERTURBATION_MAX: float = 80.0

# ─────────────────────────────────────────────
# Model Architecture (Person 2)
# ─────────────────────────────────────────────
INPUT_DIM: int = 5
HIDDEN_DIM: int = 64
OUTPUT_DIM: int = 1
DROPOUT: float = 0.1

# ─────────────────────────────────────────────
# Training Hyperparameters (Person 2)
# ─────────────────────────────────────────────
LEARNING_RATE: float = 0.001
EPOCHS: int = 200
BATCH_SIZE: int = 64
TRAIN_SPLIT: float = 0.8

# ─────────────────────────────────────────────
# PINN Loss Weights (Person 2)
# ─────────────────────────────────────────────
DATA_LOSS_WEIGHT: float = 1.0
PHYSICS_LOSS_WEIGHT: float = 0.1
SAFETY_LOSS_WEIGHT: float = 0.5
