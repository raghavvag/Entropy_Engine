"""
Entropy Engine — AI Module Configuration
==========================================
All constants and hyperparameters for the AI optimization layer.
No magic numbers should exist outside this file.
"""

# ─────────────────────────────────────────────
# Backend Connection
# ─────────────────────────────────────────────
BACKEND_URL: str = "http://localhost:8000"
METRICS_ENDPOINT: str = f"{BACKEND_URL}/metrics"
CONTROL_ENDPOINT: str = f"{BACKEND_URL}/control"
STATUS_ENDPOINT: str = f"{BACKEND_URL}/status"

# ─────────────────────────────────────────────
# Control Loop Timing
# ─────────────────────────────────────────────
CONTROL_INTERVAL: float = 1.0           # seconds between AI decisions

# ─────────────────────────────────────────────
# Safety Thresholds
# ─────────────────────────────────────────────
PRESSURE_SAFETY_LIMIT: float = 7.5      # bar — start reducing before 8.0
PRESSURE_HARD_LIMIT: float = 8.0        # bar — absolute max
PRESSURE_CRITICAL: float = 7.8          # bar — emergency reduce
TEMPERATURE_CRITICAL: float = 590.0     # °C  — emergency reduce

# ─────────────────────────────────────────────
# Valve Constraints
# ─────────────────────────────────────────────
MIN_VALVE: float = 0.0
MAX_VALVE: float = 100.0
MAX_VALVE_CHANGE_PER_TICK: float = 5.0  # max % change per decision (anti-oscillation)

# ─────────────────────────────────────────────
# Optimization Targets
# ─────────────────────────────────────────────
TARGET_POWER: float = 300.0             # kW — ideal maximum
TARGET_TEMPERATURE: float = 520.0       # °C — sweet spot

# ─────────────────────────────────────────────
# Heuristic Controller Params
# ─────────────────────────────────────────────
HEURISTIC_HIGH_TEMP: float = 560.0
HEURISTIC_LOW_TEMP: float = 440.0
HEURISTIC_WARM_TEMP: float = 520.0
HEURISTIC_COOL_TEMP: float = 480.0
HEURISTIC_VALVE_STEP: float = 3.0
HEURISTIC_VALVE_SMALL_STEP: float = 1.0

# ─────────────────────────────────────────────
# Data Collection
# ─────────────────────────────────────────────
DATA_DIR: str = "ai/data"
DATA_FILE: str = "ai/data/training_data.csv"
MIN_TRAINING_SAMPLES: int = 1000
COLLECTION_INTERVAL: float = 1.0        # seconds
VALVE_PERTURBATION_INTERVAL: int = 50   # perturb every N samples
VALVE_PERTURBATION_MIN: float = 20.0
VALVE_PERTURBATION_MAX: float = 80.0

# ─────────────────────────────────────────────
# Model Architecture
# ─────────────────────────────────────────────
INPUT_DIM: int = 5                      # [temp, pressure, flow, valve, power]
HIDDEN_DIM: int = 64
OUTPUT_DIM: int = 1                     # predicted power
DROPOUT: float = 0.1

# ─────────────────────────────────────────────
# Training Hyperparameters
# ─────────────────────────────────────────────
LEARNING_RATE: float = 0.001
EPOCHS: int = 200
BATCH_SIZE: int = 64
TRAIN_SPLIT: float = 0.8               # 80% train, 20% val

# ─────────────────────────────────────────────
# PINN Loss Weights
# ─────────────────────────────────────────────
DATA_LOSS_WEIGHT: float = 1.0           # λ_data
PHYSICS_LOSS_WEIGHT: float = 0.1        # λ_physics
SAFETY_LOSS_WEIGHT: float = 0.5         # λ_safety

# ─────────────────────────────────────────────
# Physics Constants (must match Person 1's config)
# ─────────────────────────────────────────────
PRESSURE_COEFFICIENT: float = 0.004     # c1: pressure = c1 * T * F_eff
TURBINE_EFFICIENCY: float = 0.35        # η:  power   = η  * P * F_eff

# ─────────────────────────────────────────────
# MPC Controller
# ─────────────────────────────────────────────
MPC_NUM_CANDIDATES: int = 50            # valve positions to evaluate
MPC_SEARCH_RANGE: float = 15.0         # ±% around current valve
MPC_CONFIDENCE_THRESHOLD: float = 0.01  # below this → fall back to heuristic

# ─────────────────────────────────────────────
# Model Save Path
# ─────────────────────────────────────────────
MODEL_SAVE_PATH: str = "ai/models/pinn_model.pt"
