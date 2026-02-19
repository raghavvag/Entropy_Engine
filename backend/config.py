"""
Entropy Engine — Configuration & Constants
===========================================
All tunable parameters for the waste heat recovery plant simulation.
No magic numbers should exist outside this file.
"""

# ─────────────────────────────────────────────
# Simulation Timing
# ─────────────────────────────────────────────
TICK_INTERVAL: float = 1.0          # seconds between physics updates

# ─────────────────────────────────────────────
# Ambient / Environment
# ─────────────────────────────────────────────
AMBIENT_TEMPERATURE: float = 25.0   # °C — surrounding air temperature

# ─────────────────────────────────────────────
# Heat Decay
# ─────────────────────────────────────────────
HEAT_DECAY_CONSTANT: float = 0.02   # k  in  dT/dt = -k * (T - T_amb)

# ─────────────────────────────────────────────
# Initial Conditions (plant startup values)
# ─────────────────────────────────────────────
INITIAL_TEMPERATURE: float = 500.0      # °C
INITIAL_FLOW_RATE: float = 3.0          # kg/s
INITIAL_PRESSURE: float = 6.0           # bar
INITIAL_VALVE_POSITION: float = 50.0    # %
INITIAL_POWER_OUTPUT: float = 200.0     # kW

# ─────────────────────────────────────────────
# Physical / Thermodynamic Constants
# ─────────────────────────────────────────────
PRESSURE_COEFFICIENT: float = 0.004     # c1 : pressure = c1 * T * F_eff
TURBINE_EFFICIENCY: float = 0.35        # η  : power   = η  * P * F_eff

# ─────────────────────────────────────────────
# Operating Ranges (hard clamps)
# ─────────────────────────────────────────────
MIN_TEMPERATURE: float = 400.0     # °C
MAX_TEMPERATURE: float = 600.0     # °C

MIN_FLOW_RATE: float = 2.0        # kg/s
MAX_FLOW_RATE: float = 5.0        # kg/s

MIN_PRESSURE: float = 4.0         # bar
MAX_PRESSURE: float = 8.0         # bar  ← safety relief valve

MIN_POWER: float = 150.0          # kW
MAX_POWER: float = 300.0          # kW

# ─────────────────────────────────────────────
# Realism Modifiers
# ─────────────────────────────────────────────
SENSOR_NOISE_PERCENT: float = 0.02      # ±2% noise on sensor readings
HEAT_SPIKE_PROBABILITY: float = 0.02    # 2% chance per tick
HEAT_SPIKE_MAGNITUDE: float = 30.0      # °C added on spike event
VALVE_RESPONSE_RATE: float = 0.1        # valve moves 10% toward target per tick
INERTIA_FACTOR: float = 0.85            # smoothing: new = 0.85*old + 0.15*calc

# ─────────────────────────────────────────────
# Flow Rate Drift (keeps flow_rate dynamic)
# ─────────────────────────────────────────────
FLOW_DRIFT_MAGNITUDE: float = 0.05      # max kg/s change per tick

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
LOG_INTERVAL: int = 10                  # print state every N ticks
DEBUG_MODE: bool = True                 # verbose logging

# ─────────────────────────────────────────────
# Server
# ─────────────────────────────────────────────
API_HOST: str = "0.0.0.0"
API_PORT: int = 8000
