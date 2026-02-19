# 🏭 ENTROPY ENGINE — Person 1 (Backend) Full Execution Plan

> **Role:** Backend Simulation Engine Developer  
> **Project:** Waste Heat Recovery Plant Simulator  
> **Stack:** Python, FastAPI, Uvicorn, AsyncIO  
> **Timeline:** 3 Days  

---

## 📋 TABLE OF CONTENTS

1. [Project Overview](#1-project-overview)
2. [Architecture & Data Flow](#2-architecture--data-flow)
3. [File Structure](#3-file-structure)
4. [Phase 1 — Config & Constants](#4-phase-1--config--constants)
5. [Phase 2 — Data Models](#5-phase-2--data-models)
6. [Phase 3 — Simulation Engine](#6-phase-3--simulation-engine)
7. [Phase 4 — API Layer](#7-phase-4--api-layer)
8. [Phase 5 — Concurrency & Stability](#8-phase-5--concurrency--stability)
9. [Phase 6 — Logging & Debug](#9-phase-6--logging--debug)
10. [Phase 7 — Testing Checklist](#10-phase-7--testing-checklist)
11. [Phase 8 — AI Integration Prep](#11-phase-8--ai-integration-prep)
12. [API Contract (for Frontend/AI team)](#12-api-contract)
13. [Risk & Edge Cases](#13-risk--edge-cases)
14. [Day-by-Day Schedule](#14-day-by-day-schedule)

---

## 1. PROJECT OVERVIEW

We simulate a **waste heat recovery plant** with 4 core components:

| Component          | Role                        |
|--------------------|-----------------------------|
| **Furnace**        | Waste heat source           |
| **Heat Exchanger** | Transfers heat to working fluid |
| **Steam Drum**     | Pressure buildup zone       |
| **Turbine Generator** | Converts steam → electricity |

### System Variables

| Variable          | Unit   | Range      | Description              |
|-------------------|--------|------------|--------------------------|
| `temperature`     | °C     | 400–600    | Exhaust gas temperature   |
| `flow_rate`       | kg/s   | 2–5        | Gas mass flow rate        |
| `pressure`        | bar    | 4–8        | Steam drum pressure       |
| `valve_position`  | %      | 0–100      | Control valve opening     |
| `power_output`    | kW     | 150–300    | Turbine electrical output |

---

## 2. ARCHITECTURE & DATA FLOW

```
┌─────────────────────────────────────────────────────┐
│                  SIMULATION LOOP (1 Hz)              │
│                                                     │
│   ┌──────────┐    ┌──────────────┐    ┌──────────┐  │
│   │ Furnace  │───▶│Heat Exchanger│───▶│Steam Drum│  │
│   │ (heat)   │    │ (transfer)   │    │(pressure)│  │
│   └──────────┘    └──────────────┘    └────┬─────┘  │
│                                            │        │
│                                     ┌──────▼──────┐ │
│                                     │  Turbine    │ │
│                                     │ (power out) │ │
│                                     └─────────────┘ │
│                                                     │
│   Update State ──▶ Apply Physics ──▶ Add Noise     │
│   ──▶ Clamp Values ──▶ Store in current_state       │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │   FastAPI       │
              │                │
              │  GET /metrics  │◀─── Frontend polls
              │  POST /control │◀─── AI / Frontend sends
              └────────────────┘
```

**Key principle:** Simulation loop runs **independently** of API. API only reads/writes shared state.

---

## 3. FILE STRUCTURE

```
d:\BankokHack\
└── backend/
    ├── config.py               # All constants, tuning params
    ├── models.py               # Pydantic data models
    ├── simulation_engine.py    # Core physics engine + loop
    ├── api.py                  # FastAPI endpoints
    ├── requirements.txt        # Dependencies
    └── README.md               # Quick start guide
```

---

## 4. PHASE 1 — Config & Constants

**File:** `backend/config.py`  
**Time estimate:** 30 minutes  

### What to define:

```python
# ─── Simulation Timing ───
TICK_INTERVAL = 1.0              # seconds between updates

# ─── Ambient / Environment ───
AMBIENT_TEMPERATURE = 25.0       # °C

# ─── Heat Decay ───
HEAT_DECAY_CONSTANT = 0.02       # k in dT/dt = -k*(T - T_amb)

# ─── Initial Conditions ───
INITIAL_TEMPERATURE = 500.0      # °C
INITIAL_FLOW_RATE = 3.0          # kg/s
INITIAL_PRESSURE = 6.0           # bar
INITIAL_VALVE_POSITION = 50.0    # %
INITIAL_POWER_OUTPUT = 200.0     # kW

# ─── Physical Constants ───
PRESSURE_COEFFICIENT = 0.004     # c1: pressure = c1 * T * effective_flow
TURBINE_EFFICIENCY = 0.35        # η for power = η * P * flow

# ─── Safety Limits ───
MIN_TEMPERATURE = 400.0
MAX_TEMPERATURE = 600.0
MIN_FLOW_RATE = 2.0
MAX_FLOW_RATE = 5.0
MIN_PRESSURE = 4.0
MAX_PRESSURE = 8.0               # SAFETY CAP
MIN_POWER = 150.0
MAX_POWER = 300.0

# ─── Realism ───
SENSOR_NOISE_PERCENT = 0.02      # ±2% noise
HEAT_SPIKE_PROBABILITY = 0.02    # 2% chance per tick
HEAT_SPIKE_MAGNITUDE = 30.0      # °C spike
VALVE_RESPONSE_RATE = 0.1        # 10% per tick (inertia)
INERTIA_FACTOR = 0.85            # smoothing factor for state changes

# ─── Logging ───
LOG_INTERVAL = 10                # log every N ticks
DEBUG_MODE = True

# ─── Server ───
API_HOST = "0.0.0.0"
API_PORT = 8000
```

### Checklist:
- [ ] No magic numbers anywhere else in code
- [ ] Every constant has a comment
- [ ] Easy to tune during demo

---

## 5. PHASE 2 — Data Models

**File:** `backend/models.py`  
**Time estimate:** 20 minutes  

### Models to create:

```python
from pydantic import BaseModel, Field

class PlantMetrics(BaseModel):
    """Current state of the plant — returned by GET /metrics"""
    temperature: float = Field(..., ge=400, le=600, description="Exhaust temperature °C")
    pressure: float = Field(..., ge=4, le=8, description="Steam pressure bar")
    flow_rate: float = Field(..., ge=2, le=5, description="Gas flow rate kg/s")
    valve_position: float = Field(..., ge=0, le=100, description="Valve opening %")
    power_output: float = Field(..., ge=150, le=300, description="Turbine power kW")
    timestamp: float = Field(..., description="Unix timestamp of reading")

class ControlInput(BaseModel):
    """Control command — received by POST /control"""
    valve_position: float = Field(..., ge=0, le=100, description="Target valve %")

class SystemStatus(BaseModel):
    """Health/status response"""
    status: str
    uptime_seconds: float
    tick_count: int
    ai_mode: bool
```

### Checklist:
- [ ] All fields have validation ranges
- [ ] Descriptions present (shows in /docs)
- [ ] Timestamp included for frontend sync

---

## 6. PHASE 3 — Simulation Engine (CORE)

**File:** `backend/simulation_engine.py`  
**Time estimate:** 3–4 hours (most critical)  

### Class: `SimulationEngine`

#### 6.1 — Constructor `__init__`

```python
def __init__(self):
    # Internal true state (before noise)
    self._temperature = INITIAL_TEMPERATURE
    self._flow_rate = INITIAL_FLOW_RATE
    self._pressure = INITIAL_PRESSURE
    self._valve_position = INITIAL_VALVE_POSITION
    self._power_output = INITIAL_POWER_OUTPUT

    # Target valve (for inertia/delay)
    self._target_valve_position = INITIAL_VALVE_POSITION

    # Timing
    self._tick_count = 0
    self._start_time = time.time()
    self._running = False

    # Thread safety
    self._lock = asyncio.Lock()

    # Public noisy state (what API returns)
    self.current_state = {}
```

#### 6.2 — Physics Update `_update_physics()`

Execute **in this exact order** every tick:

```
Step 1: Heat Decay
    dT = -HEAT_DECAY_CONSTANT * (T - AMBIENT_TEMPERATURE) * dt
    T_new = T + dT

Step 2: Random Heat Spike (2% chance)
    if random() < HEAT_SPIKE_PROBABILITY:
        T_new += HEAT_SPIKE_MAGNITUDE

Step 3: Valve Inertia
    valve_current += VALVE_RESPONSE_RATE * (valve_target - valve_current)
    (valve doesn't jump instantly)

Step 4: Effective Flow
    effective_flow = flow_rate * (valve_position / 100)

Step 5: Pressure Calculation
    pressure = PRESSURE_COEFFICIENT * temperature * effective_flow

Step 6: Power Output
    power = TURBINE_EFFICIENCY * pressure * effective_flow

Step 7: Apply Inertia Smoothing
    value_new = INERTIA_FACTOR * value_old + (1 - INERTIA_FACTOR) * value_calculated

Step 8: Clamp All Values
    temperature = clamp(T, 400, 600)
    pressure = clamp(P, 4, 8)       ← SAFETY CAP
    flow_rate = clamp(F, 2, 5)
    power = clamp(W, 150, 300)

Step 9: Add Sensor Noise (for API output only)
    noisy_value = true_value * (1 + uniform(-NOISE, +NOISE))
```

#### 6.3 — Main Loop `run()`

```python
async def run(self):
    self._running = True
    while self._running:
        async with self._lock:
            self._update_physics()
            self._tick_count += 1

            if self._tick_count % LOG_INTERVAL == 0:
                self._log_state()

        await asyncio.sleep(TICK_INTERVAL)
```

#### 6.4 — Control Method `set_valve(position)`

```python
async def set_valve(self, position: float):
    async with self._lock:
        self._target_valve_position = clamp(position, 0, 100)
        # Note: actual valve moves gradually via inertia in _update_physics
```

#### 6.5 — State Getter `get_metrics()`

```python
async def get_metrics(self) -> dict:
    async with self._lock:
        return dict(self.current_state)  # returns noisy snapshot
```

### Detailed Physics Equations

```
┌─────────────────────────────────────────────────────┐
│ EQUATION 1: Heat Decay                              │
│                                                     │
│   dT/dt = -k × (T - T_ambient)                     │
│                                                     │
│   Where:                                            │
│     k = 0.02 (decay constant)                       │
│     T_ambient = 25°C                                │
│     dt = 1.0 second                                 │
│                                                     │
│   Effect: Temperature slowly drops toward ambient   │
│   unless heat source maintains it.                  │
├─────────────────────────────────────────────────────┤
│ EQUATION 2: Effective Flow                          │
│                                                     │
│   F_eff = flow_rate × (valve_position / 100)        │
│                                                     │
│   Effect: Valve controls how much flow actually     │
│   passes through the system.                        │
├─────────────────────────────────────────────────────┤
│ EQUATION 3: Pressure                                │
│                                                     │
│   P = c1 × T × F_eff                               │
│                                                     │
│   Where c1 = 0.004                                  │
│                                                     │
│   Effect: Higher temp + higher flow = more pressure │
│   Capped at 8 bar for safety.                       │
├─────────────────────────────────────────────────────┤
│ EQUATION 4: Power Output                            │
│                                                     │
│   W = η × P × F_eff                                 │
│                                                     │
│   Where η = 0.35 (turbine efficiency)               │
│                                                     │
│   Effect: Power depends on both pressure and flow.  │
└─────────────────────────────────────────────────────┘
```

### Realism Features — Implementation Detail

| Feature | How to Implement | Why |
|---------|-----------------|-----|
| **Sensor Noise** | Multiply each output value by `(1 + random.uniform(-0.02, 0.02))` | Real sensors are noisy |
| **Heat Spikes** | 2% chance per tick: add 30°C to temperature | Simulates furnace bursts |
| **Valve Delay** | Move valve 10% toward target per tick, not instant | Real valves are slow |
| **Pressure Cap** | `min(pressure, 8.0)` always enforced | Safety relief valve |
| **Inertia** | `new = 0.85 * old + 0.15 * calculated` | Thermal mass, no instant jumps |

### Checklist:
- [ ] All 4 equations implemented
- [ ] Clamp function works for all variables
- [ ] Noise only on API output, not internal state
- [ ] Valve target vs actual separated
- [ ] Lock protects all shared state
- [ ] Heat spike randomness working
- [ ] Loop runs at exactly 1 Hz
- [ ] Tick counter incrementing
- [ ] Log output every 10 ticks

---

## 7. PHASE 4 — API Layer

**File:** `backend/api.py`  
**Time estimate:** 1.5 hours  

### 7.1 — App Setup

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Create engine instance (singleton)
engine = SimulationEngine()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start simulation on server boot
    task = asyncio.create_task(engine.run())
    yield
    # Shutdown
    engine.stop()
    task.cancel()

app = FastAPI(
    title="Entropy Engine",
    description="Industrial Waste Heat Recovery Simulator",
    version="1.0.0",
    lifespan=lifespan
)

# CORS — allow frontend from any origin during hackathon
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 7.2 — Endpoints

#### `GET /metrics`

```python
@app.get("/metrics", response_model=PlantMetrics)
async def get_metrics():
    state = await engine.get_metrics()
    return state
```

**Response:**
```json
{
  "temperature": 512.3,
  "pressure": 6.31,
  "flow_rate": 3.12,
  "valve_position": 55.0,
  "power_output": 224.8,
  "timestamp": 1740000000.123
}
```

#### `POST /control`

```python
@app.post("/control")
async def set_control(input: ControlInput):
    await engine.set_valve(input.valve_position)
    return {"status": "accepted", "target_valve_position": input.valve_position}
```

**Request:**
```json
{
  "valve_position": 65
}
```

**Response:**
```json
{
  "status": "accepted",
  "target_valve_position": 65
}
```

#### `GET /status`

```python
@app.get("/status", response_model=SystemStatus)
async def get_status():
    return {
        "status": "running",
        "uptime_seconds": engine.uptime,
        "tick_count": engine.tick_count,
        "ai_mode": False
    }
```

#### `GET /health`

```python
@app.get("/health")
async def health_check():
    return {"status": "ok"}
```

### Checklist:
- [ ] CORS enabled (frontend needs this)
- [ ] Simulation starts automatically with server
- [ ] Simulation stops cleanly on shutdown
- [ ] All endpoints return proper JSON
- [ ] FastAPI docs accessible at `/docs`
- [ ] Response time < 100ms (just reading dict)

---

## 8. PHASE 5 — Concurrency & Stability

**Time estimate:** 1 hour  

### Architecture Decision: AsyncIO (not threading)

```
┌──────────────────────────────────┐
│         Python AsyncIO           │
│                                  │
│  Task 1: Simulation Loop         │
│    └─ runs every 1 second        │
│                                  │
│  Task 2: Uvicorn (FastAPI)       │
│    └─ handles HTTP requests      │
│                                  │
│  Shared: engine.current_state    │
│    └─ protected by asyncio.Lock  │
└──────────────────────────────────┘
```

### Why asyncio.Lock (not threading.Lock)?

- FastAPI is async-native
- Simulation loop uses `await asyncio.sleep()`
- Both tasks run in same event loop
- No GIL issues, no race conditions
- Simpler than threading

### Critical Rules:
1. **Never** do blocking I/O in the simulation loop
2. **Never** use `time.sleep()` — always `await asyncio.sleep()`
3. **Always** acquire lock before reading/writing state
4. Lock acquisition should be fast (< 1ms)

### Error Handling in Loop:

```python
async def run(self):
    self._running = True
    while self._running:
        try:
            async with self._lock:
                self._update_physics()
                self._tick_count += 1
            await asyncio.sleep(TICK_INTERVAL)
        except Exception as e:
            logger.error(f"Simulation error: {e}")
            await asyncio.sleep(TICK_INTERVAL)  # don't crash, keep going
```

### Checklist:
- [ ] Simulation survives exceptions
- [ ] API works even if simulation has error
- [ ] No deadlocks possible
- [ ] Clean shutdown via `engine.stop()`

---

## 9. PHASE 6 — Logging & Debug

**Time estimate:** 30 minutes  

### Logging Setup

```python
import logging

logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("entropy-engine")
```

### What to Log

| When | What | Level |
|------|------|-------|
| Every 10 ticks | Full state snapshot | INFO |
| Valve change | Old → New target | INFO |
| Heat spike | "🔥 Heat spike! +30°C" | WARNING |
| Pressure cap hit | "⚠️ Pressure capped at 8 bar" | WARNING |
| API request | Endpoint + response time | DEBUG |
| Error | Stack trace | ERROR |
| Startup | Config summary | INFO |

### Sample Log Output

```
14:23:01 | INFO    | Engine started | T=500.0°C P=6.0bar V=50%
14:23:11 | INFO    | Tick 10 | T=498.2°C P=5.97bar F=3.0kg/s V=50.0% W=199.1kW
14:23:15 | WARNING | 🔥 Heat spike! T jumped to 528.2°C
14:23:21 | INFO    | Tick 20 | T=521.4°C P=6.23bar F=3.0kg/s V=50.0% W=218.7kW
14:23:25 | INFO    | Valve target changed: 50.0% → 65.0%
14:23:31 | INFO    | Tick 30 | T=515.8°C P=6.55bar F=3.0kg/s V=58.2% W=231.2kW
```

### Checklist:
- [ ] Logs are readable and timestamped
- [ ] No spam (only every 10 ticks)
- [ ] Warnings for anomalies
- [ ] Debug mode toggleable from config

---

## 10. PHASE 7 — Testing Checklist

**Time estimate:** 1.5 hours  

### 10.1 — Manual Testing via `/docs`

1. Start server: `python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000`
2. Open: `http://localhost:8000/docs`
3. Test each endpoint interactively

### 10.2 — Test Scenarios

| # | Test | Method | Expected Result |
|---|------|--------|-----------------|
| 1 | Get initial metrics | GET /metrics | All values in range |
| 2 | Wait 10s, get metrics again | GET /metrics | Values slightly changed (decay) |
| 3 | Set valve to 100% | POST /control `{"valve_position": 100}` | Accepted, valve moves gradually |
| 4 | Set valve to 0% | POST /control `{"valve_position": 0}` | Pressure/power drop over time |
| 5 | Send invalid valve (-10) | POST /control `{"valve_position": -10}` | 422 validation error |
| 6 | Send valve > 100 | POST /control `{"valve_position": 150}` | 422 validation error |
| 7 | Rapid polling (10 req/s) | GET /metrics in loop | All responses < 100ms |
| 8 | Check status | GET /status | uptime increasing, tick count > 0 |
| 9 | Health check | GET /health | `{"status": "ok"}` |
| 10 | Wait for heat spike | Watch logs | Spike appears within ~50 ticks |

### 10.3 — curl Commands for Quick Testing

```bash
# Get metrics
curl http://localhost:8000/metrics

# Set valve
curl -X POST http://localhost:8000/control \
  -H "Content-Type: application/json" \
  -d '{"valve_position": 75}'

# Health check  
curl http://localhost:8000/health

# Status
curl http://localhost:8000/status
```

### 10.4 — Automated Test Script (optional, bonus)

```python
# test_simulation.py
import requests, time

BASE = "http://localhost:8000"

# Test 1: Metrics in range
r = requests.get(f"{BASE}/metrics").json()
assert 400 <= r["temperature"] <= 600
assert 4 <= r["pressure"] <= 8
assert 0 <= r["valve_position"] <= 100
print("✅ Metrics in range")

# Test 2: Control accepted
r = requests.post(f"{BASE}/control", json={"valve_position": 70})
assert r.status_code == 200
print("✅ Control accepted")

# Test 3: Values change over time
m1 = requests.get(f"{BASE}/metrics").json()
time.sleep(5)
m2 = requests.get(f"{BASE}/metrics").json()
assert m1["timestamp"] != m2["timestamp"]
print("✅ State updating over time")

print("\n🎉 All tests passed!")
```

### Checklist:
- [ ] All endpoints return 200
- [ ] Values stay within defined ranges
- [ ] Valve changes take effect gradually
- [ ] No crashes after 5 minutes of running
- [ ] `/docs` page loads and looks professional

---

## 11. PHASE 8 — AI Integration Prep

**Time estimate:** 30 minutes  

### Simple Auto-Control Mode

Even before AI model is ready, implement a basic rule-based controller:

```python
# Inside simulation engine or as separate module

async def ai_auto_control(engine):
    """Simple rule-based AI placeholder"""
    while True:
        state = await engine.get_metrics()
        temp = state["temperature"]

        if temp > 550:
            # Too hot → open valve to release pressure
            await engine.set_valve(min(state["valve_position"] + 5, 100))
        elif temp < 450:
            # Too cold → close valve to build pressure
            await engine.set_valve(max(state["valve_position"] - 5, 0))

        await asyncio.sleep(2)  # AI decides every 2 seconds
```

### Integration Points for Real AI

```
AI System connects via:
  POST /control → {"valve_position": X}

AI System reads via:
  GET /metrics → full plant state

No code changes needed in backend.
The API contract is the integration boundary.
```

### Checklist:
- [ ] Auto-control mode works as demo
- [ ] Can toggle AI mode on/off
- [ ] API contract documented for AI team
- [ ] No backend changes needed when real AI connects

---

## 12. API CONTRACT

> **Share this section with Frontend and AI team members.**

### Base URL
```
http://localhost:8000
```

### Endpoints

#### `GET /metrics`
Returns current plant sensor readings.

```json
{
  "temperature": 512.3,
  "pressure": 6.31,
  "flow_rate": 3.12,
  "valve_position": 55.0,
  "power_output": 224.8,
  "timestamp": 1740000000.123
}
```

#### `POST /control`
Send a control command to adjust the valve.

**Request Body:**
```json
{
  "valve_position": 65.0
}
```

**Response:**
```json
{
  "status": "accepted",
  "target_valve_position": 65.0
}
```

**Validation:** `valve_position` must be 0–100. Returns 422 if invalid.

#### `GET /status`
System health and uptime.

```json
{
  "status": "running",
  "uptime_seconds": 342.5,
  "tick_count": 342,
  "ai_mode": false
}
```

#### `GET /health`
Simple liveness check.

```json
{
  "status": "ok"
}
```

### Polling Recommendation
- Frontend: Poll `GET /metrics` every **1–2 seconds**
- AI: Poll every **2–5 seconds**, send control every **2–5 seconds**

---

## 13. RISK & EDGE CASES

| Risk | Mitigation |
|------|-----------|
| Simulation drifts out of range | Clamp all values every tick |
| Pressure exceeds safety limit | Hard cap at 8 bar (safety relief valve simulation) |
| API called before engine starts | Return last known state or default values |
| Multiple rapid valve commands | Only target changes; inertia smooths actual movement |
| Frontend disconnects | Simulation continues independently |
| Division by zero (valve at 0%) | effective_flow = 0, pressure/power = 0, then clamped to min |
| Python float precision | Round display values to 2 decimal places |
| Server crash mid-tick | Exception handler in loop prevents crash, logs error |
| High request load | State is pre-computed; GET just returns dict (< 1ms) |

---

## 14. DAY-BY-DAY SCHEDULE

### DAY 1 (Focus: Core Engine)

| Time | Task | Deliverable |
|------|------|-------------|
| Hour 1 | Set up project folder, virtualenv, install deps | `requirements.txt` ready |
| Hour 2 | Write `config.py` with all constants | All tuning params defined |
| Hour 3 | Write `models.py` with Pydantic schemas | Data contracts ready |
| Hour 4–6 | Build `simulation_engine.py` | Physics loop running standalone |
| Hour 7 | Test engine in isolation (print state) | Verify values change correctly |
| Hour 8 | Add noise, spikes, inertia | Realistic behavior confirmed |

**Day 1 Exit Criteria:**
- [ ] Engine runs and prints state every second
- [ ] Temperature decays naturally
- [ ] Valve inertia works
- [ ] Heat spikes occur randomly
- [ ] All values stay in range

---

### DAY 2 (Focus: API + Integration)

| Time | Task | Deliverable |
|------|------|-------------|
| Hour 1–2 | Build `api.py` with all endpoints | Server starts, `/docs` works |
| Hour 3 | Wire simulation as background task | Engine + API run together |
| Hour 4 | Test all endpoints via `/docs` | All return correct JSON |
| Hour 5 | Add CORS, error handling | Frontend can connect |
| Hour 6 | Add logging system | Clean console output |
| Hour 7 | Implement basic AI auto-control | Demo-ready AI placeholder |
| Hour 8 | Run full integration test | All test scenarios pass |

**Day 2 Exit Criteria:**
- [ ] `GET /metrics` returns live updating data
- [ ] `POST /control` changes valve gradually
- [ ] `/docs` interactive page works
- [ ] Logs print every 10 seconds
- [ ] No crashes for 10+ minutes

---

### DAY 3 (Focus: Polish + Demo Prep)

| Time | Task | Deliverable |
|------|------|-------------|
| Hour 1–2 | Fix any bugs from Day 2 | Stable system |
| Hour 3 | Code cleanup, comments, docstrings | Professional code |
| Hour 4 | Tune physics constants for demo | Visually interesting data |
| Hour 5 | Write test script | Automated verification |
| Hour 6 | Coordinate with frontend team | API contract confirmed |
| Hour 7 | Final end-to-end test | Full flow working |
| Hour 8 | Buffer / demo rehearsal | Ready to present |

**Day 3 Exit Criteria:**
- [ ] Zero crashes in 30-minute run
- [ ] Data looks realistic and interesting
- [ ] Code is clean and documented
- [ ] Frontend team has confirmed API works
- [ ] Ready for live demo

---

## 🚀 QUICK START COMMANDS

```bash
# 1. Setup
cd d:\BankokHack\backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 2. Run
uvicorn api:app --reload --host 0.0.0.0 --port 8000

# 3. Test
# Open http://localhost:8000/docs in browser

# 4. Monitor
# Watch console for log output every 10 seconds
```

---

## 📦 REQUIREMENTS.TXT

```
fastapi==0.115.0
uvicorn[standard]==0.30.0
pydantic==2.9.0
```

---

## ✅ FINAL DELIVERY CHECKLIST

- [ ] `config.py` — all constants, no magic numbers
- [ ] `models.py` — Pydantic models with validation
- [ ] `simulation_engine.py` — physics loop, noise, safety
- [ ] `api.py` — 4 endpoints, CORS, background task
- [ ] `requirements.txt` — pinned versions
- [ ] Server starts with one command
- [ ] `/docs` page works
- [ ] Values stay in range forever
- [ ] Valve responds with delay (realistic)
- [ ] Logs are clean and informative
- [ ] API response < 100ms
- [ ] No crashes after extended run
- [ ] Frontend team can connect and poll
- [ ] AI team can send POST /control
- [ ] Code is demo-ready

---

*Entropy Engine — Backend Simulation*  
*Version 1.0 | February 2026*
