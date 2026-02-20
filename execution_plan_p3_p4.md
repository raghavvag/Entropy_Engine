# ENTROPY ENGINE — EXECUTION PLAN
## Person 3: System Integration + Control Orchestration
## Person 4: Frontend + 3D Experience + Product Design

**Version:** 1.0  
**Date:** February 2026  
**Status:** Planning

---

## TABLE OF CONTENTS

1. [Current State Assessment](#1-current-state-assessment)
2. [Person 3 — Phase 1: Orchestrator Layer](#2-person-3--phase-1-orchestrator-layer)
3. [Person 3 — Phase 2: AI Mode Controller](#3-person-3--phase-2-ai-mode-controller)
4. [Person 3 — Phase 3: Safety Fallback System](#4-person-3--phase-3-safety-fallback-system)
5. [Person 3 — Phase 4: Model Confidence Monitor](#5-person-3--phase-4-model-confidence-monitor)
6. [Person 3 — Phase 5: Logging & Monitoring](#6-person-3--phase-5-logging--monitoring)
7. [Person 3 — Phase 6: Environment Configuration](#7-person-3--phase-6-environment-configuration)
8. [Person 3 — Phase 7: Deployment (Docker)](#8-person-3--phase-7-deployment-docker)
9. [Person 4 — Phase 1: Core Dashboard UI](#9-person-4--phase-1-core-dashboard-ui)
10. [Person 4 — Phase 2: 3D Factory Visualization](#10-person-4--phase-2-3d-factory-visualization)
11. [Person 4 — Phase 3: Professional UI Design](#11-person-4--phase-3-professional-ui-design)
12. [Person 4 — Phase 4: Before vs After Comparison](#12-person-4--phase-4-before-vs-after-comparison)
13. [Person 4 — Phase 5: Business Metrics Section](#13-person-4--phase-5-business-metrics-section)
14. [Person 4 — Phase 6: Demo Animation Flow](#14-person-4--phase-6-demo-animation-flow)
15. [Risk & Edge Cases](#15-risk--edge-cases)
16. [Day-by-Day Schedule](#16-day-by-day-schedule)
17. [Final Delivery Checklist](#17-final-delivery-checklist)

---

## 1. CURRENT STATE ASSESSMENT

### What Already Exists

```
d:\BankokHack\
├── backend/                    ← PERSON 1 (COMPLETE ✅)
│   ├── config.py               # Physics constants
│   ├── models.py               # Pydantic schemas (PlantMetrics, ControlInput, SystemStatus)
│   ├── simulation_engine.py    # 1 Hz physics loop (heat decay, pressure, power)
│   ├── api.py                  # FastAPI — 5 endpoints
│   └── requirements.txt        # fastapi, uvicorn, pydantic
│
├── ai/                         ← PERSON 2 (COMPLETE ✅)
│   ├── config.py               # AI hyperparams
│   ├── baseline_controller.py  # Rule-based heuristic
│   ├── data_collector.py       # Polls /metrics → CSV
│   ├── model.py                # PyTorch PlantDynamicsModel
│   ├── train.py                # Training loop
│   ├── pinn_loss.py            # Physics + safety loss
│   ├── mpc_controller.py       # Model Predictive Control
│   ├── control_loop.py         # Main AI loop (heuristic/mpc/hybrid)
│   ├── safety.py               # Hard safety overrides
│   ├── utils.py                # AIReport metrics class
│   ├── test_ai_integration.py  # E2E test
│   ├── data/training_data.csv  # 1500 samples
│   └── models/pinn_model.pt    # Trained checkpoint
```

### Existing API Contract (Person 1 Backend)

| Endpoint             | Method | Purpose                     |
|----------------------|--------|-----------------------------|
| `/metrics`           | GET    | Current plant state (6 fields) |
| `/control`           | POST   | Set valve_position (0–100)  |
| `/status`            | GET    | Uptime, tick count, ai_mode |
| `/ai-mode`           | POST   | Toggle built-in auto-control |
| `/health`            | GET    | Liveness probe              |

### Existing AI Capabilities (Person 2)

- 3 modes: `heuristic`, `mpc`, `hybrid`
- PINN-trained model with physics + safety loss
- Safety enforcement in `ai/safety.py`
- `control_loop.py` already polls → decides → sends

### What Person 3 Adds ON TOP

Person 2's `control_loop.py` is the *AI brain*. Person 3 builds the **orchestration shell** around it — the production-grade wrapper that:

1. Exposes a **unified API** for the frontend (Person 4) to consume
2. Manages **AI mode toggle** via HTTP (not just CLI)
3. Adds **enhanced fallback** with structured error recovery
4. Runs a **confidence monitor** that auto-disables AI
5. Provides **structured logging** and health dashboards
6. Wraps everything in **.env config** and **Docker**

### What Person 4 Builds

The React frontend that consumes Person 3's orchestrator API and Person 1's simulation API to display:
- Real-time KPI cards
- Live charts
- AI toggle
- 3D factory visualization
- Before/After comparison
- Business metrics

---

## 2. PERSON 3 — PHASE 1: Orchestrator Layer

**Directory:** `integrator/`  
**Time estimate:** 1.5 hours  
**Goal:** Create the central orchestration service that bridges AI ↔ Backend ↔ Frontend.

### File Structure

```
integrator/
├── orchestrator.py      # Main orchestration loop + FastAPI endpoints
├── config.py            # Environment-driven configuration
├── safety.py            # Enhanced safety fallback system
├── confidence.py        # Model confidence monitoring
├── logger.py            # Structured logging
├── requirements.txt     # Dependencies
├── .env                 # Environment variables
└── .env.example         # Template for team
```

### orchestrator.py — Core Design

```python
"""
Entropy Engine — Control Orchestrator
======================================
Central service that:
  1. Polls simulation API for plant state
  2. Forwards state to AI controller
  3. Applies safety constraints
  4. Sends final valve command
  5. Exposes orchestrator API for frontend

Endpoints for Frontend (Person 4):
  GET  /api/state         → current plant state + AI decision + safety
  POST /api/ai/toggle     → enable/disable AI mode
  GET  /api/ai/status     → AI mode, confidence, safety level
  GET  /api/history       → recent decision history
  GET  /api/comparison    → baseline vs AI metrics
  GET  /api/health        → orchestrator health
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import httpx
import asyncio
import time
import logging

app = FastAPI(title="Entropy Engine Orchestrator", version="1.0.0")

# ── State ──
ai_enabled: bool = False
control_history: list[dict] = []
baseline_power: float | None = None
ai_power_readings: list[float] = []
current_safety_level: str = "NORMAL"


class Orchestrator:
    """
    Main control loop that runs as a background task.
    
    States:
      - IDLE:    AI off, just collecting metrics for baseline
      - ACTIVE:  AI on, making control decisions
      - FALLBACK: AI failed, using safe valve position
    """
    
    def __init__(self):
        self.state = "IDLE"        # IDLE | ACTIVE | FALLBACK
        self.ai_mode = False
        self.last_metrics = {}
        self.last_decision = {}
        self.history = []          # last 300 ticks (5 min)
        self.baseline_readings = []
        self.ai_readings = []
        self.confidence = 0.0
        self.safety_level = "NORMAL"
        self.error_count = 0
        self.fallback_valve = 50.0  # safe default
        self.start_time = time.time()
    
    async def tick(self):
        """One orchestration cycle."""
        
        # 1. Read plant state from simulation
        metrics = await self._fetch_metrics()
        if not metrics:
            return  # backend down, skip tick
        
        self.last_metrics = metrics
        
        # 2. Decide valve position
        if self.ai_mode and self.state == "ACTIVE":
            decision = await self._run_ai(metrics)
        else:
            decision = {"valve": metrics["valve_position"], "mode": "manual"}
        
        # 3. Apply safety
        decision = self._apply_safety(metrics, decision)
        
        # 4. Send control command (only if AI is active)
        if self.ai_mode:
            await self._send_control(decision["valve"])
        
        # 5. Record history
        self._record(metrics, decision)
    
    async def _fetch_metrics(self) -> dict | None:
        """Poll GET /metrics with timeout handling."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(SIM_METRICS_URL)
                resp.raise_for_status()
                self.error_count = 0
                return resp.json()
        except Exception as e:
            self.error_count += 1
            if self.error_count > 5:
                self._enter_fallback("Backend unreachable")
            return None
    
    async def _run_ai(self, metrics: dict) -> dict:
        """Call AI model for optimal valve."""
        try:
            # Use MPC controller directly
            from ai_bridge import get_ai_decision
            decision = get_ai_decision(metrics)
            self.confidence = decision.get("confidence", 0)
            return decision
        except Exception as e:
            self._enter_fallback(f"AI error: {e}")
            return {"valve": self.fallback_valve, "mode": "fallback"}
    
    def _apply_safety(self, metrics, decision):
        """Enhanced safety with structured fallback."""
        pressure = metrics["pressure"]
        temp = metrics["temperature"]
        
        if pressure > 7.8:
            decision["valve"] = max(0, metrics["valve_position"] - 10)
            decision["safety_override"] = True
            self.safety_level = "CRITICAL"
        elif pressure > 7.5:
            if decision["valve"] > metrics["valve_position"]:
                decision["valve"] = metrics["valve_position"]
            decision["safety_override"] = True
            self.safety_level = "WARNING"
        elif temp > 590:
            decision["valve"] = min(decision["valve"],
                                     metrics["valve_position"] - 5)
            self.safety_level = "WARNING"
        else:
            self.safety_level = "NORMAL"
            decision["safety_override"] = False
        
        decision["valve"] = max(0, min(100, decision["valve"]))
        return decision
```

### API Endpoints for Frontend

```python
# ── Endpoints that Person 4 consumes ──

@app.get("/api/state")
async def get_full_state():
    """Everything the frontend needs in one call."""
    return {
        "metrics": orchestrator.last_metrics,
        "ai_decision": orchestrator.last_decision,
        "ai_mode": orchestrator.ai_mode,
        "safety_level": orchestrator.safety_level,
        "confidence": orchestrator.confidence,
        "state": orchestrator.state,       # IDLE | ACTIVE | FALLBACK
        "uptime": time.time() - orchestrator.start_time,
    }


@app.post("/api/ai/toggle")
async def toggle_ai(enable: bool):
    """Frontend toggle switch → this endpoint."""
    if enable:
        orchestrator.ai_mode = True
        orchestrator.state = "ACTIVE"
        # Snapshot current power as baseline
        if orchestrator.last_metrics:
            orchestrator.baseline_power = orchestrator.last_metrics["power_output"]
    else:
        orchestrator.ai_mode = False
        orchestrator.state = "IDLE"
    return {"ai_mode": orchestrator.ai_mode, "state": orchestrator.state}


@app.get("/api/ai/status")
async def ai_status():
    """Detailed AI status for dashboard."""
    return {
        "mode": orchestrator.state,
        "ai_enabled": orchestrator.ai_mode,
        "confidence": orchestrator.confidence,
        "safety_level": orchestrator.safety_level,
        "error_count": orchestrator.error_count,
        "total_decisions": len(orchestrator.history),
    }


@app.get("/api/history")
async def get_history(limit: int = 60):
    """Recent decision history for charts."""
    return orchestrator.history[-limit:]


@app.get("/api/comparison")
async def get_comparison():
    """Baseline vs AI performance comparison."""
    baseline_avg = (sum(orchestrator.baseline_readings) /
                    len(orchestrator.baseline_readings)
                    if orchestrator.baseline_readings else 0)
    ai_avg = (sum(orchestrator.ai_readings) /
              len(orchestrator.ai_readings)
              if orchestrator.ai_readings else 0)
    improvement = ((ai_avg - baseline_avg) / max(baseline_avg, 1) * 100
                   if baseline_avg > 0 else 0)
    
    return {
        "baseline_avg_power": round(baseline_avg, 1),
        "ai_avg_power": round(ai_avg, 1),
        "improvement_pct": round(improvement, 1),
        "baseline_samples": len(orchestrator.baseline_readings),
        "ai_samples": len(orchestrator.ai_readings),
    }


@app.get("/api/health")
async def orchestrator_health():
    return {
        "orchestrator": "ok",
        "ai_loaded": orchestrator.ai_mode,
        "backend_connected": orchestrator.error_count == 0,
        "uptime": round(time.time() - orchestrator.start_time, 1),
    }
```

### Checklist:
- [ ] Orchestrator polls /metrics every 1 second
- [ ] AI decision forwarded to /control
- [ ] Frontend gets unified /api/state endpoint
- [ ] /api/ai/toggle works
- [ ] /api/history returns chart-ready data
- [ ] /api/comparison returns baseline vs AI

---

## 3. PERSON 3 — PHASE 2: AI Mode Controller

**Time estimate:** 1 hour  
**Goal:** Clean toggle between manual and AI control.

### State Machine

```
                ┌──────────┐
                │   IDLE   │ ← Default (AI off)
                └────┬─────┘
                     │ POST /api/ai/toggle {enable: true}
                     ▼
                ┌──────────┐
                │  ACTIVE  │ ← AI making decisions
                └────┬─────┘
                     │ AI crash / timeout / low confidence
                     ▼
                ┌──────────┐
                │ FALLBACK │ ← Safe valve, auto-recovery attempted
                └────┬─────┘
                     │ Recovery success
                     ▼
                ┌──────────┐
                │  ACTIVE  │
                └──────────┘
```

### Mode Logic

```python
class AIModeManger:
    """
    Manages transitions between control modes.
    """
    
    def __init__(self):
        self.mode = "IDLE"          # IDLE | ACTIVE | FALLBACK
        self.ai_enabled = False
        self.fallback_reason = None
        self.recovery_attempts = 0
        self.max_recovery = 3
    
    def enable_ai(self):
        self.ai_enabled = True
        self.mode = "ACTIVE"
        self.recovery_attempts = 0
        self.fallback_reason = None
        logger.info("[AI-MODE] ✅ AI ACTIVATED")
    
    def disable_ai(self):
        self.ai_enabled = False
        self.mode = "IDLE"
        logger.info("[AI-MODE] ⏹ AI DEACTIVATED")
    
    def enter_fallback(self, reason: str):
        self.mode = "FALLBACK"
        self.fallback_reason = reason
        logger.warning("[AI-MODE] ⚠️ FALLBACK: %s", reason)
    
    def attempt_recovery(self) -> bool:
        """Try to return to ACTIVE after fallback."""
        if self.recovery_attempts >= self.max_recovery:
            logger.error("[AI-MODE] ❌ Max recovery attempts — staying in FALLBACK")
            return False
        self.recovery_attempts += 1
        self.mode = "ACTIVE"
        logger.info("[AI-MODE] 🔄 Recovery attempt %d/%d",
                     self.recovery_attempts, self.max_recovery)
        return True
    
    def get_status(self) -> dict:
        return {
            "mode": self.mode,
            "ai_enabled": self.ai_enabled,
            "fallback_reason": self.fallback_reason,
            "recovery_attempts": self.recovery_attempts,
        }
```

### How Frontend Uses It

```
User clicks "AI: ON"
  → POST /api/ai/toggle {"enable": true}
  → Orchestrator: mode = ACTIVE
  → Control loop starts making AI decisions
  → Frontend polls /api/state every second
  → Dashboard shows: "AI Optimization: ACTIVE 🟢"

AI crashes:
  → Orchestrator: mode = FALLBACK
  → Valve set to safe position (50%)
  → Frontend shows: "AI Optimization: FALLBACK ⚠️"
  → Auto-recovery attempted after 10 seconds

User clicks "AI: OFF"
  → POST /api/ai/toggle {"enable": false}
  → Orchestrator: mode = IDLE
  → No control commands sent
  → Frontend shows: "AI Optimization: OFF ⚪"
```

### Checklist:
- [ ] IDLE → ACTIVE transition works
- [ ] ACTIVE → FALLBACK on error
- [ ] FALLBACK → ACTIVE recovery
- [ ] Manual disable always works
- [ ] Frontend gets mode status

---

## 4. PERSON 3 — PHASE 3: Safety Fallback System

**Time estimate:** 1 hour  
**Goal:** Production-grade failsafe that judges will love.

### Trigger Conditions

| Trigger | Threshold | Action |
|---------|-----------|--------|
| Pressure critical | > 7.8 bar | Force valve -10%, enter FALLBACK |
| Pressure high | > 7.5 bar | Block valve increase |
| Temperature critical | > 590°C | Force valve -5% |
| AI timeout | > 3 seconds | Use safe valve (50%) |
| AI crash | Exception | Use safe valve, log error |
| Confidence low | < 0.01 | Fall back to heuristic |
| Backend down | 5+ failures | Stop sending commands |

### Enhanced Safety Implementation

```python
class SafetyFallback:
    """
    Production safety layer.
    Sits between AI and simulation — cannot be bypassed.
    """
    
    SAFE_VALVE = 50.0  # default safe position
    
    def __init__(self):
        self.overrides = 0
        self.last_override_reason = None
        self.consecutive_overrides = 0
    
    def check(self, metrics: dict, proposed_valve: float) -> tuple[float, dict]:
        """
        Returns (safe_valve, safety_report).
        """
        report = {
            "original_valve": proposed_valve,
            "final_valve": proposed_valve,
            "overridden": False,
            "reason": None,
            "level": "NORMAL",
        }
        
        pressure = metrics.get("pressure", 0)
        temp = metrics.get("temperature", 0)
        current = metrics.get("valve_position", 50)
        valve = proposed_valve
        
        # CRITICAL: Pressure emergency
        if pressure > 7.8:
            valve = max(0, current - 10)
            report.update(overridden=True, reason="PRESSURE_CRITICAL",
                         level="CRITICAL")
        
        # WARNING: Pressure high
        elif pressure > 7.5:
            if valve > current:
                valve = current
            report.update(overridden=True, reason="PRESSURE_HIGH",
                         level="WARNING")
        
        # WARNING: Temperature critical
        if temp > 590:
            valve = min(valve, current - 5)
            report.update(overridden=True, reason="TEMP_CRITICAL",
                         level="WARNING")
        
        # CLAMP
        valve = max(0, min(100, round(valve, 2)))
        
        report["final_valve"] = valve
        if report["overridden"]:
            self.overrides += 1
            self.consecutive_overrides += 1
        else:
            self.consecutive_overrides = 0
        
        return valve, report
    
    def get_stats(self) -> dict:
        return {
            "total_overrides": self.overrides,
            "consecutive_overrides": self.consecutive_overrides,
            "last_reason": self.last_override_reason,
        }
```

### Fallback Decision Tree

```
AI proposes valve = 75%

Is pressure > 7.8?
  YES → valve = current - 10%  (EMERGENCY)
  NO  → continue

Is pressure > 7.5?
  YES → is 75% > current?
    YES → valve = current  (block increase)
    NO  → continue
  NO → continue

Is temperature > 590°C?
  YES → valve = min(valve, current - 5%)
  NO  → continue

Is valve in [0, 100]?
  Clamp if needed

Send final valve to simulation
```

### Checklist:
- [ ] Pressure > 7.8 forces valve reduction
- [ ] Pressure > 7.5 blocks increase
- [ ] Temperature > 590 reduces valve
- [ ] AI timeout → safe valve 50%
- [ ] Safety stats tracked
- [ ] Override count available for dashboard

---

## 5. PERSON 3 — PHASE 4: Model Confidence Monitor

**Time estimate:** 45 minutes  
**Goal:** Auto-disable AI when predictions are unreliable.

### Implementation

```python
class ConfidenceMonitor:
    """
    Tracks AI prediction quality in real-time.
    
    Confidence score = 1 - (prediction_error / actual_power).
    When confidence drops below threshold, AI is temporarily disabled.
    """
    
    THRESHOLD = 0.3          # below this → disable AI
    WINDOW = 30              # rolling average over 30 ticks
    RECOVERY_WINDOW = 10     # need 10 good predictions to re-enable
    
    def __init__(self):
        self.errors = []
        self.confidence_history = []
        self.disabled = False
        self.disable_reason = None
    
    def update(self, predicted_power: float | None, actual_power: float) -> float:
        """
        Record one prediction and return current confidence.
        """
        if predicted_power is None:
            return 0.0
        
        error = abs(predicted_power - actual_power)
        relative_error = error / max(actual_power, 1)
        confidence = max(0, 1 - relative_error)
        
        self.errors.append(error)
        self.confidence_history.append(confidence)
        
        # Keep window
        if len(self.confidence_history) > self.WINDOW:
            self.confidence_history = self.confidence_history[-self.WINDOW:]
            self.errors = self.errors[-self.WINDOW:]
        
        avg_confidence = sum(self.confidence_history) / len(self.confidence_history)
        
        # Auto-disable
        if avg_confidence < self.THRESHOLD and not self.disabled:
            self.disabled = True
            self.disable_reason = f"Confidence {avg_confidence:.2f} < {self.THRESHOLD}"
        
        return avg_confidence
    
    def should_use_ai(self) -> bool:
        return not self.disabled
    
    def get_report(self) -> dict:
        avg_conf = (sum(self.confidence_history) / len(self.confidence_history)
                    if self.confidence_history else 0)
        avg_err = (sum(self.errors) / len(self.errors)
                   if self.errors else 0)
        return {
            "confidence": round(avg_conf, 4),
            "avg_error_kw": round(avg_err, 2),
            "disabled": self.disabled,
            "disable_reason": self.disable_reason,
            "samples": len(self.confidence_history),
        }
```

### How It Integrates

```
Every tick:
  1. AI predicts: "valve 60% → power 245 kW"
  2. Actual result: power = 238 kW
  3. ConfidenceMonitor.update(245, 238) → confidence = 0.97
     → Good, keep AI active

After bad model:
  1. AI predicts: "valve 55% → power 300 kW"
  2. Actual result: power = 152 kW
  3. ConfidenceMonitor.update(300, 152) → confidence = 0.03
     → Below threshold → AI auto-disabled
     → Orchestrator switches to FALLBACK
     → Frontend shows: "AI paused: low confidence ⚠️"
```

### Checklist:
- [ ] Confidence computed per tick
- [ ] Rolling average over 30 samples
- [ ] Auto-disable below threshold
- [ ] Report available for frontend
- [ ] Recovery mechanism exists

---

## 6. PERSON 3 — PHASE 5: Logging & Monitoring

**Time estimate:** 45 minutes  
**Goal:** Structured logs for every control decision.

### Log Format

```
[2026-02-20 22:15:01] [INFO]  [ORCHESTRATOR] Tick #142 | T=512°C P=5.4bar V=50→55% W=234kW | mode=ACTIVE
[2026-02-20 22:15:01] [INFO]  [AI-DECISION]  MPC optimal: valve=58% predicted=248kW confidence=0.87
[2026-02-20 22:15:01] [INFO]  [SAFETY]       Check passed: pressure_headroom=2.6bar
[2026-02-20 22:15:01] [INFO]  [CONTROL]      Sent valve=55% (clamped from 58%, anti-oscillation)
[2026-02-20 22:15:32] [WARN]  [SAFETY]       Pressure 7.6bar > 7.5 → blocking valve increase
[2026-02-20 22:15:45] [ERROR] [AI-TIMEOUT]   AI decision took 4.2s > 3.0s limit → using fallback
[2026-02-20 22:16:01] [WARN]  [CONFIDENCE]   Rolling confidence 0.28 < 0.30 → AI auto-disabled
```

### Logger Implementation

```python
import logging
import json
from datetime import datetime

class StructuredLogger:
    """JSON-structured logging for production monitoring."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)-15s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def tick(self, tick_num, metrics, decision, safety):
        self.logger.info(
            "Tick #%d | T=%.0f°C P=%.1fbar V=%.0f→%.0f%% W=%.0fkW | "
            "mode=%s safety=%s conf=%.2f",
            tick_num,
            metrics["temperature"], metrics["pressure"],
            metrics["valve_position"], decision.get("valve", 0),
            metrics["power_output"],
            decision.get("mode", "?"),
            safety.get("level", "?"),
            decision.get("confidence", 0),
        )
    
    def safety_event(self, level, message):
        if level == "CRITICAL":
            self.logger.critical("🚨 SAFETY: %s", message)
        elif level == "WARNING":
            self.logger.warning("⚠️  SAFETY: %s", message)
    
    def ai_event(self, event_type, message):
        self.logger.info("[%s] %s", event_type, message)
```

### Checklist:
- [ ] Every tick logged with full state
- [ ] Safety events logged at WARN/CRITICAL
- [ ] AI mode transitions logged
- [ ] Structured format (parseable)
- [ ] No sensitive data in logs

---

## 7. PERSON 3 — PHASE 6: Environment Configuration

**Time estimate:** 30 minutes  
**Goal:** Zero hardcoded values — everything from .env.

### .env File

```env
# ─── Simulation Backend ───
SIMULATION_API_URL=http://localhost:8000
SIMULATION_METRICS_PATH=/metrics
SIMULATION_CONTROL_PATH=/control

# ─── AI Model ───
AI_MODEL_PATH=ai/models/pinn_model.pt
AI_MODE_DEFAULT=false
AI_CONFIDENCE_THRESHOLD=0.3

# ─── Safety ───
PRESSURE_LIMIT=8.0
PRESSURE_WARNING=7.5
PRESSURE_CRITICAL=7.8
TEMPERATURE_CRITICAL=590
SAFE_VALVE_POSITION=50.0

# ─── Orchestrator ───
ORCHESTRATOR_HOST=0.0.0.0
ORCHESTRATOR_PORT=8001
CONTROL_INTERVAL=1.0
MAX_HISTORY_LENGTH=300
MAX_RECOVERY_ATTEMPTS=3

# ─── Logging ───
LOG_LEVEL=INFO
LOG_FORMAT=structured

# ─── Frontend ───
FRONTEND_URL=http://localhost:3000
```

### config.py — Environment Loader

```python
import os
from dotenv import load_dotenv

load_dotenv()

# ── Simulation ──
SIM_API_URL = os.getenv("SIMULATION_API_URL", "http://localhost:8000")
SIM_METRICS_URL = f"{SIM_API_URL}{os.getenv('SIMULATION_METRICS_PATH', '/metrics')}"
SIM_CONTROL_URL = f"{SIM_API_URL}{os.getenv('SIMULATION_CONTROL_PATH', '/control')}"

# ── AI ──
AI_MODEL_PATH = os.getenv("AI_MODEL_PATH", "ai/models/pinn_model.pt")
AI_MODE_DEFAULT = os.getenv("AI_MODE_DEFAULT", "false").lower() == "true"
AI_CONFIDENCE_THRESHOLD = float(os.getenv("AI_CONFIDENCE_THRESHOLD", "0.3"))

# ── Safety ──
PRESSURE_LIMIT = float(os.getenv("PRESSURE_LIMIT", "8.0"))
PRESSURE_WARNING = float(os.getenv("PRESSURE_WARNING", "7.5"))
PRESSURE_CRITICAL = float(os.getenv("PRESSURE_CRITICAL", "7.8"))
TEMPERATURE_CRITICAL = float(os.getenv("TEMPERATURE_CRITICAL", "590"))
SAFE_VALVE_POSITION = float(os.getenv("SAFE_VALVE_POSITION", "50.0"))

# ── Orchestrator ──
ORCHESTRATOR_HOST = os.getenv("ORCHESTRATOR_HOST", "0.0.0.0")
ORCHESTRATOR_PORT = int(os.getenv("ORCHESTRATOR_PORT", "8001"))
CONTROL_INTERVAL = float(os.getenv("CONTROL_INTERVAL", "1.0"))
MAX_HISTORY = int(os.getenv("MAX_HISTORY_LENGTH", "300"))
MAX_RECOVERY = int(os.getenv("MAX_RECOVERY_ATTEMPTS", "3"))

# ── Logging ──
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
```

### Checklist:
- [ ] All values from .env
- [ ] Sensible defaults for every variable
- [ ] .env.example committed (not .env itself)
- [ ] No secrets in code
- [ ] Config validates on startup

---

## 8. PERSON 3 — PHASE 7: Deployment (Docker)

**Time estimate:** 1.5 hours  
**Goal:** Docker Compose with 4 services.

### Final Directory Structure

```
d:\BankokHack\
├── backend/
│   ├── Dockerfile
│   ├── api.py
│   ├── ...
│
├── ai/
│   ├── Dockerfile
│   ├── control_loop.py
│   ├── ...
│
├── integrator/
│   ├── Dockerfile
│   ├── orchestrator.py
│   ├── ...
│
├── frontend/
│   ├── Dockerfile
│   ├── src/
│   ├── ...
│
├── docker-compose.yml
├── .env
└── README.md
```

### docker-compose.yml

```yaml
version: "3.9"

services:
  # ─── Person 1: Simulation Engine ───
  simulation:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - API_HOST=0.0.0.0
      - API_PORT=8000
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 5s
      retries: 3

  # ─── Person 2: AI Controller ───
  ai:
    build: ./ai
    depends_on:
      simulation:
        condition: service_healthy
    environment:
      - BACKEND_URL=http://simulation:8000
    volumes:
      - ai-models:/app/models
      - ai-data:/app/data

  # ─── Person 3: Orchestrator ───
  orchestrator:
    build: ./integrator
    ports:
      - "8001:8001"
    depends_on:
      simulation:
        condition: service_healthy
    environment:
      - SIMULATION_API_URL=http://simulation:8000
      - AI_MODEL_PATH=/app/models/pinn_model.pt
      - ORCHESTRATOR_PORT=8001
    volumes:
      - ai-models:/app/models:ro

  # ─── Person 4: Frontend ───
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    depends_on:
      - orchestrator
    environment:
      - REACT_APP_ORCHESTRATOR_URL=http://localhost:8001
      - REACT_APP_SIMULATION_URL=http://localhost:8000

volumes:
  ai-models:
  ai-data:
```

### Backend Dockerfile

```dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Orchestrator Dockerfile

```dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8001
CMD ["python", "-m", "uvicorn", "orchestrator:app", "--host", "0.0.0.0", "--port", "8001"]
```

### Frontend Dockerfile

```dockerfile
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json .
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 3000
```

### Checklist:
- [ ] docker-compose up builds all 4 services
- [ ] Healthchecks in place
- [ ] Service discovery via Docker DNS
- [ ] Environment variables passed
- [ ] Volumes for model persistence
- [ ] Frontend served via nginx

---

## 9. PERSON 4 — PHASE 1: Core Dashboard UI

**Directory:** `frontend/`  
**Time estimate:** 3 hours  
**Goal:** Professional React dashboard with KPI cards, live charts, and AI toggle.

### Tech Stack

| Tool | Purpose |
|------|---------|
| React 18 + Vite | UI framework + bundler |
| Tailwind CSS | Utility-first styling |
| Recharts | Live charts |
| Axios | API calls |
| Framer Motion | Animations |
| React Three Fiber | 3D visualization |
| @react-three/drei | 3D helpers |

### File Structure

```
frontend/
├── src/
│   ├── App.jsx
│   ├── main.jsx
│   ├── index.css                    # Tailwind imports
│   ├── components/
│   │   ├── KPICard.jsx              # Animated metric card
│   │   ├── LiveChart.jsx            # Real-time line chart
│   │   ├── AIToggle.jsx             # On/Off switch
│   │   ├── SafetyIndicator.jsx      # Safety level badge
│   │   ├── ComparisonPanel.jsx      # Before vs After
│   │   ├── BusinessMetrics.jsx      # Energy/CO₂/savings
│   │   └── StatusBar.jsx            # Top status bar
│   ├── three/
│   │   ├── FactoryScene.jsx         # Main 3D scene
│   │   ├── Furnace.jsx              # Glowing furnace
│   │   ├── Pipe.jsx                 # Animated pipes
│   │   ├── Turbine.jsx              # Spinning turbine
│   │   └── SteamParticles.jsx       # Particle effects
│   ├── hooks/
│   │   ├── useMetrics.js            # Poll /api/state
│   │   └── useHistory.js            # Poll /api/history
│   ├── services/
│   │   └── api.js                   # Axios client config
│   └── constants/
│       └── theme.js                 # Colors, sizes
├── public/
├── package.json
├── vite.config.js
├── tailwind.config.js
├── postcss.config.js
└── .env
```

### Section 1 — KPI Panel

```jsx
// components/KPICard.jsx
import { motion, AnimatePresence } from "framer-motion";

export default function KPICard({ label, value, unit, icon, color, alert }) {
  return (
    <motion.div
      className={`
        relative overflow-hidden rounded-2xl p-6
        bg-gradient-to-br from-slate-800/80 to-slate-900/80
        border border-slate-700/50 backdrop-blur-xl
        shadow-lg shadow-${color}-500/10
      `}
      whileHover={{ scale: 1.02, y: -2 }}
      transition={{ type: "spring", stiffness: 300 }}
    >
      {/* Glow effect */}
      <div className={`absolute -top-10 -right-10 w-32 h-32
                        rounded-full bg-${color}-500/10 blur-3xl`} />
      
      <div className="flex items-center justify-between mb-3">
        <span className="text-slate-400 text-sm font-medium">{label}</span>
        <span className="text-2xl">{icon}</span>
      </div>
      
      <motion.div
        key={value}
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-baseline gap-2"
      >
        <span className={`text-4xl font-bold text-${color}-400`}>
          {typeof value === "number" ? value.toFixed(1) : value}
        </span>
        <span className="text-slate-500 text-lg">{unit}</span>
      </motion.div>
      
      {alert && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="mt-2 text-xs text-orange-400"
        >
          ⚠️ {alert}
        </motion.div>
      )}
    </motion.div>
  );
}

// Usage in App.jsx:
<div className="grid grid-cols-4 gap-4">
  <KPICard label="Power Output" value={metrics.power_output}
           unit="kW" icon="⚡" color="blue" />
  <KPICard label="Temperature" value={metrics.temperature}
           unit="°C" icon="🌡️" color="orange" />
  <KPICard label="Pressure" value={metrics.pressure}
           unit="bar" icon="💨" color="cyan"
           alert={metrics.pressure > 7.5 ? "Near limit!" : null} />
  <KPICard label="Valve Position" value={metrics.valve_position}
           unit="%" icon="🔧" color="emerald" />
</div>
```

### Section 2 — Live Charts

```jsx
// components/LiveChart.jsx
import { LineChart, Line, XAxis, YAxis, Tooltip,
         ResponsiveContainer, CartesianGrid } from "recharts";

export default function LiveChart({ data, dataKey, color, label, unit }) {
  return (
    <div className="rounded-2xl bg-slate-800/60 border border-slate-700/50
                    backdrop-blur-xl p-4">
      <h3 className="text-sm text-slate-400 mb-2">{label}</h3>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey="tick" stroke="#64748b" fontSize={10} />
          <YAxis stroke="#64748b" fontSize={10} />
          <Tooltip
            contentStyle={{
              background: "#1e293b",
              border: "1px solid #475569",
              borderRadius: "8px",
            }}
            formatter={(val) => [`${val.toFixed(1)} ${unit}`, label]}
          />
          <Line
            type="monotone"
            dataKey={dataKey}
            stroke={color}
            strokeWidth={2}
            dot={false}
            animationDuration={300}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
```

### Section 3 — AI Toggle

```jsx
// components/AIToggle.jsx
import { motion } from "framer-motion";
import axios from "axios";

export default function AIToggle({ enabled, onToggle }) {
  const handleToggle = async () => {
    const newState = !enabled;
    await axios.post("/api/ai/toggle", { enable: newState });
    onToggle(newState);
  };
  
  return (
    <div className="flex items-center gap-4 p-4 rounded-2xl
                    bg-slate-800/60 border border-slate-700/50">
      <span className="text-slate-300 font-medium">
        AI Optimization
      </span>
      
      <motion.button
        onClick={handleToggle}
        className={`
          relative w-16 h-8 rounded-full transition-colors duration-300
          ${enabled ? "bg-blue-500" : "bg-slate-600"}
        `}
      >
        <motion.div
          className="w-6 h-6 rounded-full bg-white shadow-lg"
          animate={{ x: enabled ? 32 : 4 }}
          transition={{ type: "spring", stiffness: 500, damping: 30 }}
        />
        {/* Glow when active */}
        {enabled && (
          <motion.div
            className="absolute inset-0 rounded-full bg-blue-400/30"
            animate={{ opacity: [0.3, 0.6, 0.3] }}
            transition={{ repeat: Infinity, duration: 2 }}
          />
        )}
      </motion.button>
      
      <motion.span
        key={enabled}
        initial={{ opacity: 0, x: -5 }}
        animate={{ opacity: 1, x: 0 }}
        className={`text-sm font-semibold ${
          enabled ? "text-blue-400" : "text-slate-500"
        }`}
      >
        {enabled ? "ACTIVE" : "OFF"}
      </motion.span>
    </div>
  );
}
```

### Data Polling Hook

```jsx
// hooks/useMetrics.js
import { useState, useEffect, useRef } from "react";
import axios from "axios";

export function useMetrics(intervalMs = 1000) {
  const [state, setState] = useState(null);
  const [history, setHistory] = useState([]);
  
  useEffect(() => {
    const timer = setInterval(async () => {
      try {
        const { data } = await axios.get("/api/state");
        setState(data);
        
        setHistory(prev => {
          const next = [...prev, {
            tick: prev.length,
            temperature: data.metrics?.temperature,
            pressure: data.metrics?.pressure,
            power_output: data.metrics?.power_output,
            valve_position: data.metrics?.valve_position,
          }];
          return next.slice(-120); // keep last 2 minutes
        });
      } catch (err) {
        console.error("Metrics poll failed:", err);
      }
    }, intervalMs);
    
    return () => clearInterval(timer);
  }, [intervalMs]);
  
  return { state, history };
}
```

### Checklist:
- [ ] 4 KPI cards with animated values
- [ ] 3+ live charts updating every second
- [ ] AI toggle switch works
- [ ] Dark theme applied
- [ ] Responsive layout

---

## 10. PERSON 4 — PHASE 2: 3D Factory Visualization

**Time estimate:** 3–4 hours  
**Goal:** Interactive 3D industrial scene that reacts to live data.

### React Three Fiber Scene

```jsx
// three/FactoryScene.jsx
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Environment } from "@react-three/drei";
import Furnace from "./Furnace";
import Pipe from "./Pipe";
import Turbine from "./Turbine";
import SteamParticles from "./SteamParticles";

export default function FactoryScene({ metrics }) {
  const temp = metrics?.temperature || 500;
  const valve = metrics?.valve_position || 50;
  const power = metrics?.power_output || 200;
  const pressure = metrics?.pressure || 5;
  
  return (
    <div className="rounded-2xl overflow-hidden border border-slate-700/50
                    bg-slate-900/80 h-[400px]">
      <Canvas camera={{ position: [8, 6, 8], fov: 45 }}>
        <ambientLight intensity={0.3} />
        <pointLight position={[5, 8, 5]} intensity={0.8} />
        
        {/* Factory components */}
        <Furnace
          temperature={temp}
          glowIntensity={(temp - 400) / 200}  // 0-1 based on temp
        />
        <Pipe
          position={[2, 1, 0]}
          flowSpeed={valve / 50}
          pressureColor={pressure > 7.5}
        />
        <Turbine
          position={[5, 1, 0]}
          rotationSpeed={power / 100}  // faster with more power
        />
        <SteamParticles
          count={Math.floor(valve * 2)}
          speed={power / 200}
        />
        
        <OrbitControls
          enablePan={false}
          maxDistance={15}
          minDistance={5}
        />
        <Environment preset="warehouse" />
      </Canvas>
    </div>
  );
}
```

### Furnace Component (Glowing)

```jsx
// three/Furnace.jsx
import { useRef } from "react";
import { useFrame } from "@react-three/fiber";

export default function Furnace({ temperature, glowIntensity }) {
  const meshRef = useRef();
  const glowRef = useRef();
  
  // Pulse glow based on temperature
  useFrame(({ clock }) => {
    if (glowRef.current) {
      const pulse = Math.sin(clock.elapsedTime * 2) * 0.1 + 0.9;
      glowRef.current.intensity = glowIntensity * pulse * 3;
    }
  });
  
  // Color: orange (cool) → red-white (hot)
  const r = Math.min(1, 0.8 + glowIntensity * 0.2);
  const g = Math.max(0, 0.4 - glowIntensity * 0.3);
  const b = Math.max(0, 0.1 - glowIntensity * 0.1);
  
  return (
    <group position={[-2, 0, 0]}>
      {/* Furnace body */}
      <mesh ref={meshRef}>
        <boxGeometry args={[2, 3, 2]} />
        <meshStandardMaterial
          color={[r, g, b]}
          emissive={[r * 0.5, g * 0.2, 0]}
          emissiveIntensity={glowIntensity * 2}
          roughness={0.7}
          metalness={0.3}
        />
      </mesh>
      
      {/* Point light for glow effect */}
      <pointLight
        ref={glowRef}
        position={[0, 2, 0]}
        color={[1, 0.4, 0.1]}
        distance={5}
      />
    </group>
  );
}
```

### Turbine Component (Rotating)

```jsx
// three/Turbine.jsx
import { useRef } from "react";
import { useFrame } from "@react-three/fiber";

export default function Turbine({ position, rotationSpeed }) {
  const bladeRef = useRef();
  
  useFrame((_, delta) => {
    if (bladeRef.current) {
      bladeRef.current.rotation.z += delta * rotationSpeed * 2;
    }
  });
  
  return (
    <group position={position}>
      {/* Housing */}
      <mesh>
        <cylinderGeometry args={[1.2, 1.2, 0.5, 32]} />
        <meshStandardMaterial color="#374151" metalness={0.8} roughness={0.2} />
      </mesh>
      
      {/* Spinning blades */}
      <group ref={bladeRef}>
        {[0, 60, 120, 180, 240, 300].map((angle) => (
          <mesh key={angle} rotation={[0, 0, (angle * Math.PI) / 180]}>
            <boxGeometry args={[0.15, 1, 0.05]} />
            <meshStandardMaterial color="#60a5fa" metalness={0.6} />
          </mesh>
        ))}
      </group>
    </group>
  );
}
```

### Data-Driven Reactions

| Plant Event | 3D Reaction |
|-------------|-------------|
| Temperature ↑ | Furnace glow intensifies (orange → white) |
| Valve opens | Pipe flow animation speeds up |
| Power ↑ | Turbine rotation faster |
| Pressure > 7.5 | Pipes turn orange/red |
| AI activates | Blue glow pulse on turbine |
| Safety override | Red flash on entire scene |

### Checklist:
- [ ] Furnace renders with dynamic glow
- [ ] Pipes show flow animation
- [ ] Turbine rotates proportional to power
- [ ] All 3D reacts to live data
- [ ] Camera orbit controls work
- [ ] Runs at 60fps

---

## 11. PERSON 4 — PHASE 3: Professional UI Design

**Time estimate:** 1.5 hours  
**Goal:** Startup-level visual polish.

### Design System

```javascript
// constants/theme.js
export const THEME = {
  colors: {
    background: "#0a0e1a",       // Deep navy
    surface: "#111827",          // Card background
    surfaceHover: "#1e293b",
    border: "#1e293b",
    
    primary: "#3b82f6",          // Blue
    primaryGlow: "#3b82f640",
    
    success: "#10b981",          // Green
    warning: "#f59e0b",          // Orange
    danger: "#ef4444",           // Red
    
    text: "#f1f5f9",             // Light
    textMuted: "#94a3b8",        // Gray
    textDim: "#475569",
  },
  
  glassmorphism: `
    bg-gradient-to-br from-slate-800/60 to-slate-900/60
    backdrop-blur-xl
    border border-slate-700/40
    shadow-xl shadow-black/20
  `,
};
```

### CSS Theme (Tailwind)

```css
/* index.css */
@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  body {
    @apply bg-[#0a0e1a] text-slate-100 antialiased;
    font-family: 'Inter', 'SF Pro Display', system-ui, sans-serif;
  }
}

@layer components {
  .glass-card {
    @apply bg-gradient-to-br from-slate-800/60 to-slate-900/60
           backdrop-blur-xl border border-slate-700/40
           rounded-2xl shadow-xl shadow-black/20;
  }
  
  .glow-blue {
    box-shadow: 0 0 20px rgba(59, 130, 246, 0.3),
                0 0 60px rgba(59, 130, 246, 0.1);
  }
  
  .glow-orange {
    box-shadow: 0 0 20px rgba(245, 158, 11, 0.3),
                0 0 60px rgba(245, 158, 11, 0.1);
  }
}
```

### Layout Structure

```
┌──────────────────────────────────────────────────┐
│ StatusBar: AI Mode + Safety Level + Uptime       │
├──────────┬──────────┬──────────┬─────────────────┤
│ Power ⚡ │ Temp 🌡️  │ Press 💨 │ Valve 🔧       │
│ 248.3 kW │ 512.3 °C │ 5.41 bar │ 58.0 %         │
├──────────┴──────────┴──────────┴─────────────────┤
│                                                   │
│          3D Factory Visualization                 │
│          (Furnace → Pipes → Turbine)              │
│                                                   │
├──────────────────────┬────────────────────────────┤
│ Temperature Chart    │ Power Output Chart          │
│ ───────────────      │ ───────────────             │
├──────────────────────┼────────────────────────────┤
│ Pressure Chart       │ AI Toggle + Confidence      │
│ ───────────────      │ ☐ AI Optimization: ON       │
├──────────────────────┴────────────────────────────┤
│ Before vs After Comparison                        │
│ Baseline: 198 kW → AI: 252 kW (+27.3%)           │
├───────────────────────────────────────────────────┤
│ Business Impact                                   │
│ Energy Saved: 45kWh | CO₂: -18kg | ₹1.2L/month   │
└───────────────────────────────────────────────────┘
```

### Checklist:
- [ ] Dark navy background
- [ ] Glassmorphism cards
- [ ] Neon blue + orange accents
- [ ] Smooth animations everywhere
- [ ] Consistent typography
- [ ] Mobile-responsive grid

---

## 12. PERSON 4 — PHASE 4: Before vs After Comparison

**Time estimate:** 1 hour  
**Goal:** Split-screen showing AI improvement.

### Component

```jsx
// components/ComparisonPanel.jsx
import { motion } from "framer-motion";

export default function ComparisonPanel({ comparison }) {
  const { baseline_avg_power, ai_avg_power, improvement_pct } = comparison;
  
  return (
    <div className="glass-card p-6">
      <h2 className="text-slate-400 text-sm mb-4 uppercase tracking-wider">
        AI Impact Analysis
      </h2>
      
      <div className="grid grid-cols-3 gap-6">
        {/* Before */}
        <div className="text-center p-4 rounded-xl bg-slate-800/40">
          <p className="text-slate-500 text-xs mb-1">BASELINE</p>
          <motion.p
            className="text-3xl font-bold text-slate-300"
            key={baseline_avg_power}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
          >
            {baseline_avg_power?.toFixed(1)}
          </motion.p>
          <p className="text-slate-500 text-sm">kW</p>
        </div>
        
        {/* Arrow + Improvement */}
        <div className="flex flex-col items-center justify-center">
          <motion.div
            className="text-4xl"
            animate={{ x: [0, 5, 0] }}
            transition={{ repeat: Infinity, duration: 1.5 }}
          >
            →
          </motion.div>
          <motion.p
            className={`text-2xl font-bold mt-2 ${
              improvement_pct > 0 ? "text-emerald-400" : "text-red-400"
            }`}
            key={improvement_pct}
            initial={{ scale: 0.8 }}
            animate={{ scale: 1 }}
          >
            {improvement_pct > 0 ? "+" : ""}{improvement_pct?.toFixed(1)}%
          </motion.p>
        </div>
        
        {/* After */}
        <div className="text-center p-4 rounded-xl bg-blue-900/20
                        border border-blue-500/20 glow-blue">
          <p className="text-blue-400 text-xs mb-1">AI OPTIMIZED</p>
          <motion.p
            className="text-3xl font-bold text-blue-300"
            key={ai_avg_power}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
          >
            {ai_avg_power?.toFixed(1)}
          </motion.p>
          <p className="text-blue-400 text-sm">kW</p>
        </div>
      </div>
    </div>
  );
}
```

### Checklist:
- [ ] Baseline vs AI side by side
- [ ] Improvement percentage animated
- [ ] Green if positive, red if negative
- [ ] Updates in real-time from /api/comparison
- [ ] Visual emphasis on AI result

---

## 13. PERSON 4 — PHASE 5: Business Metrics Section

**Time estimate:** 45 minutes  
**Goal:** Investor-ready impact numbers.

### Calculations

```javascript
// Derive business metrics from power improvement
function getBusinessMetrics(avgPower, baselinePower) {
  const extraKW = avgPower - baselinePower;
  const extraKWH_per_hour = extraKW;  // kW * 1hr = kWh
  
  return {
    energySaved: extraKWH_per_hour.toFixed(0),            // kWh/hour
    co2Reduced: (extraKWH_per_hour * 0.4).toFixed(0),     // kg/hour (grid factor)
    monthlySavings: Math.round(extraKWH_per_hour * 8 * 30 * 720 / 100), // ₹/month
    annualSavings: Math.round(extraKWH_per_hour * 8 * 365 * 720 / 100), // ₹/year
  };
}
```

### Component

```jsx
// components/BusinessMetrics.jsx
export default function BusinessMetrics({ metrics }) {
  const items = [
    { label: "Energy Recovered", value: metrics.energySaved, unit: "kWh/hr", icon: "⚡" },
    { label: "CO₂ Reduced", value: metrics.co2Reduced, unit: "kg/hr", icon: "🌱" },
    { label: "Monthly Savings", value: `₹${metrics.monthlySavings?.toLocaleString()}`, icon: "💰" },
    { label: "Annual Impact", value: `₹${metrics.annualSavings?.toLocaleString()}`, icon: "📈" },
  ];
  
  return (
    <div className="glass-card p-6">
      <h2 className="text-slate-400 text-sm mb-4 uppercase tracking-wider">
        Business Impact
      </h2>
      <div className="grid grid-cols-4 gap-4">
        {items.map(item => (
          <div key={item.label} className="text-center">
            <span className="text-2xl">{item.icon}</span>
            <p className="text-xl font-bold text-emerald-400 mt-2">
              {item.value}
            </p>
            <p className="text-xs text-slate-500">{item.unit || ""}</p>
            <p className="text-xs text-slate-400 mt-1">{item.label}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
```

### Checklist:
- [ ] Energy saved calculated from power delta
- [ ] CO₂ reduction displayed
- [ ] Monthly savings in ₹
- [ ] Animated counters
- [ ] Updates as AI runs

---

## 14. PERSON 4 — PHASE 6: Demo Animation Flow

**Time estimate:** 1 hour  
**Goal:** 5-minute walkthrough that judges will remember.

### Demo Script

```
0:00 — Page loads
  → Dark dashboard appears
  → KPI cards show baseline values
  → 3D factory renders with soft orange glow
  → Charts start streaming

0:30 — Show manual control
  → Dashboard shows "AI: OFF"
  → Baseline power: ~198 kW
  → Everything stable but unoptimized

1:00 — ACTIVATE AI
  → User clicks AI toggle
  → Toggle glows blue, pulse animation
  → Status bar: "AI Optimization: ACTIVE 🟢"
  → Console shows: "AI ACTIVATED — MPC mode"

1:30 — AI Takes Effect
  → Valve starts adjusting (visible in KPI card)
  → 3D turbine spins faster
  → Furnace glow changes
  → Power chart line begins climbing

2:30 — Peak Performance
  → Power output: ~252 kW
  → Improvement counter: "+27.3%"
  → Business metrics appear
  → 3D factory at full visual intensity

3:30 — Show Safety
  → (If possible) trigger high pressure scenario
  → Safety indicator turns orange/red
  → AI auto-adjusts valve down
  → "Safety override active" message
  → Pressure drops back to normal

4:30 — Before vs After
  → Comparison panel visible
  → Baseline: 198 kW → AI: 252 kW
  → Business impact: ₹1.2L/month savings
  → CO₂ reduction visible

5:00 — Close
  → Point to architecture diagram
  → "Physics informed neural network"
  → "Real-time model predictive control"
  → "Zero safety violations"
```

### Auto-Demo Mode (Optional)

```jsx
// For when you want the demo to run itself
async function runAutoDemoSequence() {
  // Phase 1: Show baseline (15s)
  await wait(15000);
  
  // Phase 2: Activate AI
  await axios.post("/api/ai/toggle", { enable: true });
  
  // Phase 3: Let AI optimize (45s)
  await wait(45000);
  
  // Phase 4: Show results
  // (comparison panel auto-populates)
}
```

### Checklist:
- [ ] Smooth transition from baseline to AI
- [ ] Visual impact on 3D scene
- [ ] Business metrics appear after AI runs
- [ ] Safety scenario demonstrable
- [ ] Under 5 minutes total

---

## 15. RISK & EDGE CASES

### Person 3 Risks

| Risk | Mitigation |
|------|-----------|
| AI service crashes | Fallback to safe valve (50%) |
| Backend unreachable | Retry 5x then stop sending commands |
| Confidence too low | Auto-disable AI, log warning |
| Orchestrator itself crashes | Docker restart policy: always |
| Port conflicts | Configurable via .env |
| Race conditions | Single-threaded asyncio loop |

### Person 4 Risks

| Risk | Mitigation |
|------|-----------|
| API CORS blocked | Backend already has CORS * |
| 3D too slow | Reduce polygon count, disable shadows |
| Charts lag with data | Limit history to 120 points |
| Mobile layout breaks | Responsive grid, hide 3D on mobile |
| WebGL not supported | Graceful fallback to 2D-only |
| API returns null | Null checks in every component |

---

## 16. DAY-BY-DAY SCHEDULE

### Person 3 — Integration

| Time | Task | Deliverable |
|------|------|-------------|
| Hour 1 | Create integrator/ structure + config.py | Folder ready |
| Hour 2 | Build orchestrator.py core loop | Polls + sends control |
| Hour 3 | Add AI mode toggle + state machine | /api/ai/toggle works |
| Hour 4 | Safety fallback system | Auto-fallback tested |
| Hour 5 | Confidence monitor | Auto-disable at low confidence |
| Hour 6 | Structured logging | All events logged |
| Hour 7 | .env configuration | No hardcoded values |
| Hour 8 | Docker compose | 4 services running |

### Person 4 — Frontend

| Time | Task | Deliverable |
|------|------|-------------|
| Hour 1 | React + Vite + Tailwind setup | Blank app running |
| Hour 2 | KPI cards + data hook | 4 cards updating |
| Hour 3 | Live charts (Recharts) | 3 charts streaming |
| Hour 4 | AI toggle + status bar | Toggle works |
| Hour 5 | 3D scene setup | Furnace + pipes + turbine |
| Hour 6 | 3D data binding | 3D reacts to metrics |
| Hour 7 | Before/After + Business metrics | Comparison panel |
| Hour 8 | Polish + demo flow | Demo-ready |

---

## 17. FINAL DELIVERY CHECKLIST

### Person 3 ✅

- [ ] `integrator/orchestrator.py` — polling + control loop
- [ ] `integrator/config.py` — env-driven config
- [ ] `integrator/safety.py` — enhanced safety fallback
- [ ] `integrator/confidence.py` — auto-disable on low confidence
- [ ] `integrator/logger.py` — structured logging
- [ ] `integrator/.env` — all config externalized
- [ ] AI toggle via HTTP API
- [ ] Safe fallback on AI failure
- [ ] Docker Compose with 4 services
- [ ] Health checks for all services
- [ ] Zero hardcoded values

### Person 4 ✅

- [ ] `frontend/` — React + Vite + Tailwind
- [ ] KPI cards with animated values
- [ ] Live charts (temperature, pressure, power)
- [ ] AI toggle switch with glow
- [ ] 3D factory scene (furnace, pipes, turbine)
- [ ] 3D reacts to live plant data
- [ ] Before vs After comparison panel
- [ ] Business metrics (energy, CO₂, savings)
- [ ] Dark industrial theme
- [ ] Smooth animations
- [ ] Demo-ready in under 5 minutes

---

### API Contract: Orchestrator ↔ Frontend

```
Frontend calls (Person 3's orchestrator on port 8001):

GET  /api/state       → { metrics, ai_decision, ai_mode, safety_level, confidence }
POST /api/ai/toggle   → { enable: true/false }
GET  /api/ai/status   → { mode, ai_enabled, confidence, error_count }
GET  /api/history     → [ { tick, temperature, pressure, power, valve, ... }, ... ]
GET  /api/comparison  → { baseline_avg_power, ai_avg_power, improvement_pct }
GET  /api/health      → { orchestrator: "ok", backend_connected, ai_loaded }
```

---

*Entropy Engine — System Integration + Frontend Execution Plan*  
*Version 1.0 | February 2026*
