# 🧠 ENTROPY ENGINE — Person 2 (AI) Full Execution Plan

> **Role:** AI Optimization & Control Layer Developer  
> **Project:** Physics-Informed Valve Controller for Waste Heat Recovery Plant  
> **Stack:** Python, PyTorch, httpx/aiohttp, NumPy, Pandas  
> **Timeline:** 3 Days  
> **Dependency:** Person 1's backend must be running at `http://localhost:8000`

---

## 📋 TABLE OF CONTENTS

1. [Role & Boundary](#1-role--boundary)
2. [Architecture Position](#2-architecture-position)
3. [File Structure](#3-file-structure)
4. [Phase 1 — API Contract & Environment Setup](#4-phase-1--api-contract--environment-setup)
5. [Phase 2 — Heuristic Baseline Controller](#5-phase-2--heuristic-baseline-controller)
6. [Phase 3 — Data Collection Pipeline](#6-phase-3--data-collection-pipeline)
7. [Phase 4 — Predictive Model (PyTorch)](#7-phase-4--predictive-model-pytorch)
8. [Phase 5 — Physics-Informed Neural Network (PINN)](#8-phase-5--physics-informed-neural-network-pinn)
9. [Phase 6 — Model Predictive Control (MPC)](#9-phase-6--model-predictive-control-mpc)
10. [Phase 7 — Real-Time Control Loop](#10-phase-7--real-time-control-loop)
11. [Phase 8 — Safety Enforcement Layer](#11-phase-8--safety-enforcement-layer)
12. [Phase 9 — Metrics & Confidence Reporting](#12-phase-9--metrics--confidence-reporting)
13. [Phase 10 — Integration Testing](#13-phase-10--integration-testing)
14. [Risk & Edge Cases](#14-risk--edge-cases)
15. [Day-by-Day Schedule](#15-day-by-day-schedule)

---

## 1. ROLE & BOUNDARY

### What You Build
- Physics-aware AI optimization layer
- Valve control recommender
- Safety-constrained decision engine
- Real-time backend integrator

### What You Do NOT Build
- ❌ React frontend
- ❌ Simulation engine (Person 1)
- ❌ REST API server (Person 1)
- ❌ Deployment / DevOps

### Your Single Interface
```
READ  → GET  http://localhost:8000/metrics
WRITE → POST http://localhost:8000/control
```

That's it. You consume and control.

---

## 2. ARCHITECTURE POSITION

```
┌─────────────────────────────────────────────┐
│         SIMULATION ENGINE (Person 1)         │
│   Furnace → Heat Exchanger → Steam → Turbine │
│                                              │
│   Runs at 1 Hz, updates physics state        │
└──────────────┬───────────────────────────────┘
               │
         GET /metrics
               │
               ▼
┌──────────────────────────────────────────────┐
│          AI CONTROL LAYER (You — Person 2)    │
│                                              │
│  ┌─────────────┐   ┌──────────────────────┐  │
│  │ Data        │   │ Predictive Model     │  │
│  │ Collector   │──▶│ (PyTorch + PINN)     │  │
│  └─────────────┘   └──────────┬───────────┘  │
│                               │              │
│                    ┌──────────▼───────────┐   │
│                    │ Safety Enforcer      │   │
│                    │ (hard constraints)   │   │
│                    └──────────┬───────────┘   │
│                               │              │
│                    ┌──────────▼───────────┐   │
│                    │ Control Decision     │   │
│                    │ (optimal valve)      │   │
│                    └─────────────────────┘   │
└──────────────────────┬───────────────────────┘
                       │
                 POST /control
                       │
                       ▼
              Simulation Updates State
                       │
                       ▼
              Frontend Displays (Person 3)
```

---

## 3. FILE STRUCTURE

```
d:\BankokHack\
└── ai/
    ├── config.py               # AI-specific constants & hyperparams
    ├── data_collector.py       # Polls /metrics, builds training CSV
    ├── model.py                # PyTorch model definition (PINN)
    ├── train.py                # Training loop
    ├── control_loop.py         # Real-time AI controller (main entry)
    ├── safety.py               # Hard safety constraints
    ├── utils.py                # Helpers: normalization, plotting
    ├── baseline_controller.py  # Heuristic controller (Phase 2)
    ├── requirements.txt        # AI-specific dependencies
    ├── data/
    │   └── training_data.csv   # Collected time-series (auto-generated)
    └── models/
        └── pinn_model.pt       # Saved trained model (auto-generated)
```

---

## 4. PHASE 1 — API Contract & Environment Setup

**Time estimate:** 30 minutes  
**Goal:** Confirm backend is reachable, understand data shapes.

### 4.1 — API Contract (from Person 1)

#### Read State
```
GET http://localhost:8000/metrics
```
**Response:**
```json
{
  "temperature": 510.23,
  "pressure": 6.31,
  "flow_rate": 3.12,
  "valve_position": 55.0,
  "power_output": 224.8,
  "timestamp": 1740000000.123
}
```

#### Send Control
```
POST http://localhost:8000/control
Content-Type: application/json

{"valve_position": 65.0}
```
**Response:**
```json
{
  "status": "accepted",
  "target_valve_position": 65.0
}
```

#### Constraints
| Variable | Range | Hard Limit |
|----------|-------|------------|
| `valve_position` | 0–100% | Enforced by API (422 if invalid) |
| `pressure` | 4–8 bar | **8 bar = SAFETY CAP** |
| `temperature` | 400–600°C | Clamped by simulation |
| `power_output` | 150–300 kW | **MAXIMIZE THIS** |

### 4.2 — Dependencies

**File:** `ai/requirements.txt`

```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
httpx>=0.25.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
```

### 4.3 — AI Config

**File:** `ai/config.py`

```python
# ─── Backend Connection ───
BACKEND_URL = "http://localhost:8000"
METRICS_ENDPOINT = f"{BACKEND_URL}/metrics"
CONTROL_ENDPOINT = f"{BACKEND_URL}/control"

# ─── Control Loop Timing ───
CONTROL_INTERVAL = 1.0          # seconds between AI decisions

# ─── Safety Thresholds ───
PRESSURE_SAFETY_LIMIT = 7.5     # bar — start reducing before 8.0
PRESSURE_HARD_LIMIT = 8.0       # bar — absolute max
PRESSURE_CRITICAL = 7.8         # bar — emergency reduce

# ─── Valve Constraints ───
MIN_VALVE = 0.0
MAX_VALVE = 100.0
MAX_VALVE_CHANGE_PER_TICK = 5.0 # max % change per decision (anti-oscillation)

# ─── Optimization Target ───
TARGET_POWER = 300.0            # kW — ideal maximum
TARGET_TEMPERATURE = 520.0      # °C — sweet spot for high power

# ─── Heuristic Controller Params ───
HEURISTIC_HIGH_TEMP = 560.0
HEURISTIC_LOW_TEMP = 440.0
HEURISTIC_VALVE_STEP = 3.0

# ─── Data Collection ───
DATA_FILE = "ai/data/training_data.csv"
MIN_TRAINING_SAMPLES = 1000
COLLECTION_INTERVAL = 1.0       # seconds

# ─── Model Hyperparams ───
INPUT_DIM = 5                   # [temp, pressure, flow, valve, power]
HIDDEN_DIM = 64
OUTPUT_DIM = 1                  # predicted power (or optimal valve)
LEARNING_RATE = 0.001
EPOCHS = 200
BATCH_SIZE = 64
SEQUENCE_LENGTH = 10            # lookback window for time-series

# ─── PINN Physics Loss Weight ───
PHYSICS_LOSS_WEIGHT = 0.1       # λ_physics
SAFETY_LOSS_WEIGHT = 0.5        # λ_safety
DATA_LOSS_WEIGHT = 1.0          # λ_data

# ─── Model Save Path ───
MODEL_SAVE_PATH = "ai/models/pinn_model.pt"
```

### Checklist:
- [ ] Backend is running and reachable
- [ ] `GET /metrics` returns valid JSON
- [ ] `POST /control` updates valve
- [ ] All deps installed (`pip install -r requirements.txt`)
- [ ] Config file has all hyperparams

---

## 5. PHASE 2 — Heuristic Baseline Controller

**File:** `ai/baseline_controller.py`  
**Time estimate:** 1 hour  
**Goal:** Get a working controller BEFORE any ML. This is your fallback.

### Logic

```python
def compute_valve_heuristic(metrics: dict, current_valve: float) -> float:
    """
    Rule-based controller — no ML needed.
    
    Strategy:
      - High temp → open valve (more flow → more power)
      - Pressure near limit → close valve (safety)
      - Low temp → close valve (conserve heat)
    """
    temp = metrics["temperature"]
    pressure = metrics["pressure"]
    valve = current_valve

    # ── Safety first: pressure approaching limit ──
    if pressure > 7.8:
        # EMERGENCY: reduce valve aggressively
        valve -= 8.0
    elif pressure > 7.5:
        # CAUTION: reduce valve gently
        valve -= 3.0

    # ── Optimization: temperature-based adjustment ──
    elif temp > 560:
        # Hot → open valve to extract more power
        valve += 3.0
    elif temp > 520:
        # Warm → slightly open
        valve += 1.0
    elif temp < 440:
        # Cold → close valve to let heat build up
        valve -= 3.0
    elif temp < 480:
        # Cool → slightly close
        valve -= 1.0

    # ── Anti-oscillation: limit change per tick ──
    delta = valve - current_valve
    delta = max(-5.0, min(5.0, delta))
    valve = current_valve + delta

    # ── Hard clamp ──
    return max(0.0, min(100.0, valve))
```

### Testing the Baseline

```python
async def run_baseline():
    """Run heuristic controller in a loop."""
    async with httpx.AsyncClient() as client:
        while True:
            resp = await client.get(METRICS_ENDPOINT)
            metrics = resp.json()
            
            new_valve = compute_valve_heuristic(metrics, metrics["valve_position"])
            
            await client.post(CONTROL_ENDPOINT, json={"valve_position": new_valve})
            
            print(f"T={metrics['temperature']:.1f}  P={metrics['pressure']:.2f}  "
                  f"V={metrics['valve_position']:.1f}→{new_valve:.1f}  "
                  f"W={metrics['power_output']:.1f}")
            
            await asyncio.sleep(CONTROL_INTERVAL)
```

### What to Verify:
- [ ] Power output trends upward
- [ ] Pressure never hits 8.0 bar
- [ ] Valve changes are smooth (no 0→100 oscillations)
- [ ] System reaches a stable-ish operating point
- [ ] Baseline power ≈ 200–250 kW

### Why This Matters:
- If ML model fails → fall back to this
- Gives you a **before vs after** comparison for the demo
- Judges see immediate value even without the neural network

---

## 6. PHASE 3 — Data Collection Pipeline

**File:** `ai/data_collector.py`  
**Time estimate:** 1 hour  
**Goal:** Collect 1000+ time-steps of plant state for training.

### What to Collect

Each row in CSV:
```
timestamp, temperature, pressure, flow_rate, valve_position, power_output
```

### Implementation

```python
async def collect_data(num_samples: int = 1500, interval: float = 1.0):
    """
    Poll /metrics and save to CSV.
    
    Strategy:
      - Run for ~25 minutes with random valve perturbations
      - This gives diverse training data covering the operating envelope
      - Vary valve between 20-80% to explore state space
    """
    data = []
    
    async with httpx.AsyncClient() as client:
        for i in range(num_samples):
            # Read current state
            resp = await client.get(METRICS_ENDPOINT)
            row = resp.json()
            data.append(row)
            
            # Every 50 steps, randomly perturb valve to explore state space
            if i % 50 == 0:
                random_valve = random.uniform(20, 80)
                await client.post(CONTROL_ENDPOINT, 
                                  json={"valve_position": random_valve})
            
            await asyncio.sleep(interval)
            
            if i % 100 == 0:
                print(f"Collected {i}/{num_samples} samples...")
    
    # Save to CSV
    df = pd.DataFrame(data)
    df.to_csv(DATA_FILE, index=False)
    print(f"✅ Saved {len(df)} samples to {DATA_FILE}")
    return df
```

### Data Exploration (before training)

After collection, verify:
```python
def explore_data(df):
    print(df.describe())
    print(f"\nCorrelation with power_output:")
    print(df.corr()["power_output"].sort_values(ascending=False))
    
    # Plot time series
    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    for ax, col in zip(axes, ["temperature", "pressure", "flow_rate", 
                                "valve_position", "power_output"]):
        ax.plot(df[col])
        ax.set_ylabel(col)
    plt.savefig("ai/data/exploration.png")
```

### Expected Correlations:
```
power_output ↔ pressure:        Strong positive
power_output ↔ temperature:     Moderate positive
power_output ↔ valve_position:  Complex (non-linear)
power_output ↔ flow_rate:       Moderate positive
```

### Checklist:
- [ ] CSV has 1000+ rows
- [ ] All 5 state variables present
- [ ] Timestamps are sequential
- [ ] Data covers different valve positions (20–80%)
- [ ] No NaN or missing values
- [ ] Data exploration plots look reasonable

---

## 7. PHASE 4 — Predictive Model (PyTorch)

**File:** `ai/model.py`  
**Time estimate:** 2–3 hours  
**Goal:** Learn plant dynamics — predict power given state.

### 7.1 — Model Architecture

```python
class PlantDynamicsModel(nn.Module):
    """
    Feedforward neural network that predicts power output
    given current plant state.
    
    Input:  [temperature, pressure, flow_rate, valve_position, power_output]
    Output: [predicted_power_next_step]
    """
    def __init__(self, input_dim=5, hidden_dim=64, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            
            nn.Linear(hidden_dim // 2, output_dim),
        )
    
    def forward(self, x):
        return self.net(x)
```

### 7.2 — Data Preparation

```python
class PlantDataset(Dataset):
    """
    Reads CSV, creates (state_t, power_t+1) pairs for supervised learning.
    """
    def __init__(self, csv_path, lookback=1):
        df = pd.read_csv(csv_path)
        
        # Features: current state
        features = ["temperature", "pressure", "flow_rate", 
                     "valve_position", "power_output"]
        
        self.X = df[features].values[:-lookback]   # state at time t
        self.y = df["power_output"].values[lookback:]  # power at time t+1
        
        # Normalize
        self.x_mean = self.X.mean(axis=0)
        self.x_std = self.X.std(axis=0) + 1e-8
        self.y_mean = self.y.mean()
        self.y_std = self.y.std() + 1e-8
        
        self.X = (self.X - self.x_mean) / self.x_std
        self.y = (self.y - self.y_mean) / self.y_std
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return (torch.FloatTensor(self.X[idx]), 
                torch.FloatTensor([self.y[idx]]))
```

### 7.3 — Training Loop

**File:** `ai/train.py`

```python
def train_model():
    dataset = PlantDataset(DATA_FILE)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    
    model = PlantDynamicsModel()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                pred = model(X_batch)
                val_loss += criterion(pred, y_batch).item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d} | Train: {avg_train:.6f} | Val: {avg_val:.6f}")
        
        # Save best model
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                'model_state_dict': model.state_dict(),
                'x_mean': dataset.x_mean,
                'x_std': dataset.x_std,
                'y_mean': dataset.y_mean,
                'y_std': dataset.y_std,
            }, MODEL_SAVE_PATH)
    
    print(f"✅ Best val loss: {best_val_loss:.6f}")
    print(f"✅ Model saved to {MODEL_SAVE_PATH}")
```

### Checklist:
- [ ] Model trains without errors
- [ ] Val loss decreases over epochs
- [ ] Saved model loads correctly
- [ ] Predicted vs actual power values are close
- [ ] No NaN losses

---

## 8. PHASE 5 — Physics-Informed Neural Network (PINN)

**Time estimate:** 2 hours  
**Goal:** Add physics constraints to the loss function — makes the model scientifically grounded.

### 8.1 — What Makes It "Physics-Informed"

Normal NN loss:
```
L = MSE(predicted_power, actual_power)
```

PINN loss:
```
L = λ_data  × MSE(predicted_power, actual_power)
  + λ_physics × physics_residual²
  + λ_safety  × safety_penalty
```

### 8.2 — Physics Residual

From Person 1's simulation equations:

```python
def physics_residual(state, predicted_power):
    """
    The physics say:
        effective_flow = flow_rate * (valve / 100)
        pressure = 0.004 * temperature * effective_flow
        power = 0.35 * pressure * effective_flow
    
    Any deviation from this is a physics violation.
    """
    temp = state[:, 0]       # temperature
    pressure = state[:, 1]   # pressure  
    flow = state[:, 2]       # flow_rate
    valve = state[:, 3]      # valve_position
    
    effective_flow = flow * (valve / 100.0)
    expected_pressure = 0.004 * temp * effective_flow
    expected_power = 0.35 * expected_pressure * effective_flow
    
    # Residual: how far is predicted from physics?
    residual = (predicted_power.squeeze() - expected_power) ** 2
    return residual.mean()
```

### 8.3 — Safety Penalty

```python
def safety_penalty(predicted_state):
    """
    Penalize predictions that lead to pressure > 7.5 bar.
    Uses soft ReLU to make it differentiable.
    """
    pressure = predicted_state[:, 1]
    violation = torch.relu(pressure - 7.5)  # 0 if safe, positive if exceeded
    return (violation ** 2).mean()
```

### 8.4 — Combined PINN Loss

```python
class PINNLoss(nn.Module):
    def __init__(self, lambda_data=1.0, lambda_physics=0.1, lambda_safety=0.5):
        super().__init__()
        self.lambda_data = lambda_data
        self.lambda_physics = lambda_physics
        self.lambda_safety = lambda_safety
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target, state_input):
        data_loss = self.mse(pred, target)
        phys_loss = physics_residual(state_input, pred)
        safe_loss = safety_penalty(state_input)
        
        total = (self.lambda_data * data_loss 
                + self.lambda_physics * phys_loss 
                + self.lambda_safety * safe_loss)
        
        return total, {
            "data_loss": data_loss.item(),
            "physics_loss": phys_loss.item(),
            "safety_loss": safe_loss.item(),
            "total_loss": total.item(),
        }
```

### 8.5 — Training with PINN Loss

```python
# In train.py, replace criterion:
criterion = PINNLoss(
    lambda_data=DATA_LOSS_WEIGHT,
    lambda_physics=PHYSICS_LOSS_WEIGHT,
    lambda_safety=SAFETY_LOSS_WEIGHT,
)

# Training step becomes:
loss, loss_dict = criterion(pred, y_batch, X_batch)
```

### Checklist:
- [ ] Physics residual computes without errors
- [ ] Safety penalty activates when pressure > 7.5
- [ ] Total loss has all 3 components
- [ ] Loss breakdown logged per epoch
- [ ] Model still converges (physics weight not too high)

---

## 9. PHASE 6 — Model Predictive Control (MPC)

**Time estimate:** 1.5 hours  
**Goal:** Use the trained model to find the optimal valve position.

### Strategy: Try Multiple Candidate Valves, Pick Best

```python
class ModelPredictiveController:
    """
    Model Predictive Control (MPC) — tries N candidate valve positions
    and picks the one that maximizes predicted power while staying safe.
    """
    
    def __init__(self, model, normalization_params):
        self.model = model
        self.model.eval()
        self.x_mean = normalization_params["x_mean"]
        self.x_std = normalization_params["x_std"]
        self.y_mean = normalization_params["y_mean"]
        self.y_std = normalization_params["y_std"]
    
    def predict_power(self, state: dict, candidate_valve: float) -> float:
        """Predict power output if we set valve to candidate_valve."""
        x = np.array([
            state["temperature"],
            state["pressure"],
            state["flow_rate"],
            candidate_valve,       # hypothetical valve
            state["power_output"],
        ])
        x_norm = (x - self.x_mean) / self.x_std
        x_tensor = torch.FloatTensor(x_norm).unsqueeze(0)
        
        with torch.no_grad():
            pred_norm = self.model(x_tensor).item()
        
        return pred_norm * self.y_std + self.y_mean  # denormalize
    
    def find_optimal_valve(self, state: dict, 
                            num_candidates: int = 50) -> dict:
        """
        Search over valve candidates to maximize predicted power.
        
        Returns:
            {
                "optimal_valve": float,
                "predicted_power": float,
                "confidence": float,
                "safe": bool,
                "candidates_evaluated": int
            }
        """
        current_valve = state["valve_position"]
        current_pressure = state["pressure"]
        
        # Generate candidates around current position (smooth transitions)
        candidates = np.linspace(
            max(0, current_valve - 15),
            min(100, current_valve + 15),
            num_candidates,
        )
        
        best_valve = current_valve
        best_power = -float("inf")
        predictions = []
        
        for valve in candidates:
            pred_power = self.predict_power(state, valve)
            predictions.append(pred_power)
            
            # Estimate pressure for this valve setting
            eff_flow = state["flow_rate"] * (valve / 100.0)
            est_pressure = 0.004 * state["temperature"] * eff_flow
            
            # Skip if predicted to violate pressure
            if est_pressure > 7.5:
                continue
            
            if pred_power > best_power:
                best_power = pred_power
                best_valve = valve
        
        # Confidence: how much better than current?
        current_power = state["power_output"]
        improvement = (best_power - current_power) / max(current_power, 1)
        
        return {
            "optimal_valve": round(best_valve, 2),
            "predicted_power": round(best_power, 2),
            "confidence": round(min(abs(improvement), 1.0), 3),
            "safe": current_pressure < 7.5,
            "candidates_evaluated": num_candidates,
        }
```

### How MPC Looks in the Demo

```
Current: valve=55% power=220kW pressure=6.3bar
Evaluating 50 valve candidates [40% → 70%]...
  valve=62% → predicted power=245kW ✅
  valve=65% → predicted power=252kW ✅  ← BEST
  valve=70% → predicted power=248kW ⚠️ pressure risk
Decision: Set valve to 65%
Improvement: +14.5%
```

### Checklist:
- [ ] Candidate search works
- [ ] Unsafe candidates filtered out
- [ ] Optimal valve within ±15% of current (smooth)
- [ ] Predicted power is realistic (150–300 kW range)
- [ ] Confidence metric computed

---

## 10. PHASE 7 — Real-Time Control Loop

**File:** `ai/control_loop.py`  
**Time estimate:** 1.5 hours  
**Goal:** Tie everything together — continuous AI control.

### Main Loop

```python
async def run_ai_control(mode="heuristic"):
    """
    Main AI control loop.
    
    Modes:
        "heuristic"  — rule-based baseline (no ML)
        "mpc"        — model predictive control (trained model)
        "hybrid"     — MPC with heuristic fallback
    """
    print(f"🧠 AI Control Loop starting in '{mode}' mode...")
    
    # Load model if needed
    controller = None
    if mode in ("mpc", "hybrid"):
        controller = load_trained_controller()
    
    history = []
    
    async with httpx.AsyncClient() as client:
        while True:
            try:
                # ── Step 1: Read plant state ──
                resp = await client.get(METRICS_ENDPOINT)
                metrics = resp.json()
                
                # ── Step 2: Decide valve position ──
                if mode == "heuristic":
                    new_valve = compute_valve_heuristic(
                        metrics, metrics["valve_position"]
                    )
                    decision = {"optimal_valve": new_valve}
                    
                elif mode == "mpc" and controller:
                    decision = controller.find_optimal_valve(metrics)
                    new_valve = decision["optimal_valve"]
                    
                elif mode == "hybrid" and controller:
                    decision = controller.find_optimal_valve(metrics)
                    new_valve = decision["optimal_valve"]
                    # Fall back to heuristic if confidence too low
                    if decision["confidence"] < 0.01:
                        new_valve = compute_valve_heuristic(
                            metrics, metrics["valve_position"]
                        )
                        decision["fallback"] = True
                
                # ── Step 3: Safety enforcement ──
                new_valve = enforce_safety(metrics, new_valve)
                
                # ── Step 4: Anti-oscillation (max 5% change per tick) ──
                current_valve = metrics["valve_position"]
                delta = new_valve - current_valve
                delta = max(-MAX_VALVE_CHANGE_PER_TICK, 
                           min(MAX_VALVE_CHANGE_PER_TICK, delta))
                new_valve = current_valve + delta
                new_valve = max(MIN_VALVE, min(MAX_VALVE, new_valve))
                
                # ── Step 5: Send control command ──
                await client.post(
                    CONTROL_ENDPOINT,
                    json={"valve_position": round(new_valve, 2)}
                )
                
                # ── Step 6: Log ──
                log_entry = {
                    **metrics,
                    "ai_valve_output": new_valve,
                    "mode": mode,
                }
                history.append(log_entry)
                
                print(
                    f"T={metrics['temperature']:6.1f}°C  "
                    f"P={metrics['pressure']:5.2f}bar  "
                    f"V={metrics['valve_position']:5.1f}→{new_valve:5.1f}%  "
                    f"W={metrics['power_output']:6.1f}kW"
                )
                
                await asyncio.sleep(CONTROL_INTERVAL)
                
            except httpx.ConnectError:
                print("⚠️  Backend not reachable. Retrying in 3s...")
                await asyncio.sleep(3)
            except Exception as e:
                print(f"❌ Error: {e}")
                await asyncio.sleep(CONTROL_INTERVAL)
```

### Entry Point

```python
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["heuristic", "mpc", "hybrid"],
                        default="heuristic")
    parser.add_argument("--collect", action="store_true", 
                        help="Collect training data first")
    parser.add_argument("--train", action="store_true",
                        help="Train model before running")
    args = parser.parse_args()
    
    if args.collect:
        asyncio.run(collect_data())
    if args.train:
        train_model()
    
    asyncio.run(run_ai_control(mode=args.mode))
```

### Usage Commands

```bash
# Step 1: Run baseline controller
python control_loop.py --mode heuristic

# Step 2: Collect training data
python control_loop.py --collect

# Step 3: Train the model
python control_loop.py --train

# Step 4: Run AI-powered controller
python control_loop.py --mode mpc

# Step 5: Run hybrid (ML + fallback)
python control_loop.py --mode hybrid
```

### Checklist:
- [ ] Heuristic mode works standalone
- [ ] MPC mode loads model and makes decisions
- [ ] Hybrid mode falls back gracefully
- [ ] Anti-oscillation limits valve jumps to ±5%
- [ ] Backend disconnect handled (retry)
- [ ] Clean console output

---

## 11. PHASE 8 — Safety Enforcement Layer

**File:** `ai/safety.py`  
**Time estimate:** 30 minutes  
**Goal:** Absolute last line of defense — even if model is wrong.

### Hard Safety Rules

```python
def enforce_safety(metrics: dict, proposed_valve: float) -> float:
    """
    Override AI decision if safety is at risk.
    This runs AFTER the model, BEFORE sending control.
    
    Rules (ordered by priority):
      1. Pressure > 7.8 → force reduce valve by 10%
      2. Pressure > 7.5 → cap valve increase
      3. Temperature > 590 → reduce valve
      4. Always clamp valve 0–100
    """
    pressure = metrics["pressure"]
    temperature = metrics["temperature"]
    current_valve = metrics["valve_position"]
    valve = proposed_valve
    
    # ── RULE 1: Emergency pressure ──
    if pressure > 7.8:
        valve = current_valve - 10.0
        print(f"🚨 SAFETY: Pressure {pressure:.1f}bar > 7.8 → forcing valve down")
    
    # ── RULE 2: High pressure — don't increase valve ──
    elif pressure > 7.5:
        if valve > current_valve:
            valve = current_valve  # block increase
            print(f"⚠️  SAFETY: Pressure {pressure:.1f}bar > 7.5 → blocking increase")
    
    # ── RULE 3: Temperature too high ──
    if temperature > 590:
        valve = min(valve, current_valve - 5.0)
        print(f"⚠️  SAFETY: Temp {temperature:.1f}°C > 590 → reducing valve")
    
    # ── RULE 4: Hard clamp ──
    valve = max(0.0, min(100.0, valve))
    
    return valve
```

### Safety Status Report (for frontend)

```python
def get_safety_status(metrics: dict) -> dict:
    """Return safety assessment for dashboard display."""
    pressure = metrics["pressure"]
    temp = metrics["temperature"]
    
    if pressure > 7.8 or temp > 590:
        level = "CRITICAL"
        color = "red"
    elif pressure > 7.5 or temp > 570:
        level = "WARNING"
        color = "orange"
    else:
        level = "NORMAL"
        color = "green"
    
    return {
        "safety_level": level,
        "color": color,
        "pressure_headroom": round(8.0 - pressure, 2),
        "temp_headroom": round(600 - temp, 1),
    }
```

### Checklist:
- [ ] Emergency pressure override works
- [ ] Valve increase blocked when pressure > 7.5
- [ ] Temperature override works
- [ ] Valve always stays 0–100
- [ ] Safety status returns correct level

---

## 12. PHASE 9 — Metrics & Confidence Reporting

**File:** `ai/utils.py`  
**Time estimate:** 45 minutes  
**Goal:** Provide transparent AI metrics for frontend display.

### What to Report

```python
class AIReport:
    """Metrics the AI exposes for frontend/dashboard."""
    
    def __init__(self):
        self.history = []
        self.power_before_ai = None  # baseline measurement
    
    def record(self, metrics, ai_decision):
        self.history.append({
            "timestamp": metrics["timestamp"],
            "actual_power": metrics["power_output"],
            "predicted_power": ai_decision.get("predicted_power"),
            "valve_sent": ai_decision["optimal_valve"],
            "safety_level": get_safety_status(metrics)["safety_level"],
        })
    
    def get_summary(self) -> dict:
        if len(self.history) < 10:
            return {"status": "collecting_baseline"}
        
        recent = self.history[-30:]  # last 30 readings
        avg_power = np.mean([r["actual_power"] for r in recent])
        
        # Prediction accuracy
        pred_errors = [
            abs(r["predicted_power"] - r["actual_power"]) 
            for r in recent 
            if r["predicted_power"] is not None
        ]
        avg_error = np.mean(pred_errors) if pred_errors else None
        
        # Improvement over baseline
        improvement = None
        if self.power_before_ai:
            improvement = ((avg_power - self.power_before_ai) 
                          / self.power_before_ai * 100)
        
        return {
            "avg_power_kw": round(avg_power, 1),
            "prediction_error_kw": round(avg_error, 2) if avg_error else None,
            "power_improvement_pct": round(improvement, 1) if improvement else None,
            "total_decisions": len(self.history),
            "safety_violations": sum(
                1 for r in self.history if r["safety_level"] == "CRITICAL"
            ),
        }
```

### Dashboard-Ready Output

```json
{
  "avg_power_kw": 258.3,
  "prediction_error_kw": 4.2,
  "power_improvement_pct": 14.5,
  "total_decisions": 342,
  "safety_violations": 0
}
```

### Checklist:
- [ ] Prediction vs actual tracked
- [ ] Power improvement calculated
- [ ] Safety violation count maintained
- [ ] Summary available as JSON
- [ ] No NaN in reports

---

## 13. PHASE 10 — Integration Testing

**Time estimate:** 1.5 hours  
**Goal:** End-to-end validation with Person 1's backend.

### Test Script

```python
# test_ai_integration.py

async def test_full_integration():
    """
    End-to-end test:
      1. Start with no AI → record baseline power
      2. Enable heuristic AI → observe improvement
      3. Enable MPC AI → observe further improvement
      4. Verify zero safety violations throughout
    """
    
    async with httpx.AsyncClient() as client:
        # ── Phase A: Baseline (no AI, 30 seconds) ──
        print("📊 Phase A: Recording baseline (30s)...")
        baseline_powers = []
        for _ in range(30):
            r = (await client.get(METRICS_ENDPOINT)).json()
            baseline_powers.append(r["power_output"])
            await asyncio.sleep(1)
        
        avg_baseline = np.mean(baseline_powers)
        print(f"   Baseline avg power: {avg_baseline:.1f} kW")
        
        # ── Phase B: Heuristic AI (60 seconds) ──
        print("🤖 Phase B: Heuristic controller (60s)...")
        heuristic_powers = []
        for _ in range(60):
            r = (await client.get(METRICS_ENDPOINT)).json()
            new_valve = compute_valve_heuristic(r, r["valve_position"])
            new_valve = enforce_safety(r, new_valve)
            await client.post(CONTROL_ENDPOINT, 
                            json={"valve_position": new_valve})
            heuristic_powers.append(r["power_output"])
            await asyncio.sleep(1)
        
        avg_heuristic = np.mean(heuristic_powers[-30:])  # last 30s
        print(f"   Heuristic avg power: {avg_heuristic:.1f} kW")
        improvement_h = (avg_heuristic - avg_baseline) / avg_baseline * 100
        print(f"   Improvement: {improvement_h:+.1f}%")
        
        # ── Phase C: Safety check ──
        all_pressures = [r["pressure"] for r in 
                        [json.loads(x) for x in baseline_powers + heuristic_powers]
                        if isinstance(r, dict)]
        max_pressure = max(r["pressure"] for _ in range(5)
                          if (r := (await client.get(METRICS_ENDPOINT)).json()))
        
        print(f"\n{'='*50}")
        print(f"RESULTS:")
        print(f"  Baseline power:    {avg_baseline:.1f} kW")
        print(f"  Heuristic power:   {avg_heuristic:.1f} kW")
        print(f"  Improvement:       {improvement_h:+.1f}%")
        print(f"  Max pressure seen: {max_pressure:.2f} bar")
        print(f"  Safety:            {'✅ PASS' if max_pressure <= 8.0 else '❌ FAIL'}")
        print(f"{'='*50}")
```

### Test Scenarios

| # | Test | Expected |
|---|------|----------|
| 1 | Baseline power (no AI) | ~200 kW |
| 2 | Heuristic AI running | +5–15% power |
| 3 | MPC AI running | +10–20% power |
| 4 | Pressure during AI control | Never exceeds 8.0 bar |
| 5 | Valve transitions | Smooth, no oscillation |
| 6 | Backend disconnects mid-run | AI retries, no crash |
| 7 | AI starts before backend | Waits and retries |
| 8 | Rapid state changes (heat spike) | Safety kicks in |

### Checklist:
- [ ] Baseline recorded before AI starts
- [ ] Power improvement measured quantitatively
- [ ] Pressure never exceeds 8.0 bar
- [ ] No crashes in 5-minute run
- [ ] Results printable for judges

---

## 14. RISK & EDGE CASES

| Risk | Mitigation |
|------|-----------|
| Model predicts nonsense | Clamp output to 0–100, add heuristic fallback |
| Model not trained yet | Default to heuristic mode |
| Backend is down | Retry loop with 3s backoff |
| Pressure spikes during control | Safety layer overrides AI |
| Oscillating valve | Max ±5% change per tick |
| Training data too uniform | Random valve perturbations during collection |
| Normalization breaks on new data | Save mean/std with model, clip extremes |
| Model overconfident | Track prediction error, alert if > 20kW |
| NaN in predictions | `torch.isnan()` check before sending control |
| CSV file missing | Check file exists before training, prompt user |

---

## 15. DAY-BY-DAY SCHEDULE

### DAY 1 (Focus: Baseline + Data)

| Time | Task | Deliverable |
|------|------|-------------|
| Hour 1 | Set up `ai/` folder, install PyTorch, httpx | Environment ready |
| Hour 2 | Write `config.py` with all AI constants | Config done |
| Hour 3 | Confirm backend API works (`GET /metrics`) | Connection verified |
| Hour 4 | Build `baseline_controller.py` | Heuristic working |
| Hour 5 | Test baseline — observe power improvement | Baseline validated |
| Hour 6 | Build `data_collector.py` | Data pipeline ready |
| Hour 7 | Collect 1000+ samples | `training_data.csv` saved |
| Hour 8 | Explore data, check correlations | Data quality confirmed |

**Day 1 Exit Criteria:**
- [ ] Heuristic controller running and improving power
- [ ] 1000+ rows in `training_data.csv`
- [ ] Pressure never exceeded 8.0 bar
- [ ] Backend API integration confirmed

---

### DAY 2 (Focus: Model + Control)

| Time | Task | Deliverable |
|------|------|-------------|
| Hour 1 | Build `model.py` — PlantDynamicsModel | Architecture defined |
| Hour 2 | Build `train.py` — training loop | Model trains on CSV |
| Hour 3 | Train model, evaluate val loss | Saved `pinn_model.pt` |
| Hour 4 | Add physics residual loss (PINN) | PINN loss working |
| Hour 5 | Retrain with PINN loss | Better model |
| Hour 6 | Build MPC controller | Candidate search works |
| Hour 7 | Build `control_loop.py` — full AI loop | MPC mode running |
| Hour 8 | Test MPC vs heuristic | Improvement measured |

**Day 2 Exit Criteria:**
- [ ] Model trains and converges
- [ ] PINN loss includes physics + safety penalties
- [ ] MPC finds better valve positions than heuristic
- [ ] Real-time control loop works

---

### DAY 3 (Focus: Safety + Polish)

| Time | Task | Deliverable |
|------|------|-------------|
| Hour 1 | Build `safety.py` — enforcement layer | Safety rules working |
| Hour 2 | Test safety during heat spikes | Emergency override works |
| Hour 3 | Build confidence & reporting (`utils.py`) | AI metrics available |
| Hour 4 | Integration test with Person 1 backend | Full flow verified |
| Hour 5 | Tune hyperparams for demo | Bigger improvement % |
| Hour 6 | Add hybrid mode (MPC + heuristic fallback) | Robust controller |
| Hour 7 | Record demo numbers (before/after) | Results ready |
| Hour 8 | Buffer / coordinate with team | Demo-ready |

**Day 3 Exit Criteria:**
- [ ] Zero safety violations in 30-minute run
- [ ] Power improvement ≥ 10% over baseline
- [ ] AI metrics printable for judges
- [ ] Hybrid mode handles all edge cases
- [ ] Clean, documented code

---

## 🚀 QUICK START COMMANDS

```bash
# 1. Setup
cd d:\BankokHack\ai
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 2. Make sure backend is running (Person 1)
# In another terminal:
cd d:\BankokHack\backend
uvicorn api:app --host 0.0.0.0 --port 8000

# 3. Run baseline controller
python control_loop.py --mode heuristic

# 4. Collect training data (~25 minutes)
python control_loop.py --collect

# 5. Train PINN model
python control_loop.py --train

# 6. Run AI-powered controller
python control_loop.py --mode mpc

# 7. Run hybrid (production mode)
python control_loop.py --mode hybrid
```

---

## ✅ FINAL DELIVERY CHECKLIST

- [ ] `ai/config.py` — all hyperparams, no magic numbers
- [ ] `ai/baseline_controller.py` — working heuristic
- [ ] `ai/data_collector.py` — collects and saves CSV
- [ ] `ai/model.py` — PyTorch model with PINN loss
- [ ] `ai/train.py` — trains and saves model
- [ ] `ai/control_loop.py` — real-time AI controller
- [ ] `ai/safety.py` — hard safety constraints
- [ ] `ai/utils.py` — confidence, metrics, reporting
- [ ] `ai/requirements.txt` — pinned dependencies
- [ ] Baseline controller improves power by ~5–10%
- [ ] ML controller improves power by ~10–20%
- [ ] Pressure never exceeds 8.0 bar
- [ ] Valve transitions are smooth
- [ ] Backend disconnect handled gracefully
- [ ] AI metrics available as JSON for frontend
- [ ] Demo-ready with before/after numbers

---

*Entropy Engine — AI Optimization Layer*  
*Version 1.0 | February 2026*
