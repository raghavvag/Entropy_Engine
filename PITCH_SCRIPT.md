# 🎤 ENTROPY ENGINE — Pitch Script

### Bangkok Hack 2026 | Technical Demo Walkthrough
#### Duration: ~5–6 minutes | 4 Speakers

---

## Speaker Assignments

| Speaker | Role | Sections | Time |
|:--------|:-----|:---------|:-----|
| **Person 1** (Backend) | Opens + Simulation | Opening hook, Architecture (backend layer), Physics demo | ~1:20 |
| **Person 2** (AI) | PINN + MPC Deep Dive | AI architecture, PINN loss function, MPC algorithm, Training | ~1:20 |
| **Person 3** (Integrator) | Orchestrator + Safety | State machine, Confidence monitor, Triple-layer safety, Fallback | ~1:20 |
| **Person 4** (Frontend) | Live Demo + Close | Dashboard walkthrough, AI activation, 3D scene, Comparison, Business impact, Closing | ~1:30 |

---

---

## 🎙 PERSON 1 — Backend & Opening

### [OPENING — 0:00]

> "Every year, industrial power plants worldwide waste **15 to 30 percent** of their potential energy output — not because of equipment failure, but because of **suboptimal manual control**. Operators rely on conservative, rule-based settings. They can't process dozens of interacting variables in real time. The result? Billions of dollars in lost efficiency, and millions of tonnes of unnecessary carbon emissions."

> "We asked ourselves: **what if AI could close that gap — safely, in real time, with physics it actually understands?**"

> "This is **Entropy Engine** — an AI-powered industrial plant optimizer built on a Physics-Informed Neural Network and Model Predictive Control. Let me walk you through how we built it."

### [ARCHITECTURE — BACKEND LAYER — 0:30]

> "At the foundation, I built a **physics simulation engine** — a FastAPI service running at 1 Hz that models a real industrial power plant: **furnace → heat exchanger → steam drum → turbine generator**."

> "Every tick computes five state variables using real thermodynamic equations:"
> - "**Temperature** — Newton's law of cooling: $\frac{dT}{dt} = -k(T - T_{amb})$, with $k = 0.02$"
> - "**Pressure** — ideal gas approximation: $P = c_1 \times T \times F_{eff}$"
> - "**Power output** — turbine conversion: $W = \eta \times P \times F_{eff}$, where $\eta = 0.35$"

> "To make it realistic, I added **sensor noise** at ±2%, random **furnace heat spikes**, **valve inertia** so the valve can't teleport to a new position, and **flow drift**. Every state transition is smoothed with a thermal inertia factor — 85% previous value, 15% calculated."

> "The simulation exposes 5 REST endpoints — `/metrics` for real-time state, `/control` to accept AI commands, `/status`, `/ai-mode`, and `/health`. All validated through **Pydantic** schemas."

> "This gives us a **faithful digital twin** of a real plant. Now I'll hand it over to **[Person 2]** to explain how we taught AI to understand this physics."

*[ ~1:20 elapsed ]*

---

---

## 🎙 PERSON 2 — AI Pipeline

### [PINN MODEL — 1:20]

> "Thanks. I built the AI brain — a **Physics-Informed Neural Network**, or PINN."

> "The architecture is a feedforward network: **5 input features** — temperature, pressure, flow rate, valve position, and current power — flowing through **three hidden layers** of 64, 64, and 32 neurons, with **BatchNorm** and **Dropout** at 10%, down to a **single output**: predicted optimal power."

> "But here's what makes a PINN different from a standard neural network. Our **loss function** has three terms:"

$$\mathcal{L} = \underbrace{1.0 \times \text{MSE}(\hat{W}, W)}_{\text{data fidelity}} + \underbrace{0.1 \times \frac{R^2_{\text{physics}}}{\sigma^2_W}}_{\text{physics residual}} + \underbrace{0.5 \times \text{mean}(\text{ReLU}(P - 7.5)^2)}_{\text{safety penalty}}$$

> "The **data term** matches observed power. The **physics residual** enforces that predictions must satisfy the thermodynamic equation $W = \eta \times P \times F^2_{eff}$ — the model literally **cannot** learn patterns that violate physics. And the **safety penalty** punishes any prediction where pressure exceeds 7.5 bar — safety is baked into the model's DNA."

> "We trained on just **1,500 samples** — about 25 minutes of plant data — for 200 epochs with early stopping. That's the power of physics-informed learning: you need far less data because the laws of thermodynamics fill in the gaps."

### [MPC CONTROLLER — 2:00]

> "On top of the PINN sits our **Model Predictive Controller**. Every second, it:"
> 1. "Generates **50 candidate** valve positions within ±15% of current"
> 2. "Feeds each through the PINN to **predict** the resulting power output"
> 3. "**Rejects** any candidate that would push estimated pressure above 7.5 bar"
> 4. "Selects the **optimal** valve position"
> 5. "Applies an **anti-oscillation clamp** — no more than ±5% change per tick"

> "If confidence drops below 1%, it automatically falls back to a **rule-based heuristic controller** — a set of if-then rules I wrote covering every dangerous scenario: high pressure, high temperature, low temperature. The AI degrades gracefully, never dangerously."

> "Now **[Person 3]** will explain how we integrated all of this into a production-ready system."

*[ ~2:40 elapsed ]*

---

---

## 🎙 PERSON 3 — Orchestrator & Safety

### [ORCHESTRATOR — 2:40]

> "Thanks. I built the **orchestrator** — the central nervous system that ties everything together."

> "It runs as a separate FastAPI service on port 8001 with a **state machine** controlling the AI lifecycle:"

```
IDLE  →  ACTIVE  →  FALLBACK
 ↑                      │
 └──────────────────────┘
        (auto-recovery)
```

> "In **IDLE** state, the system collects baseline metrics passively — no AI commands are sent. When you enable AI, it transitions to **ACTIVE** — the MPC makes live decisions every second. If anything goes wrong — model crash, timeout, or low confidence — it drops to **FALLBACK**: valve locks at a safe 50%, and the system attempts auto-recovery every 10 ticks, up to 3 attempts."

### [CONFIDENCE MONITOR — 3:10]

> "I built a **real-time confidence monitor** that tracks AI accuracy on a rolling 30-tick window:"

$$\text{confidence}_t = \max\left(0,\; 1 - \frac{|\hat{W}_t - W_t|}{\max(W_t, 1)}\right)$$

> "If the rolling average drops below **30%**, the AI is **automatically disabled** — no human intervention needed. It re-enables only after **10 consecutive good predictions**. This means the system is self-healing."

### [TRIPLE-LAYER SAFETY — 3:30]

> "Safety is our most important differentiator. We enforce it at **three independent layers** — no single failure can bypass protection:"

> "**Layer 1 — Training time**: The PINN's loss function penalizes unsafe pressure predictions. The model is structurally biased toward safe outputs."

> "**Layer 2 — Decision time**: The MPC rejects any candidate valve position that would produce pressure above 7.5 bar, **before** it's even considered."

> "**Layer 3 — Execution time**: Hard overrides. If pressure exceeds 7.8 bar, we force the valve down 10% regardless of what the AI says. If temperature exceeds 590°C, valve drops 5%. These rules are **non-negotiable** and apply after every AI decision."

> "The result: the AI has issued **zero safety violations** across thousands of ticks. Now **[Person 4]** will show you all of this running live."

*[ ~4:00 elapsed ]*

---

---

## 🎙 PERSON 4 — Live Demo & Closing

### [DEMO: HERO LANDING — 4:00]

*[ Open browser → http://localhost:3000 ]*

> "Let me show you Entropy Engine in action. This is our landing page — the 3D factory you see is a **live Three.js scene** with real-time lighting, volumetric steam particles from the chimney, and auto-rotating camera. Built in React 19 with React Three Fiber."

> "The green dot confirms all backend services are connected. Let's launch the dashboard."

*[ Click "Launch Dashboard →" ]*

### [DEMO: BASELINE — 4:15]

> "Here's the live control dashboard. Four KPI cards at the top streaming every second: **power, temp, pressure, valve**."

> "On the left — the **3D factory**. The furnace is glowing orange based on real temperature data. You can see **four workers** patrolling the factory floor — a furnace operator, pipe inspector, turbine tech, and supervisor — each following their own route."

> "Right now AI is **OFF**. Power is hovering around **150 to 180 kilowatts**. These are our baseline readings."

### [DEMO: ACTIVATE AI — 4:35]

*[ Click AI Toggle → ON ]*

> "I'm activating the AI now. Watch the status bar — it switches to **ACTIVE**."

*[ Wait 15 seconds ]*

> "Look at the changes. **Power output is climbing**. In the 3D scene — the turbine blades are now glowing **blue**, spinning faster. The pipe flow has sped up. The MPC is evaluating 50 valve positions per second and picking the best one."

> "The **safety indicator stays green**. The **confidence meter reads 92%**. The charts show a clear upward trend in power with intelligent micro-adjustments on the valve — no wild swings, thanks to the anti-oscillation clamp."

### [DEMO: RESULTS — 5:05]

*[ Scroll to comparison section ]*

> "Here's the payoff — our **AI Impact Analysis**."

> "Baseline average: **198 kilowatts**. With AI: **252 kilowatts**. That's a **27% improvement** — with zero safety violations."

> "In business terms: **54 extra kilowatt-hours per hour**, **22 kg less CO₂ per hour**, roughly **₹93,000 saved per month**, or over **₹11 lakh per year** — from a single plant unit. Scale this across a facility with multiple units, and you're looking at crores in annual savings."

### [CLOSING — 5:30]

> "Entropy Engine proves that AI can optimize complex industrial systems **right now**. Not by replacing human operators, but by **augmenting** their capabilities with physics-aware intelligence that decides, enforces safety, and self-corrects — in a closed loop, every second."

> "We're not just predicting. We're **controlling, protecting, and improving** — all in real time."

> "This is Entropy Engine. Thank you."

---

---

## 📝 Quick Reference — Key Numbers for All Speakers

| Metric | Value |
|:-------|:------|
| PINN Architecture | 5 → 64 → 64 → 32 → 1 with BatchNorm + Dropout |
| Training Samples | 1,500 (just 25 minutes of data) |
| MPC Candidates | 50 valve positions evaluated per tick |
| Tick Rate | 1 Hz (1 decision per second) |
| Safety Layers | 3 independent layers |
| Confidence Window | Rolling average over 30 ticks |
| Auto-disable Threshold | Confidence < 30% |
| Power Improvement | ~27% over baseline |
| Anti-oscillation | ±5% max valve change per tick |
| Tech Stack | Python 3.13, PyTorch 2.10, FastAPI, React 19, Three.js r183 |

---

## 🗣 Speaker Notes & Judge Q&A

**For all speakers — Pacing:**
- Don't rush. Let the demo breathe — give charts 15–20 seconds to update after toggling AI.
- Each speaker should have the next person's name ready for a smooth handoff.

**Anticipated questions & who answers:**

| Question | Who answers |
|:---------|:-----------|
| "How realistic is your simulation?" | **Person 1** — mention noise, heat spikes, inertia, flow drift |
| "Why PINN over standard NN?" | **Person 2** — physics residual means fewer data needed, no thermodynamic violations |
| "How is this different from PID control?" | **Person 2** — PID is reactive, single-variable. MPC is predictive, multi-variable, physics-constrained |
| "What happens if AI crashes?" | **Person 3** — FALLBACK state, safe valve 50%, auto-recovery, max 3 attempts |
| "Can this work on a real plant?" | **Person 3** — architecture is simulation-agnostic; swap `/metrics` + `/control` to SCADA/OPC-UA |
| "What's the 3D built with?" | **Person 4** — React Three Fiber + drei on Three.js r183, all meshes data-driven |
| "Show the fallback" | **Person 4** — toggle AI off → power drops → toggle on → recovery (proves it's real) |

**Fallback demo** (if judges ask): Toggle AI off → show power dropping → toggle back on → show recovery. Proves the improvement is causal, not coincidental.
