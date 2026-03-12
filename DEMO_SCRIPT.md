# 🎬 ENTROPY ENGINE — Video Demo Script

### Pre-recorded Demo for Bangkok Hack 2026
#### Total Duration: ~5–6 minutes | Screen Recording + Voiceover

---

## 📋 Pre-Demo Setup Checklist

Before recording, ensure all three services are running:

```bash
# Terminal 1 — Backend Simulation (port 8000)
cd backend && .\venv\Scripts\Activate.ps1
python -m uvicorn api:app --host 0.0.0.0 --port 8000

# Terminal 2 — Orchestrator + AI (port 8001)
cd integrator && .\venv\Scripts\Activate.ps1
python -m uvicorn orchestrator:app --host 0.0.0.0 --port 8001

# Terminal 3 — Frontend (port 3000)
cd frontend && npm run dev
```

Verify health:
```
GET http://localhost:8000/health   → {"status":"ok"}
GET http://localhost:8001/api/health → {"orchestrator":"ok","ai_loaded":false,...}
```

**Important:** Let the system run for ~30 seconds baseline **before** recording so the charts have data.

---

## 🎬 RECORDING SEQUENCE

---

### CLIP 1 — Hero Landing Page (10 seconds)

**Screen:** Browser at `http://localhost:3000` — full screen

**What's visible:**
- 3D factory scene in background with auto-rotating camera
- "ENTROPY ENGINE" title with gradient text
- Feature pills: PINN Model | MPC Controller | Real-Time Safety | Live 3D Viz
- Green dot at bottom: "Backend connected · Simulation running"
- "Launch Dashboard →" blue button

**🎙 Voiceover:**
> "This is Entropy Engine — an AI-powered industrial power plant optimizer. The 3D factory you see is a live Three.js scene built with React Three Fiber. The green dot confirms our backend services are connected and the simulation is running."

**Action:** Click "Launch Dashboard →"

---

### CLIP 2 — Baseline Dashboard (20 seconds)

**Screen:** Dashboard with AI **OFF**

**What's visible:**
- **Status Bar (top):** AI = IDLE | Safety = NORMAL | Confidence = 0% | Tick counter
- **4 KPI Cards:** Power ~170-180 kW | Temp ~475-530°C | Pressure ~5.5-6.5 bar | Valve = 50%
- **3D Factory (left):** Furnace glowing orange based on temperature, chimney steam particles, 4 workers patrolling routes
- **AI Toggle (right):** Shows "OFF — Manual baseline"
- **Safety Indicator:** Green — "ALL CLEAR", showing pressure headroom ~1.5 bar, temp headroom ~60°C
- **Charts (below):** Power Output, Temperature, Pressure, Valve Position — all flat/slightly varying

**🎙 Voiceover:**
> "Here's our live control dashboard. The four KPI cards stream real sensor data every second — power output, temperature, pressure, and valve position. On the left, the 3D factory scene shows the furnace glowing orange based on actual temperature data. You can see four workers patrolling the factory floor — a furnace operator, pipe inspector, turbine tech, and supervisor. Right now AI is OFF. Power output is hovering around 170 to 180 kilowatts. This is our baseline performance — what a manually-controlled plant looks like."

**Key highlight:** Point cursor slowly across each KPI card as you mention it.

---

### CLIP 3 — Activate AI & Watch Optimization (40 seconds)

**This is the MONEY SHOT — give it time to breathe.**

**Action:** Click the AI Toggle → **ON**

**🎙 Voiceover (immediately after click):**
> "I'm activating the AI now. Watch the status bar — it switches from IDLE to ACTIVE."

**Wait 5 seconds** (PINN model lazy-loads — you'll see status change)

**What changes on screen:**
- Status bar: **AI = ACTIVE** (blue)
- AI Toggle: Shows "ACTIVE — PINN + MPC actively controlling plant"
- Confidence meter starts filling up (→ ~90%)
- Valve Position starts moving (from 50% → climbing to 55-65%)

**🎙 Voiceover (after 5-10 seconds):**
> "The Physics-Informed Neural Network just loaded. The MPC controller is now evaluating 50 candidate valve positions every second and selecting the optimal one. Watch the valve position chart — you can see intelligent micro-adjustments, no wild swings thanks to the anti-oscillation clamp limiting changes to ±5% per tick."

**Wait 10 more seconds** — power starts climbing

**What changes:**
- Power Output KPI: climbing from ~180 → 200 → 220 → 240+ kW
- Power chart: clear upward trend with blue "AI Predicted" line tracking actual power
- Valve chart: green "AI Valve" line diverging from baseline, optimizing toward higher valve
- Turbine in 3D scene: glowing **blue**, spinning faster
- Pressure may approach 7.0-7.5 bar (still safe)

**🎙 Voiceover (15-20 seconds in):**
> "Look at the power output — it's climbing steadily. The MPC found that the plant was under-utilizing its capacity. In the 3D scene, the turbine blades are now glowing blue and spinning faster. The safety indicator stays green — the confidence meter reads over 90%. The PINN model's physics-informed predictions are matching reality with only 15-20 kilowatt average error."

---

### CLIP 4 — Safety & Alerts in Action (15 seconds)

**What's visible:**
- Safety Indicator: shows "ALL CLEAR" with pressure headroom and temp headroom numbers
- Safety Overrides counter: **0** — zero violations
- If temperature approaches 580°C: **orange "⚠️ Near limit!" alert** appears on Temperature KPI card
- If pressure approaches 7.5 bar: **orange "⚠️ High pressure!" alert** appears on Pressure KPI card
- Status bar Safety pill stays GREEN

**🎙 Voiceover:**
> "Safety is our most important feature. We enforce it at three independent layers. The PINN's loss function penalizes unsafe predictions during training. The MPC rejects any valve position that would push pressure above 7.5 bar before it's even considered. And hard runtime overrides force the valve down if pressure exceeds 7.8 bar or temperature exceeds 590°C. Zero safety violations — the AI optimizes aggressively but never dangerously."

**Key highlight:** Move cursor to the Safety Indicator showing "0 overrides" and the green "ALL CLEAR" status.

---

### CLIP 5 — Comparison & Business Impact (20 seconds)

**Action:** Scroll down to the "AI Impact Analysis" and "Business Impact" sections

**What's visible:**
- **AI Impact Analysis panel:**
  - BASELINE: ~177 kW avg (with sample count)
  - Arrow with animated pulse
  - **+27%** improvement (green)
  - AI OPTIMIZED: ~225 kW avg (blue glow)
- **Business Impact panel:**
  - ⚡ Energy Recovered: **~48 kWh/hr**
  - 🌱 CO₂ Reduced: **~19 kg/hr**
  - 💰 Monthly Savings: **~₹83,000**
  - 📈 Annual Impact: **~₹10+ lakh**

**🎙 Voiceover:**
> "Here's the payoff — our AI Impact Analysis. Baseline average: around 177 kilowatts. With AI optimization: 225 kilowatts. That's a 27% improvement with zero safety violations. In business terms: nearly 50 extra kilowatt-hours per hour, 19 kilograms less CO₂ per hour, roughly 83 thousand rupees saved per month, or over 10 lakh rupees per year — from a single plant unit. Scale this across a facility with multiple units and you're looking at crores in annual savings."

---

### CLIP 6 — Fallback Demo / AI Toggle Proof (20 seconds)

**This proves the improvement is causal, not coincidental.**

**Action:** Click AI Toggle → **OFF**

**🎙 Voiceover:**
> "To prove this is real and not coincidence, watch what happens when I turn the AI off."

**Wait 10 seconds**

**What changes:**
- Status: IDLE
- Power output starts **dropping** back toward baseline (~170 kW)
- Valve returns to 50%
- Charts show clear decline

**Action:** Click AI Toggle → **ON** again

**🎙 Voiceover:**
> "Power is dropping back to baseline. Now I'll re-enable the AI..."

**Wait 10 seconds**

**What changes:**
- Power climbs again
- Valve optimizes again

**🎙 Voiceover:**
> "And there it goes — power climbing right back up. The improvement is causal. The AI is genuinely optimizing this plant in real time."

---

### CLIP 7 — Closing Shot (10 seconds)

**Screen:** Scroll back to top showing full dashboard with AI running — all green, power high

**🎙 Voiceover:**
> "Entropy Engine proves that AI can optimize complex industrial systems right now. Not by replacing human operators, but by augmenting their capabilities with physics-aware intelligence that decides, enforces safety, and self-corrects — in a closed loop, every second. This is Entropy Engine."

---

---

## 📊 What Each Screen Section Shows

| Section | Data Source | Updates | What to Point Out |
|---------|-----------|---------|-------------------|
| **Status Bar** | `/api/state` | 1s | AI state (IDLE/ACTIVE), Safety level, Confidence %, Tick count |
| **KPI Cards** | `/api/state → metrics` | 1s | Power climbing, orange alerts on high temp/pressure |
| **3D Factory** | `metrics.temperature`, `aiActive` | Every frame | Furnace glow (orange), Turbine glow (blue when AI on), Steam particles, Worker animations |
| **AI Toggle** | `/api/ai/toggle` | On click | Text changes to "PINN + MPC actively controlling" |
| **Safety Indicator** | `/api/state → safety_level` | 1s | Green = safe, headroom values, override count = 0 |
| **Confidence Card** | `/api/state → confidence` | 1s | Bar filling to 90%+, "Samples: N", "Avg Error: X kW" |
| **Power Chart** | `/api/history` | 2s | Blue "Power" line + purple "AI Predicted" line — both climbing |
| **Temperature Chart** | `/api/history` | 2s | Orange area chart — stable ~475-540°C |
| **Pressure Chart** | `/api/history` | 2s | Cyan area chart — rises with optimization but stays < 7.5 |
| **Valve Chart** | `/api/history` | 2s | Blue "Current Valve" + Blue "AI Valve" — smooth ramp up |
| **AI Impact Analysis** | `/api/comparison` | 3s | Baseline vs AI avg power, % improvement |
| **Business Impact** | Computed from comparison | 3s | Energy saved, CO₂ reduced, ₹ savings |

---

## 🎯 Key Numbers to Mention

| Metric | Expected Value During Demo |
|--------|---------------------------|
| Baseline Power | ~170-180 kW |
| AI Optimized Power | ~220-250 kW |
| Improvement | ~25-30% |
| Confidence | ~88-95% |
| Safety Violations | 0 |
| Safety Overrides | 0 |
| Avg Prediction Error | ~15-25 kW |
| Valve Movement | 50% → 55-65% |
| Temperature | ~475-540°C (safe zone) |
| Pressure | ~5.5-7.0 bar (safe zone) |

---

## ⚠️ Recording Tips

1. **Resolution:** Record at 1920×1080 or higher. Browser zoomed to 100%.
2. **Wait for charts:** After toggling AI, wait **at least 15-20 seconds** before narrating the results. Charts need time to populate.
3. **Clean browser:** No bookmarks bar, no extensions visible. Full screen (F11).
4. **Smooth cursor:** Move cursor slowly when highlighting elements. Don't click randomly.
5. **PINN load delay:** The first time AI is enabled after orchestrator restart, there's a ~15-20 second delay while PyTorch loads the model. **Start the recording with AI already toggled once** (toggle ON, wait for it to load, toggle OFF, let baseline accumulate, then start recording).
6. **OBS Settings:** Use Display Capture or Window Capture of the browser. Record audio separately for cleaner editing.
7. **If things go wrong:** Toggle AI off → wait 10s → toggle on. The system self-heals.

---

## 🔄 Quick Reset Procedure

If you need to re-record a clean take:

```powershell
# 1. Kill orchestrator (Ctrl+C in its terminal)
# 2. Reset valve to baseline
Invoke-RestMethod -Uri "http://localhost:8000/control" -Method POST `
  -Body '{"valve_position": 50.0}' -ContentType "application/json"

# 3. Wait 10 seconds for plant to settle
Start-Sleep -Seconds 10

# 4. Restart orchestrator (fresh baseline)
cd integrator; .\venv\Scripts\Activate.ps1
python -m uvicorn orchestrator:app --host 0.0.0.0 --port 8001

# 5. Wait 20-30 seconds for baseline data
# 6. Start recording!
```
