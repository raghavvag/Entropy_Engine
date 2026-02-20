import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useMetrics, useHistory, useComparison } from "./hooks/useMetrics";
import { THEME } from "./constants/theme";

import StatusBar from "./components/StatusBar";
import KPICard from "./components/KPICard";
import LiveChart from "./components/LiveChart";
import AIToggle from "./components/AIToggle";
import SafetyIndicator from "./components/SafetyIndicator";
import ComparisonPanel from "./components/ComparisonPanel";
import BusinessMetrics from "./components/BusinessMetrics";
import FactoryScene from "./three/FactoryScene";

export default function App() {
  const { state, connected } = useMetrics(1000);
  const history = useHistory(2000, 120);
  const comparison = useComparison(3000);

  const [aiEnabled, setAiEnabled] = useState(false);
  const [showDashboard, setShowDashboard] = useState(false);

  // Sync AI state from backend
  const effectiveAI = state?.ai_mode ?? aiEnabled;

  const handleToggle = useCallback((val) => {
    setAiEnabled(val);
  }, []);

  const metrics = state?.metrics || {};
  const safety = state?.safety_level ?? "NORMAL";

  // If state is available but dashboard isn't shown yet, auto-transition
  // after a small delay (but user can click immediately)
  const safetyStat = state?.confidence || {};

  if (!showDashboard) {
    return <HeroLanding onEnter={() => setShowDashboard(true)} connected={connected} metrics={metrics} aiActive={effectiveAI} />;
  }

  return (
    <div className="min-h-screen bg-grid">
      <StatusBar state={state} connected={connected} />

      <main className="max-w-[1440px] mx-auto px-4 py-6 space-y-5">
        {/* ── Row 1: KPI Cards ── */}
        <motion.div
          className="grid grid-cols-2 md:grid-cols-4 gap-4"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <KPICard
            label="Power Output" value={metrics.power_output} unit="kW"
            icon="⚡" color="blue"
            sub={effectiveAI ? "AI optimized" : "Baseline"}
          />
          <KPICard
            label="Temperature" value={metrics.temperature} unit="°C"
            icon="🌡️" color="orange"
            alert={metrics.temperature > 580 ? "Near limit!" : null}
          />
          <KPICard
            label="Pressure" value={metrics.pressure} unit="bar"
            icon="💨" color="cyan"
            alert={metrics.pressure > 7.5 ? "High pressure!" : null}
          />
          <KPICard
            label="Valve Position" value={metrics.valve_position} unit="%"
            icon="🔧" color="emerald"
          />
        </motion.div>

        {/* ── Row 2: 3D Scene + AI Controls ── */}
        <motion.div
          className="grid grid-cols-1 lg:grid-cols-3 gap-4"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
        >
          <div className="lg:col-span-2">
            <FactoryScene metrics={metrics} aiActive={effectiveAI} />
          </div>
          <div className="space-y-4">
            <AIToggle enabled={effectiveAI} onToggle={handleToggle} />
            <SafetyIndicator
              level={safety}
              overrides={safetyStat?.stats?.total_overrides ?? 0}
              pressureHeadroom={8.0 - (metrics.pressure ?? 5)}
              tempHeadroom={590 - (metrics.temperature ?? 450)}
            />
            <ConfidenceCard confidence={state?.confidence} />
          </div>
        </motion.div>

        {/* ── Row 3: Live Charts ── */}
        <motion.div
          className="grid grid-cols-1 md:grid-cols-2 gap-4"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <LiveChart
            data={history}
            lines={[
              { key: "power_output", color: THEME.chart.power, name: "Power" },
              { key: "predicted_power", color: THEME.chart.predicted, name: "AI Predicted" },
            ]}
            label="Power Output"
            unit="kW"
            area
          />
          <LiveChart
            data={history}
            lines={[
              { key: "temperature", color: THEME.chart.temperature, name: "Temperature" },
            ]}
            label="Temperature"
            unit="°C"
            area
          />
          <LiveChart
            data={history}
            lines={[
              { key: "pressure", color: THEME.chart.pressure, name: "Pressure" },
            ]}
            label="Pressure"
            unit="bar"
            area
          />
          <LiveChart
            data={history}
            lines={[
              { key: "valve_position", color: THEME.chart.valve, name: "Current Valve" },
              { key: "ai_valve", color: "#3b82f6", name: "AI Valve" },
            ]}
            label="Valve Position"
            unit="%"
          />
        </motion.div>

        {/* ── Row 4: Comparison + Business ── */}
        <motion.div
          className="grid grid-cols-1 lg:grid-cols-2 gap-4"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
        >
          <ComparisonPanel comparison={comparison} />
          <BusinessMetrics comparison={comparison} />
        </motion.div>
      </main>

      {/* Footer */}
      <footer className="text-center py-6 text-[11px] text-slate-600 border-t border-slate-800/50">
        Entropy Engine — Physics-Informed Neural Network + Model Predictive Control
      </footer>
    </div>
  );
}


/* ── Confidence mini-card ── */
function ConfidenceCard({ confidence }) {
  if (!confidence) return null;
  const pct = ((confidence.confidence ?? 0) * 100).toFixed(0);
  const isGood = confidence.confidence > 0.5;

  return (
    <div className="glass-card p-4">
      <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-2 font-semibold">
        Model Confidence
      </p>
      <div className="flex items-center gap-3">
        <div className="flex-1">
          <div className="w-full h-2 rounded-full bg-slate-700/60 overflow-hidden">
            <motion.div
              className={`h-full rounded-full ${isGood ? "bg-emerald-400" : "bg-amber-400"}`}
              initial={{ width: 0 }}
              animate={{ width: `${pct}%` }}
              transition={{ duration: 0.5 }}
            />
          </div>
        </div>
        <span className={`text-sm font-bold font-mono-num ${isGood ? "text-emerald-400" : "text-amber-400"}`}>
          {pct}%
        </span>
      </div>
      <div className="flex justify-between mt-2 text-[10px] text-slate-500">
        <span>Samples: {confidence.samples ?? 0}</span>
        <span>Avg Error: {(confidence.avg_error_kw ?? 0).toFixed(1)} kW</span>
      </div>
    </div>
  );
}


/* ══════════════════════════════════════════════
   HERO LANDING PAGE
   ══════════════════════════════════════════════ */

function HeroLanding({ onEnter, connected, metrics, aiActive }) {
  return (
    <div className="min-h-screen relative overflow-hidden bg-navy-950">
      {/* Background 3D Scene */}
      <div className="absolute inset-0 opacity-40">
        <FactoryScene metrics={metrics} aiActive={aiActive} />
      </div>

      {/* Gradient overlays */}
      <div className="absolute inset-0 bg-gradient-to-b from-navy-950/80 via-transparent to-navy-950" />
      <div className="absolute inset-0 bg-gradient-to-r from-navy-950/60 via-transparent to-navy-950/60" />

      {/* Grid pattern */}
      <div className="absolute inset-0 bg-grid opacity-30" />

      {/* Content */}
      <div className="relative z-10 flex flex-col items-center justify-center min-h-screen px-4">
        {/* Logo */}
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          className="mb-8"
        >
          <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-blue-500 to-cyan-400 flex items-center justify-center shadow-2xl shadow-blue-500/30">
            <span className="text-4xl font-black text-white">E</span>
          </div>
        </motion.div>

        {/* Title */}
        <motion.h1
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="text-5xl md:text-7xl font-black text-center tracking-tight"
        >
          <span className="bg-gradient-to-r from-blue-400 via-cyan-300 to-blue-500 bg-clip-text text-transparent">
            ENTROPY
          </span>
          <br />
          <span className="text-white">ENGINE</span>
        </motion.h1>

        {/* Subtitle */}
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="text-slate-400 text-lg md:text-xl text-center mt-4 max-w-xl leading-relaxed"
        >
          AI-Powered Industrial Power Plant Optimization
          <br />
          <span className="text-slate-500 text-sm">
            Physics-Informed Neural Network · Model Predictive Control · Real-Time Safety
          </span>
        </motion.p>

        {/* Feature pills */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.6 }}
          className="flex flex-wrap justify-center gap-3 mt-8"
        >
          {["PINN Model", "MPC Controller", "Real-Time Safety", "Live 3D Viz"].map((tag, i) => (
            <span
              key={tag}
              className="px-4 py-1.5 rounded-full text-xs font-semibold bg-slate-800/60 border border-slate-700/40 text-slate-300"
            >
              {tag}
            </span>
          ))}
        </motion.div>

        {/* CTA Button */}
        <motion.button
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.8 }}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={onEnter}
          className="mt-12 px-10 py-4 rounded-2xl bg-gradient-to-r from-blue-500 to-cyan-400 text-white font-bold text-lg shadow-2xl shadow-blue-500/30 cursor-pointer relative overflow-hidden group"
        >
          <span className="relative z-10">Launch Dashboard →</span>
          <motion.div
            className="absolute inset-0 bg-gradient-to-r from-cyan-400 to-blue-500 opacity-0 group-hover:opacity-100 transition-opacity"
          />
        </motion.button>

        {/* Connection status */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.2 }}
          className="mt-8 flex items-center gap-2"
        >
          <div className={`w-2 h-2 rounded-full ${connected ? "bg-emerald-400" : "bg-red-400"} animate-pulse`} />
          <span className="text-[11px] text-slate-500">
            {connected ? "Backend connected · Simulation running" : "Connecting to backend..."}
          </span>
        </motion.div>
      </div>

      {/* Bottom gradient */}
      <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-navy-950 to-transparent" />
    </div>
  );
}
