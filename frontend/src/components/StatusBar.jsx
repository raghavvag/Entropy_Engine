import { motion } from "framer-motion";

export default function StatusBar({ state, connected }) {
  const metrics   = state?.metrics || {};
  const aiMode    = state?.ai_mode ?? false;
  const aiState   = state?.ai_state ?? "IDLE";
  const safety    = state?.safety_level ?? "NORMAL";
  const tick      = state?.tick_count ?? 0;
  const uptime    = state?.uptime ?? 0;
  const conf      = state?.confidence?.confidence ?? 0;

  const formatUptime = (s) => {
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}m ${sec}s`;
  };

  const stateColor = {
    IDLE:     "text-slate-400",
    ACTIVE:   "text-blue-400",
    FALLBACK: "text-amber-400",
  };

  const safetyColor = {
    NORMAL:   "text-emerald-400",
    WARNING:  "text-amber-400",
    CRITICAL: "text-red-400",
  };

  return (
    <div className="flex items-center justify-between px-6 py-3 border-b border-slate-800/80 bg-navy-950/80 backdrop-blur-lg sticky top-0 z-50">
      {/* Left: Brand */}
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-cyan-400 flex items-center justify-center text-white font-bold text-sm shadow-lg shadow-blue-500/20">
          E
        </div>
        <div>
          <h1 className="text-sm font-bold text-white tracking-tight">ENTROPY ENGINE</h1>
          <p className="text-[10px] text-slate-500 -mt-0.5">AI Power Optimization</p>
        </div>
      </div>

      {/* Center: Status pills */}
      <div className="flex items-center gap-3">
        <StatusPill label="AI" value={aiState} color={stateColor[aiState]} />
        <StatusPill label="Safety" value={safety} color={safetyColor[safety]} />
        <StatusPill
          label="Confidence"
          value={`${(conf * 100).toFixed(0)}%`}
          color={conf > 0.5 ? "text-emerald-400" : conf > 0.3 ? "text-amber-400" : "text-slate-400"}
        />
        <StatusPill label="Tick" value={tick} color="text-slate-300" />
      </div>

      {/* Right: Connection */}
      <div className="flex items-center gap-2">
        <span className="text-[10px] text-slate-500">{formatUptime(uptime)}</span>
        <motion.div
          className={`w-2 h-2 rounded-full ${connected ? "bg-emerald-400" : "bg-red-400"}`}
          animate={{ opacity: [1, 0.4, 1] }}
          transition={{ repeat: Infinity, duration: 2 }}
        />
      </div>
    </div>
  );
}

function StatusPill({ label, value, color }) {
  return (
    <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-slate-800/60 border border-slate-700/40">
      <span className="text-[10px] text-slate-500 uppercase">{label}</span>
      <span className={`text-xs font-bold font-mono-num ${color}`}>{value}</span>
    </div>
  );
}
