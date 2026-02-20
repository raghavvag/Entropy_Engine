import { motion } from "framer-motion";

const LEVEL_CONFIG = {
  NORMAL:   { color: "text-emerald-400", bg: "bg-emerald-500/10", border: "border-emerald-500/30", icon: "🟢", label: "ALL CLEAR" },
  WARNING:  { color: "text-amber-400",   bg: "bg-amber-500/10",   border: "border-amber-500/30",   icon: "🟡", label: "WARNING" },
  CRITICAL: { color: "text-red-400",     bg: "bg-red-500/10",     border: "border-red-500/30",     icon: "🔴", label: "CRITICAL" },
};

export default function SafetyIndicator({ level = "NORMAL", overrides = 0, pressureHeadroom, tempHeadroom }) {
  const cfg = LEVEL_CONFIG[level] || LEVEL_CONFIG.NORMAL;

  return (
    <motion.div
      className={`glass-card p-4 border ${cfg.border}`}
      animate={level === "CRITICAL" ? { scale: [1, 1.01, 1] } : {}}
      transition={level === "CRITICAL" ? { repeat: Infinity, duration: 0.8 } : {}}
    >
      <div className="flex items-center gap-2 mb-3">
        <span>{cfg.icon}</span>
        <p className={`text-xs font-bold uppercase tracking-wider ${cfg.color}`}>
          Safety: {cfg.label}
        </p>
      </div>

      <div className="grid grid-cols-2 gap-3 text-[11px]">
        <div>
          <p className="text-slate-500">Pressure Headroom</p>
          <p className="font-mono-num text-slate-300 font-semibold">
            {pressureHeadroom != null ? `${pressureHeadroom.toFixed(2)} bar` : "—"}
          </p>
        </div>
        <div>
          <p className="text-slate-500">Temp Headroom</p>
          <p className="font-mono-num text-slate-300 font-semibold">
            {tempHeadroom != null ? `${tempHeadroom.toFixed(0)} °C` : "—"}
          </p>
        </div>
        <div>
          <p className="text-slate-500">Safety Overrides</p>
          <p className={`font-mono-num font-semibold ${overrides > 0 ? "text-amber-400" : "text-emerald-400"}`}>
            {overrides}
          </p>
        </div>
      </div>
    </motion.div>
  );
}
