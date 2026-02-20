import { motion } from "framer-motion";

const COLOR_MAP = {
  blue:    { text: "text-blue-400",    bg: "bg-blue-500/10",  glow: "shadow-blue-500/10" },
  orange:  { text: "text-orange-400",  bg: "bg-orange-500/10", glow: "shadow-orange-500/10" },
  cyan:    { text: "text-cyan-400",    bg: "bg-cyan-500/10",   glow: "shadow-cyan-500/10" },
  emerald: { text: "text-emerald-400", bg: "bg-emerald-500/10", glow: "shadow-emerald-500/10" },
  purple:  { text: "text-purple-400",  bg: "bg-purple-500/10", glow: "shadow-purple-500/10" },
};

export default function KPICard({ label, value, unit, icon, color = "blue", alert, sub }) {
  const c = COLOR_MAP[color] || COLOR_MAP.blue;

  return (
    <motion.div
      className={`
        relative overflow-hidden rounded-2xl p-5
        bg-gradient-to-br from-slate-800/80 to-slate-900/80
        border border-slate-700/50 backdrop-blur-xl
        shadow-lg ${c.glow}
      `}
      whileHover={{ scale: 1.03, y: -2 }}
      transition={{ type: "spring", stiffness: 300, damping: 20 }}
    >
      {/* corner glow */}
      <div className={`absolute -top-10 -right-10 w-32 h-32 rounded-full ${c.bg} blur-3xl pointer-events-none`} />

      <div className="flex items-center justify-between mb-2">
        <span className="text-slate-400 text-xs font-semibold uppercase tracking-wider">{label}</span>
        <span className="text-xl">{icon}</span>
      </div>

      <motion.div
        key={value}
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="flex items-baseline gap-1.5"
      >
        <span className={`text-3xl font-bold font-mono-num ${c.text}`}>
          {typeof value === "number" ? value.toFixed(1) : value ?? "—"}
        </span>
        <span className="text-slate-500 text-sm">{unit}</span>
      </motion.div>

      {sub && <p className="mt-1 text-[11px] text-slate-500">{sub}</p>}

      {alert && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="mt-2 text-xs text-orange-400 font-medium"
        >
          ⚠️ {alert}
        </motion.div>
      )}
    </motion.div>
  );
}
