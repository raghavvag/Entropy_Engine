import { motion } from "framer-motion";
import { IconAlertTriangle } from "./Icons";

const COLOR_MAP = {
  blue:    { text: "text-blue-400",    bg: "bg-blue-500/10",  glow: "shadow-blue-500/10",    iconBg: "bg-blue-500/15" },
  orange:  { text: "text-orange-400",  bg: "bg-orange-500/10", glow: "shadow-orange-500/10", iconBg: "bg-orange-500/15" },
  cyan:    { text: "text-cyan-400",    bg: "bg-cyan-500/10",   glow: "shadow-cyan-500/10",   iconBg: "bg-cyan-500/15" },
  emerald: { text: "text-emerald-400", bg: "bg-emerald-500/10", glow: "shadow-emerald-500/10", iconBg: "bg-emerald-500/15" },
  purple:  { text: "text-purple-400",  bg: "bg-purple-500/10", glow: "shadow-purple-500/10", iconBg: "bg-purple-500/15" },
};

export default function KPICard({ label, value, unit, icon: Icon, color = "blue", alert, sub }) {
  const c = COLOR_MAP[color] || COLOR_MAP.blue;

  return (
    <motion.div
      className={`
        relative overflow-hidden rounded-2xl p-5
        bg-gradient-to-br from-slate-800/80 to-slate-900/80
        border border-slate-700/50 backdrop-blur-xl
        shadow-lg ${c.glow}
      `}
      whileHover={{ scale: 1.02, y: -1 }}
      transition={{ type: "spring", stiffness: 300, damping: 20 }}
    >
      {/* corner glow */}
      <div className={`absolute -top-10 -right-10 w-32 h-32 rounded-full ${c.bg} blur-3xl pointer-events-none`} />

      <div className="flex items-center justify-between mb-3">
        <span className="text-slate-400 text-[10px] font-semibold uppercase tracking-widest">{label}</span>
        {Icon && (
          <div className={`w-7 h-7 rounded-lg ${c.iconBg} flex items-center justify-center ${c.text}`}>
            <Icon className="w-3.5 h-3.5" />
          </div>
        )}
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

      {sub && <p className="mt-1.5 text-[11px] text-slate-500">{sub}</p>}

      {alert && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="mt-2 flex items-center gap-1.5 text-xs text-orange-400 font-medium"
        >
          <IconAlertTriangle className="w-3 h-3" />
          <span>{alert}</span>
        </motion.div>
      )}
    </motion.div>
  );
}
