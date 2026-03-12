import { motion } from "framer-motion";
import { toggleAI } from "../services/api";

export default function AIToggle({ enabled, onToggle }) {
  const handleToggle = async () => {
    try {
      const next = !enabled;
      await toggleAI(next);
      onToggle(next);
    } catch (e) {
      console.error("Toggle failed:", e);
    }
  };

  return (
    <motion.div
      className={`
        glass-card flex items-center gap-4 p-4
        ${enabled ? "glow-blue border-blue-500/30" : ""}
      `}
      layout
    >
      <div className="flex-1">
        <p className="text-slate-200 font-semibold text-sm tracking-tight">AI Optimization</p>
        <p className="text-[11px] text-slate-500 mt-0.5 leading-relaxed">
          {enabled ? "PINN + MPC actively controlling plant" : "Manual baseline — AI standing by"}
        </p>
      </div>

      <button onClick={handleToggle} className="relative w-14 h-7 rounded-full cursor-pointer flex-shrink-0 transition-colors duration-200"
        style={{ background: enabled ? "#3b82f6" : "#1e293b" }}
      >
        <motion.div
          className="absolute top-0.5 w-6 h-6 rounded-full bg-white shadow-lg"
          animate={{ left: enabled ? 30 : 2 }}
          transition={{ type: "spring", stiffness: 500, damping: 30 }}
        />
      </button>

      <motion.span
        key={String(enabled)}
        initial={{ opacity: 0, x: -5 }}
        animate={{ opacity: 1, x: 0 }}
        className={`text-xs font-bold tracking-wider min-w-[52px] text-right ${
          enabled ? "text-blue-400" : "text-slate-500"
        }`}
      >
        {enabled ? "ACTIVE" : "OFF"}
      </motion.span>
    </motion.div>
  );
}
