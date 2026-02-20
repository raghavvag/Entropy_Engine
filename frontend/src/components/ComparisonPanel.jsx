import { motion } from "framer-motion";

export default function ComparisonPanel({ comparison }) {
  if (!comparison) return null;

  const { baseline_avg_power, ai_avg_power, improvement_pct, baseline_samples, ai_samples } = comparison;
  const isPositive = improvement_pct > 0;

  return (
    <div className="glass-card p-6">
      <h2 className="text-xs text-slate-400 font-semibold uppercase tracking-wider mb-5">
        ⚡ AI Impact Analysis
      </h2>

      <div className="grid grid-cols-3 gap-4 items-center">
        {/* Baseline */}
        <div className="text-center p-4 rounded-xl bg-slate-800/40">
          <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">Baseline</p>
          <motion.p
            key={baseline_avg_power}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-2xl font-bold font-mono-num text-slate-300"
          >
            {baseline_avg_power?.toFixed(1) ?? "—"}
          </motion.p>
          <p className="text-[11px] text-slate-500 mt-0.5">kW avg</p>
          <p className="text-[10px] text-slate-600 mt-1">{baseline_samples} samples</p>
        </div>

        {/* Arrow + Improvement */}
        <div className="flex flex-col items-center justify-center">
          <motion.span
            className="text-3xl text-slate-600"
            animate={{ x: [0, 6, 0] }}
            transition={{ repeat: Infinity, duration: 1.5, ease: "easeInOut" }}
          >
            →
          </motion.span>
          <motion.p
            key={improvement_pct}
            initial={{ scale: 0.8 }}
            animate={{ scale: 1 }}
            className={`text-xl font-bold font-mono-num mt-1 ${
              isPositive ? "text-emerald-400" : improvement_pct < 0 ? "text-red-400" : "text-slate-400"
            }`}
          >
            {isPositive ? "+" : ""}{improvement_pct?.toFixed(1) ?? 0}%
          </motion.p>
          <p className="text-[10px] text-slate-500 mt-0.5">improvement</p>
        </div>

        {/* AI Optimized */}
        <div className={`text-center p-4 rounded-xl border ${
          isPositive
            ? "bg-blue-900/20 border-blue-500/20 glow-blue"
            : "bg-slate-800/40 border-slate-700/30"
        }`}>
          <p className="text-[10px] text-blue-400 uppercase tracking-wider mb-1">AI Optimized</p>
          <motion.p
            key={ai_avg_power}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-2xl font-bold font-mono-num text-blue-300"
          >
            {ai_avg_power?.toFixed(1) ?? "—"}
          </motion.p>
          <p className="text-[11px] text-blue-400/60 mt-0.5">kW avg</p>
          <p className="text-[10px] text-slate-600 mt-1">{ai_samples} samples</p>
        </div>
      </div>
    </div>
  );
}
