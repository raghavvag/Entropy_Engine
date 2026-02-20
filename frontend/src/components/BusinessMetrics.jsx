import { motion } from "framer-motion";

function getBusinessMetrics(aiPower, baselinePower) {
  const extra = aiPower - baselinePower;
  if (extra <= 0) return { energySaved: 0, co2Reduced: 0, monthlySavings: 0, annualSavings: 0 };
  return {
    energySaved:    Math.round(extra),                              // kWh/hr
    co2Reduced:     Math.round(extra * 0.4),                        // kg/hr
    monthlySavings: Math.round(extra * 8 * 30 * 7.2),              // ₹/month
    annualSavings:  Math.round(extra * 8 * 365 * 7.2),             // ₹/year
  };
}

const ITEMS = [
  { key: "energySaved",    label: "Energy Recovered", unit: "kWh/hr", icon: "⚡" },
  { key: "co2Reduced",     label: "CO₂ Reduced",      unit: "kg/hr",  icon: "🌱" },
  { key: "monthlySavings", label: "Monthly Savings",   prefix: "₹",   icon: "💰" },
  { key: "annualSavings",  label: "Annual Impact",     prefix: "₹",   icon: "📈" },
];

export default function BusinessMetrics({ comparison }) {
  if (!comparison) return null;

  const bm = getBusinessMetrics(comparison.ai_avg_power || 0, comparison.baseline_avg_power || 0);

  return (
    <div className="glass-card p-6">
      <h2 className="text-xs text-slate-400 font-semibold uppercase tracking-wider mb-5">
        💼 Business Impact
      </h2>

      <div className="grid grid-cols-4 gap-4">
        {ITEMS.map((item, i) => {
          const val = bm[item.key];
          const display = item.prefix
            ? `${item.prefix}${val.toLocaleString()}`
            : val.toLocaleString();

          return (
            <motion.div
              key={item.key}
              className="text-center"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.1 }}
            >
              <span className="text-2xl">{item.icon}</span>
              <p className="text-lg font-bold font-mono-num text-emerald-400 mt-2">
                {display}
              </p>
              {item.unit && <p className="text-[10px] text-slate-500">{item.unit}</p>}
              <p className="text-[11px] text-slate-400 mt-1">{item.label}</p>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
