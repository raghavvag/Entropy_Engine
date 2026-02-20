import {
  LineChart, Line, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid, Area, AreaChart,
} from "recharts";
import { THEME } from "../constants/theme";

function CustomTooltip({ active, payload, label, unit, name }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="glass-card px-3 py-2 text-xs">
      <p className="text-slate-500 mb-1">Tick {label}</p>
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.stroke || p.color }}>
          {p.name}: <span className="font-mono-num font-semibold">{Number(p.value).toFixed(2)}</span> {unit}
        </p>
      ))}
    </div>
  );
}

export default function LiveChart({
  data,
  lines = [],        // [{ key, color, name }]
  label,
  unit = "",
  height = 180,
  area = false,
}) {
  const ChartComp = area ? AreaChart : LineChart;
  const LineComp  = area ? Area : Line;

  return (
    <div className="glass-card p-4">
      <h3 className="text-xs text-slate-400 font-semibold uppercase tracking-wider mb-3">
        {label}
      </h3>
      <ResponsiveContainer width="100%" height={height}>
        <ChartComp data={data} margin={{ top: 4, right: 8, bottom: 0, left: -10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={THEME.chart.grid} opacity={0.4} />
          <XAxis
            dataKey="tick"
            stroke="#475569"
            fontSize={10}
            tickLine={false}
            axisLine={false}
          />
          <YAxis stroke="#475569" fontSize={10} tickLine={false} axisLine={false} />
          <Tooltip content={<CustomTooltip unit={unit} />} />
          {lines.map(({ key, color, name }) =>
            area ? (
              <Area
                key={key}
                type="monotone"
                dataKey={key}
                stroke={color}
                fill={color}
                fillOpacity={0.08}
                strokeWidth={2}
                dot={false}
                name={name || key}
                animationDuration={300}
              />
            ) : (
              <Line
                key={key}
                type="monotone"
                dataKey={key}
                stroke={color}
                strokeWidth={2}
                dot={false}
                name={name || key}
                animationDuration={300}
              />
            )
          )}
        </ChartComp>
      </ResponsiveContainer>
    </div>
  );
}
