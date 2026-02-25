import { Canvas } from "@react-three/fiber";
import { OrbitControls, Environment, ContactShadows } from "@react-three/drei";
import Furnace from "./Furnace";
import HeatExchanger from "./HeatExchanger";
import Pipe from "./Pipe";
import Turbine from "./Turbine";
import SteamParticles from "./SteamParticles";
import Floor from "./Floor";
import Worker from "./Worker";

/* ─────────────────────────────────────────────────────────
   FACTORY SCENE — Complete Industrial Layout
   Furnace → HeatExchanger → Turbine-Generator with multi-
   pipe routing.  Structural I-beam columns, overhead truss,
   control cabinet, safety railing, walkway, fire equipment.
   ───────────────────────────────────────────────────────── */

/* ── tiny reusable sub-components ────────────────────── */
function IBeamColumn({ position }) {
  return (
    <group position={position}>
      {/* web */}
      <mesh><boxGeometry args={[0.06, 5.5, 0.25]} /><meshStandardMaterial color="#374151" metalness={0.88} roughness={0.2} /></mesh>
      {/* flanges */}
      {[-0.03, 0.03].map((x) => (
        <mesh key={x} position={[x, 0, 0]}><boxGeometry args={[0.005, 5.5, 0.35]} /><meshStandardMaterial color="#4b5563" metalness={0.85} roughness={0.22} /></mesh>
      ))}
      {/* base plate */}
      <mesh position={[0, -2.75, 0]}><boxGeometry args={[0.5, 0.06, 0.5]} /><meshStandardMaterial color="#1f2937" metalness={0.85} roughness={0.25} /></mesh>
    </group>
  );
}

function ControlCabinet({ position }) {
  return (
    <group position={position}>
      <mesh><boxGeometry args={[0.8, 1.6, 0.45]} /><meshStandardMaterial color="#1e293b" metalness={0.82} roughness={0.28} /></mesh>
      {/* door seam */}
      <mesh position={[0, 0, 0.226]}><boxGeometry args={[0.65, 1.45, 0.005]} /><meshStandardMaterial color="#263040" metalness={0.85} roughness={0.22} /></mesh>
      {/* handle */}
      <mesh position={[0.25, 0, 0.24]}><boxGeometry args={[0.03, 0.15, 0.03]} /><meshStandardMaterial color="#9ca3af" metalness={0.95} roughness={0.08} /></mesh>
      {/* vents */}
      {[-0.15, -0.05, 0.05, 0.15].map((y) => (
        <mesh key={y} position={[0, y + 0.5, 0.228]}><boxGeometry args={[0.4, 0.02, 0.004]} /><meshStandardMaterial color="#111827" /></mesh>
      ))}
      {/* indicator lights */}
      {[[-0.15, -0.55, "#22c55e"], [0, -0.55, "#3b82f6"], [0.15, -0.55, "#f59e0b"]].map(([x, y, c]) => (
        <mesh key={x} position={[x, y, 0.228]}><sphereGeometry args={[0.02, 8, 6]} /><meshStandardMaterial color={c} emissive={c} emissiveIntensity={1.5} /></mesh>
      ))}
      {/* conduit out bottom */}
      <mesh position={[0, -1.0, 0]}><cylinderGeometry args={[0.02, 0.02, 0.4, 8]} /><meshStandardMaterial color="#4b5563" metalness={0.85} roughness={0.2} /></mesh>
    </group>
  );
}

function SafetyRailing({ position, length = 4, rotation = [0, 0, 0] }) {
  const postCount = Math.max(2, Math.ceil(length / 1.2));
  return (
    <group position={position} rotation={rotation}>
      {Array.from({ length: postCount }).map((_, i) => {
        const x = -length / 2 + (length / (postCount - 1)) * i;
        return (
          <group key={i} position={[x, 0, 0]}>
            {/* post */}
            <mesh position={[0, 0.45, 0]}><cylinderGeometry args={[0.02, 0.02, 0.9, 8]} /><meshStandardMaterial color="#eab308" metalness={0.7} roughness={0.3} /></mesh>
            {/* base */}
            <mesh position={[0, 0, 0]}><boxGeometry args={[0.12, 0.04, 0.12]} /><meshStandardMaterial color="#1f2937" metalness={0.85} roughness={0.25} /></mesh>
          </group>
        );
      })}
      {/* top rail */}
      <mesh position={[0, 0.9, 0]} rotation={[0, 0, Math.PI / 2]}>
        <cylinderGeometry args={[0.018, 0.018, length, 8]} />
        <meshStandardMaterial color="#eab308" metalness={0.7} roughness={0.3} />
      </mesh>
      {/* mid rail */}
      <mesh position={[0, 0.5, 0]} rotation={[0, 0, Math.PI / 2]}>
        <cylinderGeometry args={[0.014, 0.014, length, 8]} />
        <meshStandardMaterial color="#eab308" metalness={0.7} roughness={0.3} />
      </mesh>
    </group>
  );
}

function FireExtinguisher({ position }) {
  return (
    <group position={position}>
      <mesh position={[0, 0.22, 0]}><cylinderGeometry args={[0.06, 0.06, 0.45, 12]} /><meshStandardMaterial color="#dc2626" metalness={0.6} roughness={0.35} /></mesh>
      <mesh position={[0, 0.47, 0]}><cylinderGeometry args={[0.025, 0.025, 0.08, 8]} /><meshStandardMaterial color="#1f2937" metalness={0.9} roughness={0.15} /></mesh>
      <mesh position={[0.03, 0.5, 0]}><boxGeometry args={[0.06, 0.03, 0.02]} /><meshStandardMaterial color="#1f2937" metalness={0.85} roughness={0.2} /></mesh>
      {/* wall bracket */}
      <mesh position={[0, 0.22, -0.08]}><boxGeometry args={[0.15, 0.18, 0.04]} /><meshStandardMaterial color="#374151" metalness={0.85} roughness={0.25} /></mesh>
    </group>
  );
}

/* ─── MAIN SCENE ─────────────────────────────────────── */
export default function FactoryScene({ metrics, aiActive = false }) {
  const temp      = metrics?.temperature ?? 450;
  const valve     = metrics?.valve_position ?? 50;
  const power     = metrics?.power_output ?? 150;
  const pressure  = metrics?.pressure ?? 5;
  const pressHigh = pressure > 7.5;

  return (
    <div className="rounded-2xl overflow-hidden border border-slate-700/50 bg-slate-900/80 h-[420px] relative">
      {/* Scene label */}
      <div className="absolute top-3 left-4 z-10 flex items-center gap-2">
        <div className={`w-2 h-2 rounded-full ${aiActive ? "bg-blue-400 animate-pulse-glow" : "bg-slate-500"}`} />
        <span className="text-[10px] text-slate-400 uppercase tracking-widest font-semibold">
          Live Factory View
        </span>
      </div>

      <Canvas
        camera={{ position: [12, 8, 12], fov: 40 }}
        dpr={[1, 2]}
        gl={{ antialias: true, toneMapping: 3 /* ACESFilmic */ }}
        shadows
      >
        {/* ═══ LIGHTING ═══ */}
        <ambientLight intensity={0.2} color="#c7d2fe" />
        <directionalLight position={[10, 14, 6]} intensity={0.7} castShadow shadow-mapSize={1024} />
        <directionalLight position={[-8, 6, -4]} intensity={0.2} color="#fef3c7" />
        <pointLight position={[-4.5, 5, 2]} intensity={0.35} color="#f59e0b" />
        <pointLight position={[5, 4, 2]} intensity={0.25} color="#93c5fd" />

        {/* atmospheric fog */}
        <fog attach="fog" args={["#0a0e1a", 18, 40]} />

        {/* ═══ FLOOR ═══ */}
        <Floor />

        {/* ═══ FURNACE ═══ */}
        <Furnace temperature={temp} />

        {/* ═══ HEAT EXCHANGER (between furnace & turbine) ═══ */}
        <HeatExchanger position={[0, 0, 0]} pressure={pressure} temperature={temp} />

        {/* ═══ TURBINE-GENERATOR ═══ */}
        <Turbine position={[5, 0, 0]} power={power} aiActive={aiActive} />

        {/* ═══ PIPE ROUTING ═══ */}
        {/* Upper main steam line: furnace → heat exchanger */}
        <Pipe
          from={[-2.9, 1.6, 0]}
          to={[-0.6, 1.6, 0]}
          flowSpeed={valve / 35}
          pressureHigh={pressHigh}
          segments={4}
          radius={0.1}
        />
        {/* Upper main steam line: heat exchanger → turbine */}
        <Pipe
          from={[0.6, 1.6, 0]}
          to={[3.2, 1.6, 0]}
          flowSpeed={valve / 35}
          pressureHigh={pressHigh}
          segments={5}
          showValve
          radius={0.1}
        />

        {/* Lower return line: furnace → heat exchanger */}
        <Pipe
          from={[-2.9, 0.6, 0.6]}
          to={[-0.6, 0.6, 0.6]}
          flowSpeed={valve / 55}
          pressureHigh={false}
          segments={3}
          radius={0.07}
        />
        {/* Lower return line: heat exchanger → turbine */}
        <Pipe
          from={[0.6, 0.6, 0.6]}
          to={[3.2, 0.6, 0.6]}
          flowSpeed={valve / 55}
          pressureHigh={false}
          segments={3}
          radius={0.07}
        />

        {/* Small bypass pipe (upper, offset Z) */}
        <Pipe
          from={[-2.9, 1.2, -0.6]}
          to={[0.6, 1.2, -0.6]}
          flowSpeed={valve / 60}
          pressureHigh={pressHigh}
          segments={3}
          radius={0.06}
        />

        {/* ═══ STEAM FROM CHIMNEY ═══ */}
        <SteamParticles
          count={Math.floor(40 + valve * 0.6)}
          speed={0.4 + valve / 90}
          position={[-4.2, 4.3, -0.4]}
          spread={0.3}
          riseHeight={4}
        />

        {/* ═══ STRUCTURAL COLUMNS (4 I-beams) ═══ */}
        <IBeamColumn position={[-8, 2.4, -3.5]} />
        <IBeamColumn position={[-8, 2.4, 3.5]} />
        <IBeamColumn position={[9, 2.4, -3.5]} />
        <IBeamColumn position={[9, 2.4, 3.5]} />

        {/* overhead truss beam (front) */}
        <mesh position={[0.5, 5.15, 3.5]} rotation={[0, 0, Math.PI / 2]}>
          <boxGeometry args={[0.12, 17, 0.2]} />
          <meshStandardMaterial color="#374151" metalness={0.88} roughness={0.2} />
        </mesh>
        {/* overhead truss beam (back) */}
        <mesh position={[0.5, 5.15, -3.5]} rotation={[0, 0, Math.PI / 2]}>
          <boxGeometry args={[0.12, 17, 0.2]} />
          <meshStandardMaterial color="#374151" metalness={0.88} roughness={0.2} />
        </mesh>
        {/* cross beams */}
        {[-6, -2, 2, 6].map((x) => (
          <mesh key={`cb-${x}`} position={[x, 5.15, 0]} rotation={[Math.PI / 2, 0, 0]}>
            <boxGeometry args={[0.1, 7, 0.15]} />
            <meshStandardMaterial color="#374151" metalness={0.88} roughness={0.2} />
          </mesh>
        ))}

        {/* ═══ CONTROL CABINET ═══ */}
        <ControlCabinet position={[3, 0.8, -2.5]} />

        {/* ═══ SAFETY RAILING ═══ */}
        <SafetyRailing position={[-4.5, -0.35, 2.5]} length={4} />
        <SafetyRailing position={[5, -0.35, -2.5]} length={6} />

        {/* ═══ FIRE EXTINGUISHER ═══ */}
        <FireExtinguisher position={[3.5, 0, -2.85]} />

        {/* ═══ FACTORY WORKERS (4, patrolling) ═══ */}
        {/* Worker 1 — Furnace operator, paces near furnace */}
        <Worker
          path={[[-6, 0, 2], [-3, 0, 2], [-3, 0, -1.5], [-6, 0, -1.5]]}
          speed={0.5}
          color="#2563eb"
          hatColor="#eab308"
          startOffset={0}
        />
        {/* Worker 2 — Pipe inspector, walks along the pipe run */}
        <Worker
          path={[[-2, 0, 3], [1, 0, 3], [4, 0, 3], [4, 0, 1.5], [-2, 0, 1.5]]}
          speed={0.4}
          color="#dc2626"
          hatColor="#f97316"
          startOffset={0.3}
        />
        {/* Worker 3 — Turbine tech, circles the turbine */}
        <Worker
          path={[[3, 0, -1.5], [7, 0, -1.5], [7, 0, 1.5], [3, 0, 1.5]]}
          speed={0.55}
          color="#16a34a"
          hatColor="#ffffff"
          startOffset={0.6}
        />
        {/* Worker 4 — Supervisor, long perimeter walk */}
        <Worker
          path={[[-7, 0, -3], [8, 0, -3], [8, 0, 0], [0, 0, 0], [-7, 0, 0]]}
          speed={0.35}
          color="#7c3aed"
          hatColor="#eab308"
          startOffset={0.15}
        />

        {/* ═══ INDUSTRIAL OVERHEAD LIGHTS ═══ */}
        {[-3, 2, 7].map((x) => (
          <group key={`light-${x}`} position={[x, 5.0, 0]}>
            <mesh><boxGeometry args={[0.6, 0.06, 0.2]} /><meshStandardMaterial color="#374151" metalness={0.85} roughness={0.2} /></mesh>
            {/* lamp face */}
            <mesh position={[0, -0.04, 0]}>
              <boxGeometry args={[0.5, 0.02, 0.15]} />
              <meshStandardMaterial color="#fef9c3" emissive="#fef9c3" emissiveIntensity={0.15} />
            </mesh>
            {/* hanging rod */}
            <mesh position={[0, 0.18, 0]}><cylinderGeometry args={[0.01, 0.01, 0.3, 6]} /><meshStandardMaterial color="#6b7280" metalness={0.85} roughness={0.2} /></mesh>
          </group>
        ))}

        {/* ═══ CONTACT SHADOWS ═══ */}
        <ContactShadows position={[0.5, -0.34, 0]} opacity={0.5} scale={25} blur={2.5} far={7} color="#000000" />

        {/* ═══ CAMERA CONTROLS ═══ */}
        <OrbitControls
          enablePan={false}
          maxDistance={22}
          minDistance={6}
          maxPolarAngle={Math.PI / 2.1}
          autoRotate
          autoRotateSpeed={0.25}
        />

        {/* ═══ ENVIRONMENT ═══ */}
        <Environment preset="warehouse" />
      </Canvas>
    </div>
  );
}
