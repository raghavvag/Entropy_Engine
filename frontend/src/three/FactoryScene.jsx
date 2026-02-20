import { Canvas } from "@react-three/fiber";
import { OrbitControls, Environment, ContactShadows, Float } from "@react-three/drei";
import Furnace from "./Furnace";
import Pipe from "./Pipe";
import Turbine from "./Turbine";
import SteamParticles from "./SteamParticles";
import Floor from "./Floor";

export default function FactoryScene({ metrics, aiActive = false }) {
  const temp      = metrics?.temperature ?? 450;
  const valve     = metrics?.valve_position ?? 50;
  const power     = metrics?.power_output ?? 150;
  const pressure  = metrics?.pressure ?? 5;
  const pressHigh = pressure > 7.5;

  return (
    <div className="rounded-2xl overflow-hidden border border-slate-700/50 bg-slate-900/80 h-[400px] relative">
      {/* Scene label */}
      <div className="absolute top-3 left-4 z-10 flex items-center gap-2">
        <div className={`w-2 h-2 rounded-full ${aiActive ? "bg-blue-400 animate-pulse-glow" : "bg-slate-500"}`} />
        <span className="text-[10px] text-slate-400 uppercase tracking-widest font-semibold">
          Live Factory View
        </span>
      </div>

      <Canvas
        camera={{ position: [10, 7, 10], fov: 42 }}
        dpr={[1, 2]}
        gl={{ antialias: true }}
      >
        {/* Lighting */}
        <ambientLight intensity={0.25} />
        <directionalLight position={[8, 12, 5]} intensity={0.6} castShadow />
        <pointLight position={[-3, 5, 2]} intensity={0.4} color="#f59e0b" />

        {/* Factory floor */}
        <Floor />

        {/* Furnace (left) */}
        <Furnace
          temperature={temp}
          glowIntensity={(temp - 380) / 220}
        />

        {/* Pipes (furnace → turbine) */}
        <Pipe
          from={[-1.5, 1.2, 0]}
          to={[3.5, 1.2, 0]}
          flowSpeed={valve / 40}
          pressureHigh={pressHigh}
          segments={5}
        />

        {/* Secondary pipe (lower) */}
        <Pipe
          from={[-1.5, 0.5, 0.6]}
          to={[3.5, 0.5, 0.6]}
          flowSpeed={valve / 50}
          pressureHigh={pressHigh}
          segments={4}
        />

        {/* Turbine (right) */}
        <Turbine
          position={[5, 1, 0]}
          power={power}
          aiActive={aiActive}
        />

        {/* Steam from chimney */}
        <SteamParticles
          count={Math.floor(30 + valve)}
          speed={0.5 + valve / 80}
          position={[-3, 3.4, -0.5]}
        />

        {/* Contact shadows */}
        <ContactShadows
          position={[1, -0.29, 0]}
          opacity={0.4}
          scale={20}
          blur={2}
          far={6}
          color="#000000"
        />

        {/* Camera controls */}
        <OrbitControls
          enablePan={false}
          maxDistance={18}
          minDistance={6}
          maxPolarAngle={Math.PI / 2.1}
          autoRotate
          autoRotateSpeed={0.3}
        />

        {/* Environment lighting */}
        <Environment preset="warehouse" />
      </Canvas>
    </div>
  );
}
