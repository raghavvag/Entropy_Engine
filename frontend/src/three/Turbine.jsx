import { useRef, useMemo } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";

/* ─────────────────────────────────────────────────────────
   TURBINE — Steam Turbine-Generator Set
   Multi-section housing: inlet cone → turbine casing →
   coupling → generator body → exciter. 8 visible blades,
   exhaust port, cooling fins, bearing pedestals, control
   box, foundation. AI-reactive blue glow.
   ───────────────────────────────────────────────────────── */
export default function Turbine({ position = [5, 0, 0], power = 150, aiActive = false }) {
  const bladeGroupRef = useRef();
  const aiGlowRef = useRef();
  const shaftRef = useRef();

  const speed = Math.max(0.3, power / 60);
  const pn = Math.max(0, Math.min(1, (power - 100) / 200));

  useFrame((_, delta) => {
    if (bladeGroupRef.current) bladeGroupRef.current.rotation.z += delta * speed * 2;
    if (shaftRef.current) shaftRef.current.rotation.z += delta * speed * 2;
    if (aiGlowRef.current)
      aiGlowRef.current.intensity = aiActive
        ? 1.8 + Math.sin(Date.now() * 0.003) * 0.6
        : 0;
  });

  const aiColor = aiActive ? "#3b82f6" : "#475569";
  const aiEmissive = aiActive ? new THREE.Color(0.15, 0.35, 1) : new THREE.Color(0, 0, 0);

  return (
    <group position={position}>
      {/* ══════ FOUNDATION ══════ */}
      <mesh position={[0, -0.3, 0]} receiveShadow>
        <boxGeometry args={[5.6, 0.15, 2.8]} />
        <meshStandardMaterial color="#374151" roughness={0.92} metalness={0.1} />
      </mesh>
      <mesh position={[0, -0.15, 0]}>
        <boxGeometry args={[5.2, 0.16, 2.5]} />
        <meshStandardMaterial color="#1f2937" metalness={0.85} roughness={0.25} />
      </mesh>

      {/* ══════ BEARING PEDESTALS (2) ══════ */}
      {[-1.4, 1.4].map((x) => (
        <group key={`ped-${x}`} position={[x, 0.15, 0]}>
          <mesh><boxGeometry args={[0.5, 0.4, 0.8]} /><meshStandardMaterial color="#374151" metalness={0.88} roughness={0.2} /></mesh>
          {/* bearing cap (semi-circle top) */}
          <mesh position={[0, 0.22, 0]} rotation={[Math.PI / 2, 0, 0]}>
            <cylinderGeometry args={[0.18, 0.18, 0.52, 16, 1, false, 0, Math.PI]} />
            <meshStandardMaterial color="#4b5563" metalness={0.9} roughness={0.15} />
          </mesh>
          {/* bolts */}
          {[[-0.18, -0.05, 0.35], [0.18, -0.05, 0.35], [-0.18, -0.05, -0.35], [0.18, -0.05, -0.35]].map(([bx, by, bz], i) => (
            <mesh key={`bolt-${i}`} position={[bx, by, bz]}>
              <cylinderGeometry args={[0.025, 0.025, 0.12, 6]} />
              <meshStandardMaterial color="#9ca3af" metalness={0.95} roughness={0.08} />
            </mesh>
          ))}
        </group>
      ))}

      {/* ══════ INLET NOZZLE (steam entry, top-left) ══════ */}
      <group position={[-1.8, 1.1, 0]}>
        <mesh><cylinderGeometry args={[0.14, 0.14, 0.7, 12]} /><meshStandardMaterial color="#4b5563" metalness={0.85} roughness={0.22} /></mesh>
        <mesh position={[0, 0.38, 0]}><cylinderGeometry args={[0.2, 0.2, 0.06, 12]} /><meshStandardMaterial color="#374151" metalness={0.92} roughness={0.12} /></mesh>
        {/* 90° elbow into casing */}
        <mesh position={[0.15, -0.2, 0]} rotation={[0, 0, Math.PI / 4]}>
          <cylinderGeometry args={[0.12, 0.12, 0.35, 12]} />
          <meshStandardMaterial color="#4b5563" metalness={0.85} roughness={0.22} />
        </mesh>
      </group>

      {/* ══════ TURBINE CASING (tapered, left section) ══════ */}
      <mesh position={[-0.7, 0.7, 0]} rotation={[Math.PI / 2, 0, Math.PI / 2]} castShadow>
        <cylinderGeometry args={[0.75, 0.95, 1.6, 24]} />
        <meshStandardMaterial color="#4b5563" metalness={0.85} roughness={0.18} />
      </mesh>
      {/* casing flange rings */}
      {[-1.5, 0.1].map((x) => (
        <mesh key={`cf-${x}`} position={[x, 0.7, 0]} rotation={[Math.PI / 2, 0, Math.PI / 2]}>
          <cylinderGeometry args={[1.0, 1.0, 0.06, 24]} />
          <meshStandardMaterial color="#374151" metalness={0.92} roughness={0.12} />
        </mesh>
      ))}

      {/* ══════ COUPLING SECTION ══════ */}
      <mesh position={[0.35, 0.7, 0]} rotation={[Math.PI / 2, 0, Math.PI / 2]}>
        <cylinderGeometry args={[0.35, 0.35, 0.45, 16]} />
        <meshStandardMaterial color="#374151" metalness={0.9} roughness={0.15} />
      </mesh>
      {/* coupling bolts (visible ring of bolts) */}
      {[0, 45, 90, 135, 180, 225, 270, 315].map((deg) => {
        const r = (deg * Math.PI) / 180;
        return (
          <mesh key={`cb-${deg}`} position={[0.35, 0.7 + Math.sin(r) * 0.28, Math.cos(r) * 0.28]}>
            <sphereGeometry args={[0.025, 8, 6]} />
            <meshStandardMaterial color="#9ca3af" metalness={0.95} roughness={0.08} />
          </mesh>
        );
      })}

      {/* ══════ GENERATOR BODY (larger cylinder, right section) ══════ */}
      <mesh position={[1.3, 0.7, 0]} rotation={[Math.PI / 2, 0, Math.PI / 2]} castShadow>
        <cylinderGeometry args={[0.85, 0.85, 1.5, 24]} />
        <meshStandardMaterial
          color="#4b5563" metalness={0.82} roughness={0.2}
          emissive={aiEmissive} emissiveIntensity={aiActive ? 0.15 : 0}
        />
      </mesh>
      {/* generator end cap */}
      <mesh position={[2.1, 0.7, 0]} rotation={[Math.PI / 2, 0, Math.PI / 2]}>
        <cylinderGeometry args={[0.6, 0.85, 0.12, 24]} />
        <meshStandardMaterial color="#374151" metalness={0.9} roughness={0.12} />
      </mesh>

      {/* ══════ COOLING FINS (on generator body) ══════ */}
      {[0.75, 1.05, 1.35, 1.65].map((x) => (
        <mesh key={`fin-${x}`} position={[x, 0.7, 0]} rotation={[Math.PI / 2, 0, Math.PI / 2]}>
          <cylinderGeometry args={[0.92, 0.92, 0.02, 24]} />
          <meshStandardMaterial color="#6b7280" metalness={0.88} roughness={0.15} />
        </mesh>
      ))}

      {/* ══════ EXCITER (small cylinder at far right end) ══════ */}
      <mesh position={[2.45, 0.7, 0]} rotation={[Math.PI / 2, 0, Math.PI / 2]}>
        <cylinderGeometry args={[0.35, 0.35, 0.55, 16]} />
        <meshStandardMaterial color="#6b7280" metalness={0.8} roughness={0.25} />
      </mesh>

      {/* ══════ BLADE ASSEMBLY (visible on intake face) ══════ */}
      <group position={[-1.52, 0.7, 0]}>
        {/* outer intake ring */}
        <mesh>
          <torusGeometry args={[0.85, 0.06, 10, 32]} />
          <meshStandardMaterial
            color={aiColor} emissive={aiEmissive}
            emissiveIntensity={aiActive ? 0.8 : 0} metalness={0.9} roughness={0.1}
          />
        </mesh>
        {/* inner ring */}
        <mesh>
          <torusGeometry args={[0.55, 0.03, 8, 24]} />
          <meshStandardMaterial color="#475569" metalness={0.85} roughness={0.15} />
        </mesh>
        {/* spinning blades */}
        <group ref={bladeGroupRef}>
          {[0, 45, 90, 135, 180, 225, 270, 315].map((deg) => (
            <mesh key={deg} rotation={[0, 0, (deg * Math.PI) / 180]} position={[0, 0, 0.01]}>
              <boxGeometry args={[0.07, 0.78, 0.025]} />
              <meshStandardMaterial
                color={aiActive ? "#60a5fa" : "#64748b"}
                emissive={aiActive ? new THREE.Color(0.1, 0.3, 0.8) : new THREE.Color(0, 0, 0)}
                emissiveIntensity={aiActive ? 0.6 : 0}
                metalness={0.75} roughness={0.25}
              />
            </mesh>
          ))}
          {/* hub */}
          <mesh position={[0, 0, 0.015]}>
            <cylinderGeometry args={[0.12, 0.12, 0.06, 16]} rotation={[Math.PI / 2, 0, 0]} />
            <meshStandardMaterial color="#9ca3af" metalness={0.92} roughness={0.08} />
          </mesh>
          {/* hub cap bolt */}
          <mesh position={[0, 0, 0.05]}>
            <cylinderGeometry args={[0.04, 0.04, 0.02, 6]} rotation={[Math.PI / 2, 0, 0]} />
            <meshStandardMaterial color="#d1d5db" metalness={0.95} roughness={0.05} />
          </mesh>
        </group>
      </group>

      {/* ══════ EXHAUST PORT (underneath, center) ══════ */}
      <group position={[-0.3, -0.08, 0]}>
        <mesh>
          <cylinderGeometry args={[0.18, 0.22, 0.5, 12]} />
          <meshStandardMaterial color="#4b5563" metalness={0.85} roughness={0.22} />
        </mesh>
        <mesh position={[0, -0.27, 0]}>
          <cylinderGeometry args={[0.27, 0.27, 0.05, 12]} />
          <meshStandardMaterial color="#374151" metalness={0.92} roughness={0.12} />
        </mesh>
      </group>

      {/* ══════ CONTROL BOX (side-mounted) ══════ */}
      <group position={[1.3, 0.7, 1.0]}>
        <mesh><boxGeometry args={[0.5, 0.4, 0.15]} /><meshStandardMaterial color="#1f2937" metalness={0.85} roughness={0.25} /></mesh>
        {/* indicator lights */}
        {[[-0.12, 0.12], [0.04, 0.12], [0.2, 0.12]].map(([x, y], i) => (
          <mesh key={`ind-${i}`} position={[x, y, 0.08]}>
            <sphereGeometry args={[0.022, 8, 6]} />
            <meshStandardMaterial
              color={i === 0 ? "#22c55e" : i === 1 ? (aiActive ? "#3b82f6" : "#475569") : "#f59e0b"}
              emissive={i === 0 ? "#22c55e" : i === 1 ? (aiActive ? "#3b82f6" : "#000") : "#f59e0b"}
              emissiveIntensity={1.5}
            />
          </mesh>
        ))}
        {/* conduit */}
        <mesh position={[0, -0.35, 0]} rotation={[0, 0, 0]}>
          <cylinderGeometry args={[0.02, 0.02, 0.3, 8]} />
          <meshStandardMaterial color="#4b5563" metalness={0.85} roughness={0.2} />
        </mesh>
      </group>

      {/* ══════ NAMEPLATE ══════ */}
      <mesh position={[1.3, 1.45, 0.01]} rotation={[0, 0, 0]}>
        <boxGeometry args={[0.6, 0.12, 0.006]} />
        <meshStandardMaterial color="#b8860b" metalness={0.95} roughness={0.08} />
      </mesh>

      {/* ══════ RPM GAUGE (front of turbine casing) ══════ */}
      <group position={[-0.7, 1.45, 0.7]}>
        <mesh rotation={[0, 0, 0]}>
          <cylinderGeometry args={[0.08, 0.08, 0.025, 16]} rotation={[Math.PI / 2, 0, 0]} />
          <meshStandardMaterial color="#1c1917" metalness={0.9} roughness={0.15} />
        </mesh>
        <mesh position={[0, 0, 0.015]}>
          <circleGeometry args={[0.065, 20]} />
          <meshStandardMaterial color="#f5f5f4" roughness={0.8} side={THREE.DoubleSide} />
        </mesh>
      </group>

      {/* ══════ AI GLOW ══════ */}
      <pointLight
        ref={aiGlowRef}
        position={[-1.5, 0.7, 1.2]}
        color="#3b82f6" distance={5} intensity={0}
      />
      {/* power glow (warmth from running) */}
      <pointLight position={[0.5, 0.7, 0]} color="#fef3c7" distance={3} intensity={pn * 0.8} />
    </group>
  );
}
