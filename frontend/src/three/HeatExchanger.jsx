import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";

/* ─────────────────────────────────────────────────────────
   HEAT EXCHANGER — Shell-and-Tube
   Horizontal cylindrical shell with visible tube bundle,
   head caps, input/output nozzles, support saddles,
   baffles, and pressure gauge. Pressure-reactive.
   ───────────────────────────────────────────────────────── */
export default function HeatExchanger({ position = [0, 0, 0], pressure = 5, temperature = 500 }) {
  const glowRef = useRef();
  const pn = Math.max(0, Math.min(1, (pressure - 4) / 4));
  const tn = Math.max(0, Math.min(1, (temperature - 380) / 220));

  useFrame(({ clock }) => {
    if (glowRef.current)
      glowRef.current.intensity = tn * 1.2 + Math.sin(clock.elapsedTime * 3) * 0.2;
  });

  const shellColor = pn > 0.85 ? "#b91c1c" : "#4b5563";

  return (
    <group position={position}>
      {/* ══════ SUPPORT SADDLES (2) ══════ */}
      {[-0.6, 0.6].map((z) => (
        <group key={`saddle-${z}`} position={[0, 0.1, z]}>
          <mesh>
            <boxGeometry args={[0.9, 0.3, 0.12]} />
            <meshStandardMaterial color="#374151" metalness={0.88} roughness={0.2} />
          </mesh>
          {/* curved cradle */}
          <mesh position={[0, 0.28, 0]} rotation={[0, 0, 0]}>
            <cylinderGeometry args={[0.52, 0.52, 0.1, 16, 1, false, 0, Math.PI]} />
            <meshStandardMaterial color="#4b5563" metalness={0.85} roughness={0.2} />
          </mesh>
          {/* base plate */}
          <mesh position={[0, -0.17, 0]}>
            <boxGeometry args={[1.1, 0.04, 0.2]} />
            <meshStandardMaterial color="#1f2937" metalness={0.85} roughness={0.25} />
          </mesh>
        </group>
      ))}

      {/* ══════ SHELL BODY (horizontal cylinder) ══════ */}
      <mesh position={[0, 0.65, 0]} rotation={[Math.PI / 2, 0, 0]} castShadow>
        <cylinderGeometry args={[0.48, 0.48, 1.8, 20]} />
        <meshStandardMaterial color={shellColor} metalness={0.82} roughness={0.2} />
      </mesh>

      {/* baffle plates visible through shell (decorative rings) */}
      {[-0.5, 0, 0.5].map((z) => (
        <mesh key={`baffle-${z}`} position={[0, 0.65, z]} rotation={[Math.PI / 2, 0, 0]}>
          <torusGeometry args={[0.49, 0.015, 8, 20]} />
          <meshStandardMaterial color="#374151" metalness={0.9} roughness={0.12} />
        </mesh>
      ))}

      {/* ══════ HEAD CAPS (both ends) ══════ */}
      {[-0.92, 0.92].map((z) => (
        <group key={`cap-${z}`} position={[0, 0.65, z]}>
          {/* flange */}
          <mesh rotation={[Math.PI / 2, 0, 0]}>
            <cylinderGeometry args={[0.55, 0.55, 0.06, 20]} />
            <meshStandardMaterial color="#374151" metalness={0.92} roughness={0.12} />
          </mesh>
          {/* dished end */}
          <mesh position={[0, 0, z > 0 ? 0.06 : -0.06]} rotation={[Math.PI / 2, 0, 0]}>
            <sphereGeometry args={[0.47, 16, 8, 0, Math.PI * 2, 0, Math.PI / 3]} />
            <meshStandardMaterial
              color="#6b7280" metalness={0.8} roughness={0.22}
              side={z > 0 ? THREE.FrontSide : THREE.BackSide}
            />
          </mesh>
          {/* flange bolts */}
          {[0, 60, 120, 180, 240, 300].map((deg) => {
            const r = (deg * Math.PI) / 180;
            return (
              <mesh key={`fb-${z}-${deg}`} position={[Math.cos(r) * 0.5, Math.sin(r) * 0.5, 0]}>
                <sphereGeometry args={[0.02, 6, 6]} />
                <meshStandardMaterial color="#9ca3af" metalness={0.95} roughness={0.08} />
              </mesh>
            );
          })}
        </group>
      ))}

      {/* ══════ TUBE BUNDLE (visible on one end) ══════ */}
      <group position={[0, 0.65, -0.98]}>
        {[
          [0, 0], [0.15, 0.1], [-0.15, 0.1], [0.15, -0.1], [-0.15, -0.1],
          [0, 0.2], [0, -0.2], [0.28, 0], [-0.28, 0],
        ].map(([x, y], i) => (
          <mesh key={`tube-${i}`} position={[x, y, 0]} rotation={[Math.PI / 2, 0, 0]}>
            <cylinderGeometry args={[0.03, 0.03, 0.06, 8]} />
            <meshStandardMaterial color="#b87333" metalness={0.85} roughness={0.15} />
          </mesh>
        ))}
      </group>

      {/* ══════ INPUT / OUTPUT NOZZLES ══════ */}
      {/* hot input (top) */}
      <group position={[0, 1.25, -0.3]}>
        <mesh><cylinderGeometry args={[0.1, 0.1, 0.3, 12]} /><meshStandardMaterial color="#4b5563" metalness={0.85} roughness={0.22} /></mesh>
        <mesh position={[0, 0.17, 0]}><cylinderGeometry args={[0.15, 0.15, 0.05, 12]} /><meshStandardMaterial color="#374151" metalness={0.92} roughness={0.12} /></mesh>
      </group>
      {/* cooled output (bottom) */}
      <group position={[0, 0.08, 0.3]}>
        <mesh><cylinderGeometry args={[0.1, 0.1, 0.3, 12]} /><meshStandardMaterial color="#4b5563" metalness={0.85} roughness={0.22} /></mesh>
        <mesh position={[0, -0.17, 0]}><cylinderGeometry args={[0.15, 0.15, 0.05, 12]} /><meshStandardMaterial color="#374151" metalness={0.92} roughness={0.12} /></mesh>
      </group>

      {/* side pass nozzles (left/right) */}
      {[-1, 1].map((side) => (
        <group key={`sn-${side}`} position={[side * 0.5, 0.65, 0]} rotation={[0, 0, (side * Math.PI) / 2]}>
          <mesh><cylinderGeometry args={[0.08, 0.08, 0.25, 10]} /><meshStandardMaterial color="#4b5563" metalness={0.85} roughness={0.22} /></mesh>
          <mesh position={[0, 0.14, 0]}><cylinderGeometry args={[0.12, 0.12, 0.04, 10]} /><meshStandardMaterial color="#374151" metalness={0.92} roughness={0.12} /></mesh>
        </group>
      ))}

      {/* ══════ PRESSURE GAUGE ══════ */}
      <group position={[0.35, 1.05, 0.35]}>
        <mesh><cylinderGeometry args={[0.06, 0.06, 0.02, 14]} rotation={[Math.PI / 2, 0, 0]} /><meshStandardMaterial color="#1c1917" metalness={0.92} roughness={0.12} /></mesh>
        <mesh position={[0, 0, 0.012]}><circleGeometry args={[0.048, 16]} /><meshStandardMaterial color="#f5f5f4" roughness={0.8} side={THREE.DoubleSide} /></mesh>
        {/* stem */}
        <mesh position={[0, -0.12, 0]}><cylinderGeometry args={[0.015, 0.015, 0.2, 8]} /><meshStandardMaterial color="#4b5563" metalness={0.85} roughness={0.2} /></mesh>
      </group>

      {/* ══════ NAMEPLATE ══════ */}
      <mesh position={[0, 0.95, 0.49]}>
        <boxGeometry args={[0.35, 0.1, 0.005]} />
        <meshStandardMaterial color="#b8860b" metalness={0.95} roughness={0.08} />
      </mesh>

      {/* ══════ HEAT GLOW ══════ */}
      <pointLight ref={glowRef} position={[0, 0.65, 0]} color="#ff8c42" distance={3} intensity={0} />
    </group>
  );
}
