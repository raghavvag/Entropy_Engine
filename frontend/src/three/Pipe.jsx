import { useRef, useMemo } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";

/* ─────────────────────────────────────────────────────────
   PIPE — Industrial Piping Segment
   Straight run between two 3-D points with flanged ends,
   pipe-support brackets on stands, a gate valve with
   handwheel at the midpoint, and animated flow-indicator
   particles.  Pressure-reactive colouring.
   ───────────────────────────────────────────────────────── */
export default function Pipe({
  from = [-1.5, 1, 0],
  to = [3.5, 1, 0],
  flowSpeed = 1,
  pressureHigh = false,
  segments = 4,
  showValve = false,
  radius = 0.1,
}) {
  const dotsRef = useRef([]);
  const valveWheelRef = useRef();

  const dir = useMemo(() => new THREE.Vector3(...to).sub(new THREE.Vector3(...from)), [from, to]);
  const len = useMemo(() => dir.length(), [dir]);
  const mid = useMemo(() => new THREE.Vector3(...from).add(dir.clone().multiplyScalar(0.5)), [from, dir]);
  const angle = useMemo(() => Math.atan2(dir.y, dir.x), [dir]);

  const pipeColor = pressureHigh ? "#dc2626" : "#6b7280";
  const flangeColor = pressureHigh ? "#991b1b" : "#374151";
  const dotColor = pressureHigh ? "#fca5a5" : "#60a5fa";

  /* slowly spin the handwheel */
  useFrame(({ clock }) => {
    const t = clock.elapsedTime;
    if (valveWheelRef.current) valveWheelRef.current.rotation.x = Math.sin(t * 0.5) * 0.15;
    dotsRef.current.forEach((dot, i) => {
      if (!dot) return;
      const phase = ((t * flowSpeed * 0.4 + i / segments) % 1);
      const pos = new THREE.Vector3(...from).add(dir.clone().multiplyScalar(phase));
      dot.position.set(pos.x, pos.y, pos.z);
      dot.scale.setScalar(0.5 + Math.sin(phase * Math.PI) * 0.5);
    });
  });

  /* how many support brackets to place */
  const supportCount = Math.max(0, Math.floor(len / 2.5));

  return (
    <group>
      {/* ══════ MAIN PIPE BODY ══════ */}
      <mesh position={[mid.x, mid.y, mid.z]} rotation={[0, 0, angle]}>
        <cylinderGeometry args={[radius, radius, len, 16]} rotation={[0, 0, Math.PI / 2]} />
        <meshStandardMaterial color={pipeColor} metalness={0.82} roughness={0.18} />
      </mesh>

      {/* thin highlight line (specular strip) */}
      <mesh position={[mid.x, mid.y + radius * 0.92, mid.z]} rotation={[0, 0, angle]}>
        <cylinderGeometry args={[0.008, 0.008, len * 0.96, 6]} rotation={[0, 0, Math.PI / 2]} />
        <meshStandardMaterial color="#d1d5db" metalness={0.95} roughness={0.05} />
      </mesh>

      {/* ══════ FLANGED ENDS ══════ */}
      {[from, to].map((pos, i) => (
        <group key={`flange-${i}`} position={pos}>
          {/* flange disc */}
          <mesh rotation={[0, 0, angle]}>
            <cylinderGeometry args={[radius * 1.65, radius * 1.65, 0.06, 16]} rotation={[0, 0, Math.PI / 2]} />
            <meshStandardMaterial color={flangeColor} metalness={0.92} roughness={0.1} />
          </mesh>
          {/* bolts (4 per flange) */}
          {[0, 90, 180, 270].map((deg) => {
            const r = (deg * Math.PI) / 180;
            return (
              <mesh key={`fb-${i}-${deg}`} position={[0, Math.sin(r) * radius * 1.35, Math.cos(r) * radius * 1.35]}>
                <sphereGeometry args={[0.018, 6, 6]} />
                <meshStandardMaterial color="#9ca3af" metalness={0.95} roughness={0.08} />
              </mesh>
            );
          })}
        </group>
      ))}

      {/* ══════ PIPE SUPPORTS ══════ */}
      {Array.from({ length: supportCount }).map((_, i) => {
        const frac = (i + 1) / (supportCount + 1);
        const sp = new THREE.Vector3(...from).add(dir.clone().multiplyScalar(frac));
        return (
          <group key={`support-${i}`} position={[sp.x, sp.y, sp.z]}>
            {/* vertical stand */}
            <mesh position={[0, -(sp.y + 0.15) / 2, 0]}>
              <boxGeometry args={[0.06, sp.y + 0.15, 0.06]} />
              <meshStandardMaterial color="#374151" metalness={0.88} roughness={0.2} />
            </mesh>
            {/* U-clamp */}
            <mesh>
              <torusGeometry args={[radius * 1.3, 0.015, 8, 12, Math.PI]} />
              <meshStandardMaterial color="#4b5563" metalness={0.9} roughness={0.12} />
            </mesh>
            {/* base plate */}
            <mesh position={[0, -(sp.y + 0.13), 0]}>
              <boxGeometry args={[0.18, 0.03, 0.18]} />
              <meshStandardMaterial color="#1f2937" metalness={0.85} roughness={0.25} />
            </mesh>
          </group>
        );
      })}

      {/* ══════ GATE VALVE (optional, at midpoint) ══════ */}
      {showValve && (
        <group position={[mid.x, mid.y, mid.z]}>
          {/* valve body */}
          <mesh>
            <boxGeometry args={[0.2, 0.28, 0.2]} />
            <meshStandardMaterial color="#374151" metalness={0.88} roughness={0.18} />
          </mesh>
          {/* bonnet (stem housing) */}
          <mesh position={[0, 0.22, 0]}>
            <cylinderGeometry args={[0.04, 0.055, 0.2, 10]} />
            <meshStandardMaterial color="#4b5563" metalness={0.85} roughness={0.2} />
          </mesh>
          {/* handwheel */}
          <group position={[0, 0.38, 0]} ref={valveWheelRef}>
            <mesh>
              <torusGeometry args={[0.1, 0.015, 8, 16]} />
              <meshStandardMaterial color="#1c1917" metalness={0.9} roughness={0.15} />
            </mesh>
            {/* spokes */}
            {[0, 60, 120].map((deg) => (
              <mesh key={`sp-${deg}`} rotation={[0, 0, (deg * Math.PI) / 180]}>
                <boxGeometry args={[0.01, 0.19, 0.01]} />
                <meshStandardMaterial color="#1c1917" metalness={0.9} roughness={0.15} />
              </mesh>
            ))}
          </group>
        </group>
      )}

      {/* ══════ FLOW INDICATOR PARTICLES ══════ */}
      {Array.from({ length: segments }).map((_, i) => (
        <mesh key={`dot-${i}`} ref={(el) => (dotsRef.current[i] = el)}>
          <sphereGeometry args={[0.04, 8, 8]} />
          <meshStandardMaterial
            color={dotColor} emissive={dotColor} emissiveIntensity={0.9}
            transparent opacity={0.85}
          />
        </mesh>
      ))}
    </group>
  );
}
