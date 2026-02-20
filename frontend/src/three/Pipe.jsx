import { useRef, useMemo } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";

export default function Pipe({ from = [-1.5, 1, 0], to = [3.5, 1, 0], flowSpeed = 1, pressureHigh = false, segments = 3 }) {
  const groupRef = useRef();
  const dotsRef = useRef([]);

  const direction = useMemo(() => {
    const d = new THREE.Vector3(...to).sub(new THREE.Vector3(...from));
    return d;
  }, [from, to]);

  const length = useMemo(() => direction.length(), [direction]);
  const mid = useMemo(() => new THREE.Vector3(...from).add(direction.clone().multiplyScalar(0.5)), [from, direction]);
  const angle = useMemo(() => Math.atan2(direction.y, direction.x), [direction]);

  const pipeColor = pressureHigh ? "#ef4444" : "#475569";
  const dotColor = pressureHigh ? "#fca5a5" : "#60a5fa";

  useFrame(({ clock }) => {
    const t = clock.elapsedTime;
    dotsRef.current.forEach((dot, i) => {
      if (!dot) return;
      const phase = ((t * flowSpeed * 0.5 + i / segments) % 1);
      const pos = new THREE.Vector3(...from).add(direction.clone().multiplyScalar(phase));
      dot.position.set(pos.x, pos.y, pos.z);
      dot.scale.setScalar(0.6 + Math.sin(phase * Math.PI) * 0.4);
    });
  });

  return (
    <group ref={groupRef}>
      {/* Main pipe */}
      <mesh position={[mid.x, mid.y, mid.z]} rotation={[0, 0, angle]}>
        <cylinderGeometry args={[0.12, 0.12, length, 16]} rotation={[0, 0, Math.PI / 2]} />
        <meshStandardMaterial color={pipeColor} metalness={0.8} roughness={0.2} />
      </mesh>

      {/* Pipe ends (flanges) */}
      {[from, to].map((pos, i) => (
        <mesh key={i} position={pos}>
          <cylinderGeometry args={[0.2, 0.2, 0.15, 16]} rotation={[0, 0, Math.PI / 2]} />
          <meshStandardMaterial color="#334155" metalness={0.9} roughness={0.1} />
        </mesh>
      ))}

      {/* Flow indicator dots */}
      {Array.from({ length: segments }).map((_, i) => (
        <mesh key={i} ref={(el) => (dotsRef.current[i] = el)}>
          <sphereGeometry args={[0.06, 8, 8]} />
          <meshStandardMaterial
            color={dotColor}
            emissive={dotColor}
            emissiveIntensity={0.8}
          />
        </mesh>
      ))}
    </group>
  );
}
