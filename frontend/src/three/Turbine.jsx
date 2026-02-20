import { useRef, useMemo } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";

export default function Turbine({ position = [5, 1, 0], power = 150, aiActive = false }) {
  const bladeRef = useRef();
  const glowRef = useRef();
  const speed = Math.max(0.5, power / 80);

  useFrame((_, delta) => {
    if (bladeRef.current) {
      bladeRef.current.rotation.z += delta * speed * 2;
    }
    if (glowRef.current && aiActive) {
      glowRef.current.intensity = 1.5 + Math.sin(Date.now() * 0.003) * 0.5;
    }
  });

  return (
    <group position={position}>
      {/* Base */}
      <mesh position={[0, -0.5, 0]}>
        <boxGeometry args={[2.2, 0.3, 2.2]} />
        <meshStandardMaterial color="#1e293b" metalness={0.8} roughness={0.3} />
      </mesh>

      {/* Housing */}
      <mesh position={[0, 0, 0]} rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[1.3, 1.3, 0.6, 32]} />
        <meshStandardMaterial color="#374151" metalness={0.85} roughness={0.15} />
      </mesh>

      {/* Inner ring */}
      <mesh position={[0, 0, 0.31]} rotation={[0, 0, 0]}>
        <torusGeometry args={[1.0, 0.08, 8, 32]} />
        <meshStandardMaterial
          color={aiActive ? "#3b82f6" : "#475569"}
          emissive={aiActive ? new THREE.Color(0.2, 0.4, 1) : new THREE.Color(0, 0, 0)}
          emissiveIntensity={aiActive ? 1 : 0}
          metalness={0.9}
          roughness={0.1}
        />
      </mesh>

      {/* Spinning blades */}
      <group ref={bladeRef} position={[0, 0, 0.32]}>
        {[0, 45, 90, 135, 180, 225, 270, 315].map((angle) => (
          <mesh key={angle} rotation={[0, 0, (angle * Math.PI) / 180]}>
            <boxGeometry args={[0.1, 0.9, 0.04]} />
            <meshStandardMaterial
              color={aiActive ? "#60a5fa" : "#64748b"}
              metalness={0.7}
              roughness={0.3}
            />
          </mesh>
        ))}
        {/* Hub */}
        <mesh>
          <cylinderGeometry args={[0.15, 0.15, 0.1, 16]} rotation={[Math.PI / 2, 0, 0]} />
          <meshStandardMaterial color="#94a3b8" metalness={0.9} roughness={0.1} />
        </mesh>
      </group>

      {/* AI glow */}
      {aiActive && (
        <pointLight
          ref={glowRef}
          position={[0, 0, 1]}
          color="#3b82f6"
          distance={5}
          intensity={1.5}
        />
      )}
    </group>
  );
}
