import { useRef, useMemo } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";

export default function Furnace({ temperature = 500, glowIntensity = 0.5 }) {
  const glowRef = useRef();
  const bodyRef = useRef();
  const t = Math.max(0, Math.min(1, (temperature - 380) / 220));

  useFrame(({ clock }) => {
    if (glowRef.current) {
      const pulse = Math.sin(clock.elapsedTime * 2) * 0.1 + 0.9;
      glowRef.current.intensity = t * pulse * 4;
    }
    if (bodyRef.current) {
      bodyRef.current.emissiveIntensity = t * 1.5 + 0.1;
    }
  });

  const color = useMemo(() => {
    const r = Math.min(1, 0.8 + t * 0.2);
    const g = Math.max(0.05, 0.4 - t * 0.35);
    const b = Math.max(0.02, 0.1 - t * 0.08);
    return new THREE.Color(r, g, b);
  }, [t]);

  const emissive = useMemo(() => {
    return new THREE.Color(0.6 * t + 0.1, 0.15 * t, 0.02);
  }, [t]);

  return (
    <group position={[-3, 0, 0]}>
      {/* Base platform */}
      <mesh position={[0, -0.15, 0]}>
        <boxGeometry args={[3, 0.3, 2.5]} />
        <meshStandardMaterial color="#1e293b" metalness={0.8} roughness={0.3} />
      </mesh>

      {/* Main furnace body */}
      <mesh position={[0, 1.2, 0]}>
        <boxGeometry args={[2.4, 2.2, 2]} />
        <meshStandardMaterial
          ref={bodyRef}
          color={color}
          emissive={emissive}
          emissiveIntensity={t * 1.5}
          roughness={0.6}
          metalness={0.3}
        />
      </mesh>

      {/* Chimney */}
      <mesh position={[0, 2.8, -0.5]}>
        <cylinderGeometry args={[0.3, 0.4, 1.2, 16]} />
        <meshStandardMaterial color="#374151" metalness={0.7} roughness={0.4} />
      </mesh>

      {/* Furnace door (glowing) */}
      <mesh position={[0, 0.8, 1.01]}>
        <boxGeometry args={[1, 0.8, 0.05]} />
        <meshStandardMaterial
          color={new THREE.Color(1, 0.3 + t * 0.3, 0.05)}
          emissive={new THREE.Color(1, 0.3, 0)}
          emissiveIntensity={t * 3}
          roughness={0.3}
        />
      </mesh>

      {/* Point light (heat glow) */}
      <pointLight
        ref={glowRef}
        position={[0, 1.5, 1.5]}
        color={new THREE.Color(1, 0.4, 0.1)}
        distance={8}
        intensity={t * 3}
      />

      {/* Label */}
      <mesh position={[0, 2.6, 1.01]}>
        <planeGeometry args={[1.5, 0.3]} />
        <meshBasicMaterial color="#0f172a" opacity={0.6} transparent />
      </mesh>
    </group>
  );
}
