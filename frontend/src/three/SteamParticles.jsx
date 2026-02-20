import { useRef, useMemo } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";

export default function SteamParticles({ count = 60, speed = 1, position = [-3, 3.4, -0.5] }) {
  const meshRef = useRef();

  const particles = useMemo(() => {
    const arr = [];
    for (let i = 0; i < count; i++) {
      arr.push({
        offset: Math.random() * 10,
        x: (Math.random() - 0.5) * 0.5,
        z: (Math.random() - 0.5) * 0.5,
        speed: 0.3 + Math.random() * 0.7,
        scale: 0.03 + Math.random() * 0.05,
      });
    }
    return arr;
  }, [count]);

  const dummy = useMemo(() => new THREE.Object3D(), []);

  useFrame(({ clock }) => {
    const t = clock.elapsedTime;
    particles.forEach((p, i) => {
      const life = ((t * p.speed * speed + p.offset) % 3) / 3; // 0 → 1
      const y = life * 3;                     // rise 3 units
      const drift = Math.sin(t + p.offset) * 0.3 * life; // horizontal drift
      const s = p.scale * (1 - life * 0.7);   // shrink as it rises
      const opacity = Math.sin(life * Math.PI); // fade in/out

      dummy.position.set(
        position[0] + p.x + drift,
        position[1] + y,
        position[2] + p.z,
      );
      dummy.scale.setScalar(s * (0.5 + opacity * 0.5));
      dummy.updateMatrix();
      meshRef.current.setMatrixAt(i, dummy.matrix);
    });
    meshRef.current.instanceMatrix.needsUpdate = true;
  });

  return (
    <instancedMesh ref={meshRef} args={[null, null, count]}>
      <sphereGeometry args={[1, 6, 6]} />
      <meshStandardMaterial
        color="#94a3b8"
        transparent
        opacity={0.25}
        depthWrite={false}
      />
    </instancedMesh>
  );
}
