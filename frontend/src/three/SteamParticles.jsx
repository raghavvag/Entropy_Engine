import { useRef, useMemo } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";

/* ─────────────────────────────────────────────────────────
   STEAM PARTICLES — Volumetric chimney exhaust
   InstancedMesh with per-particle lifecycle: rise, expand,
   drift with turbulence, fade in then out. Near the source
   particles are warm-tinted; higher up they turn grey.
   ───────────────────────────────────────────────────────── */
export default function SteamParticles({
  count = 80,
  speed = 1,
  position = [-4.2, 4.3, -0.4],
  spread = 0.35,
  riseHeight = 4,
}) {
  const meshRef = useRef();

  const particles = useMemo(() => {
    const arr = [];
    for (let i = 0; i < count; i++) {
      arr.push({
        offset: Math.random() * 12,
        x: (Math.random() - 0.5) * spread,
        z: (Math.random() - 0.5) * spread,
        speed: 0.25 + Math.random() * 0.75,
        baseScale: 0.04 + Math.random() * 0.06,
        turbPhase: Math.random() * Math.PI * 2,
        turbFreq: 1.5 + Math.random() * 1.5,
        turbAmp: 0.15 + Math.random() * 0.25,
      });
    }
    return arr;
  }, [count, spread]);

  const dummy = useMemo(() => new THREE.Object3D(), []);

  /* warm → grey colour ramp */
  const colorBase = useMemo(() => new THREE.Color("#b0b0b0"), []);

  useFrame(({ clock }) => {
    const t = clock.elapsedTime;
    particles.forEach((p, i) => {
      const life = ((t * p.speed * speed * 0.35 + p.offset) % 4) / 4; // 0 → 1
      const y = life * riseHeight;                                     // rise
      const driftX = Math.sin(t * p.turbFreq + p.turbPhase) * p.turbAmp * life;
      const driftZ = Math.cos(t * p.turbFreq * 0.7 + p.turbPhase) * p.turbAmp * 0.6 * life;

      /* scale: grow quickly then slowly shrink */
      const growPhase = Math.min(life * 4, 1);            // 0→1 in first 25 %
      const shrinkPhase = Math.max(0, 1 - (1 - life) * 3); // 0→1 in last 33 %
      const s = p.baseScale * (0.6 + growPhase * 1.8) * (1 - shrinkPhase * 0.5);

      /* opacity: smooth bell curve */
      const alpha = Math.sin(life * Math.PI) * 0.85;

      dummy.position.set(
        position[0] + p.x + driftX,
        position[1] + y,
        position[2] + p.z + driftZ,
      );
      dummy.scale.setScalar(Math.max(0.01, s * (0.3 + alpha * 0.7)));
      dummy.updateMatrix();
      meshRef.current.setMatrixAt(i, dummy.matrix);
    });
    meshRef.current.instanceMatrix.needsUpdate = true;
  });

  return (
    <instancedMesh ref={meshRef} args={[null, null, count]} frustumCulled={false}>
      <sphereGeometry args={[1, 8, 8]} />
      <meshStandardMaterial
        color={colorBase}
        transparent
        opacity={0.18}
        depthWrite={false}
        roughness={1}
        metalness={0}
      />
    </instancedMesh>
  );
}
