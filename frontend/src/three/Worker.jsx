import { useRef, useMemo } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";

/* ─────────────────────────────────────────────────────────
   WORKER — Stylised low-poly factory worker
   Head, hard hat, body, arms (swinging), legs (walking).
   Follows a closed-loop path of waypoints at a given speed.
   ───────────────────────────────────────────────────────── */
export default function Worker({
  path = [[0, 0, 0], [3, 0, 0]],
  speed = 0.6,
  color = "#3b82f6",
  hatColor = "#eab308",
  startOffset = 0,
}) {
  const groupRef = useRef();
  const leftArmRef = useRef();
  const rightArmRef = useRef();
  const leftLegRef = useRef();
  const rightLegRef = useRef();

  /* build a smooth path from waypoints */
  const curve = useMemo(() => {
    const pts = path.map((p) => new THREE.Vector3(p[0], 0, p[2] ?? p[1]));
    return new THREE.CatmullRomCurve3(pts, true, "catmullrom", 0.5);
  }, [path]);

  const totalLen = useMemo(() => curve.getLength(), [curve]);

  useFrame(({ clock }) => {
    if (!groupRef.current) return;
    const t = ((clock.elapsedTime * speed + startOffset * totalLen) / totalLen) % 1;

    /* position on path */
    const pos = curve.getPointAt(t);
    groupRef.current.position.set(pos.x, -0.35, pos.z);

    /* face direction of travel */
    const ahead = curve.getPointAt((t + 0.01) % 1);
    const angle = Math.atan2(ahead.x - pos.x, ahead.z - pos.z);
    groupRef.current.rotation.y = angle;

    /* arm & leg swing */
    const swing = Math.sin(clock.elapsedTime * speed * 8) * 0.4;
    if (leftArmRef.current) leftArmRef.current.rotation.x = swing;
    if (rightArmRef.current) rightArmRef.current.rotation.x = -swing;
    if (leftLegRef.current) leftLegRef.current.rotation.x = -swing;
    if (rightLegRef.current) rightLegRef.current.rotation.x = swing;
  });

  return (
    <group ref={groupRef}>
      {/* ── HEAD ── */}
      <mesh position={[0, 1.55, 0]}>
        <sphereGeometry args={[0.1, 12, 10]} />
        <meshStandardMaterial color="#f5d0a9" roughness={0.8} />
      </mesh>

      {/* ── HARD HAT ── */}
      <mesh position={[0, 1.65, 0]}>
        <cylinderGeometry args={[0.12, 0.11, 0.06, 12]} />
        <meshStandardMaterial color={hatColor} roughness={0.5} metalness={0.2} />
      </mesh>
      {/* hat brim */}
      <mesh position={[0, 1.62, 0]}>
        <cylinderGeometry args={[0.15, 0.15, 0.015, 12]} />
        <meshStandardMaterial color={hatColor} roughness={0.5} metalness={0.2} />
      </mesh>

      {/* ── TORSO ── */}
      <mesh position={[0, 1.25, 0]}>
        <boxGeometry args={[0.22, 0.35, 0.12]} />
        <meshStandardMaterial color={color} roughness={0.7} />
      </mesh>

      {/* ── BELT ── */}
      <mesh position={[0, 1.06, 0]}>
        <boxGeometry args={[0.23, 0.03, 0.13]} />
        <meshStandardMaterial color="#1c1917" roughness={0.6} metalness={0.3} />
      </mesh>

      {/* ── LEFT ARM ── */}
      <group position={[-0.15, 1.35, 0]} ref={leftArmRef}>
        <mesh position={[0, -0.15, 0]}>
          <boxGeometry args={[0.06, 0.3, 0.06]} />
          <meshStandardMaterial color={color} roughness={0.7} />
        </mesh>
        {/* hand */}
        <mesh position={[0, -0.32, 0]}>
          <sphereGeometry args={[0.035, 8, 6]} />
          <meshStandardMaterial color="#f5d0a9" roughness={0.8} />
        </mesh>
      </group>

      {/* ── RIGHT ARM ── */}
      <group position={[0.15, 1.35, 0]} ref={rightArmRef}>
        <mesh position={[0, -0.15, 0]}>
          <boxGeometry args={[0.06, 0.3, 0.06]} />
          <meshStandardMaterial color={color} roughness={0.7} />
        </mesh>
        <mesh position={[0, -0.32, 0]}>
          <sphereGeometry args={[0.035, 8, 6]} />
          <meshStandardMaterial color="#f5d0a9" roughness={0.8} />
        </mesh>
      </group>

      {/* ── LEFT LEG ── */}
      <group position={[-0.06, 1.03, 0]} ref={leftLegRef}>
        <mesh position={[0, -0.17, 0]}>
          <boxGeometry args={[0.07, 0.35, 0.07]} />
          <meshStandardMaterial color="#374151" roughness={0.8} />
        </mesh>
        {/* boot */}
        <mesh position={[0, -0.36, 0.015]}>
          <boxGeometry args={[0.08, 0.06, 0.1]} />
          <meshStandardMaterial color="#1c1917" roughness={0.7} metalness={0.2} />
        </mesh>
      </group>

      {/* ── RIGHT LEG ── */}
      <group position={[0.06, 1.03, 0]} ref={rightLegRef}>
        <mesh position={[0, -0.17, 0]}>
          <boxGeometry args={[0.07, 0.35, 0.07]} />
          <meshStandardMaterial color="#374151" roughness={0.8} />
        </mesh>
        <mesh position={[0, -0.36, 0.015]}>
          <boxGeometry args={[0.08, 0.06, 0.1]} />
          <meshStandardMaterial color="#1c1917" roughness={0.7} metalness={0.2} />
        </mesh>
      </group>
    </group>
  );
}
