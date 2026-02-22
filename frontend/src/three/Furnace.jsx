import { useRef, useMemo } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";

/* ─────────────────────────────────────────────────────────
   FURNACE — Industrial Fire-Tube Boiler
   ~40 mesh elements: brick base, steel cladding, riveted
   bands, fire door w/ animated glow, sight glass, two-stage
   chimney with rain cap, pipe connection stubs, gauges,
   and status indicator light.
   ───────────────────────────────────────────────────────── */
export default function Furnace({ temperature = 500 }) {
  const fireRef = useRef();
  const glowRef = useRef();
  const sightRef = useRef();
  const indicatorRef = useRef();

  const t = Math.max(0, Math.min(1, (temperature - 380) / 220));

  const fireColor = useMemo(
    () => new THREE.Color().setHSL(0.08 - t * 0.06, 0.95, 0.35 + t * 0.25),
    [t],
  );

  useFrame(({ clock }) => {
    const time = clock.elapsedTime;
    if (glowRef.current) {
      const f = Math.sin(time * 7.3) * 0.15 + Math.sin(time * 13.7) * 0.1 + 0.75;
      glowRef.current.intensity = t * 5 * f;
    }
    if (fireRef.current) {
      const f = Math.sin(time * 5.7) * 0.2 + Math.sin(time * 11.3) * 0.15 + 0.65;
      fireRef.current.emissiveIntensity = t * 2.5 * f;
    }
    if (sightRef.current)
      sightRef.current.emissiveIntensity = t * 2.5 + Math.sin(clock.elapsedTime * 4.2) * 0.4;
    if (indicatorRef.current)
      indicatorRef.current.emissiveIntensity = 1.5 + Math.sin(clock.elapsedTime * 2) * 0.5;
  });

  const indicatorColor = t > 0.75 ? "#ef4444" : t > 0.4 ? "#f59e0b" : "#22c55e";

  return (
    <group position={[-4.5, 0, 0]}>
      {/* ══════ FOUNDATION ══════ */}
      <mesh position={[0, -0.28, 0]} receiveShadow>
        <boxGeometry args={[3.8, 0.12, 3.2]} />
        <meshStandardMaterial color="#374151" roughness={0.92} metalness={0.1} />
      </mesh>
      <mesh position={[0, -0.12, 0]}>
        <boxGeometry args={[3.4, 0.2, 2.8]} />
        <meshStandardMaterial color="#1f2937" metalness={0.85} roughness={0.25} />
      </mesh>

      {/* ══════ SUPPORT LEGS (6) ══════ */}
      {[
        [-1.2, 0, 1.05], [0, 0, 1.05], [1.2, 0, 1.05],
        [-1.2, 0, -1.05], [0, 0, -1.05], [1.2, 0, -1.05],
      ].map(([x, , z], i) => (
        <group key={`leg-${i}`}>
          <mesh position={[x, 0.15, z]}>
            <boxGeometry args={[0.12, 0.35, 0.12]} />
            <meshStandardMaterial color="#1f2937" metalness={0.9} roughness={0.2} />
          </mesh>
          {/* gusset */}
          <mesh position={[x, 0.25, z + (i < 3 ? -0.09 : 0.09)]}>
            <boxGeometry args={[0.1, 0.12, 0.025]} />
            <meshStandardMaterial color="#1f2937" metalness={0.85} roughness={0.25} />
          </mesh>
        </group>
      ))}

      {/* ══════ LOWER BODY — Refractory Brick ══════ */}
      <mesh position={[0, 0.72, 0]} castShadow>
        <boxGeometry args={[2.8, 1.05, 2.3]} />
        <meshStandardMaterial
          color="#92400e" roughness={0.92} metalness={0.05}
          emissive={fireColor} emissiveIntensity={t * 0.25}
        />
      </mesh>
      {/* mortar course lines */}
      {[0.38, 0.58, 0.78, 0.98].map((y) => (
        <mesh key={`bl-${y}`} position={[0, y, 1.151]}>
          <boxGeometry args={[2.82, 0.012, 0.004]} />
          <meshBasicMaterial color="#78350f" />
        </mesh>
      ))}
      {/* vertical joints */}
      {[-1.0, -0.5, 0, 0.5, 1.0].map((x) => (
        <mesh key={`bv-${x}`} position={[x, 0.72, 1.151]}>
          <boxGeometry args={[0.009, 1.06, 0.004]} />
          <meshBasicMaterial color="#78350f" />
        </mesh>
      ))}

      {/* ══════ UPPER BODY — Steel Cladding ══════ */}
      <mesh position={[0, 1.65, 0]} castShadow>
        <boxGeometry args={[2.7, 0.95, 2.2]} />
        <meshStandardMaterial
          color="#4b5563" roughness={0.3} metalness={0.82}
          emissive={fireColor} emissiveIntensity={t * 0.1}
        />
      </mesh>
      {/* panel seam lines */}
      {[-0.9, 0, 0.9].map((x) => (
        <mesh key={`ps-${x}`} position={[x, 1.65, 1.101]}>
          <boxGeometry args={[0.018, 0.97, 0.004]} />
          <meshStandardMaterial color="#374151" metalness={0.9} roughness={0.15} />
        </mesh>
      ))}

      {/* ══════ RIVETED BANDS (3, wrapping all 4 sides) ══════ */}
      {[1.28, 1.65, 2.02].map((y) => (
        <group key={`band-${y}`}>
          {/* front + back strips */}
          {[1.106, -1.106].map((z) => (
            <mesh key={`strip-${y}-${z}`} position={[0, y, z]}>
              <boxGeometry args={[2.72, 0.05, 0.014]} />
              <meshStandardMaterial color="#374151" metalness={0.92} roughness={0.12} />
            </mesh>
          ))}
          {/* side strips */}
          {[-1.356, 1.356].map((x) => (
            <mesh key={`ss-${y}-${x}`} position={[x, y, 0]}>
              <boxGeometry args={[0.014, 0.05, 2.22]} />
              <meshStandardMaterial color="#374151" metalness={0.92} roughness={0.12} />
            </mesh>
          ))}
          {/* front rivets */}
          {[-1.2, -0.85, -0.5, -0.15, 0.2, 0.55, 0.9, 1.2].map((x) => (
            <mesh key={`rv-${y}-${x}`} position={[x, y, 1.12]}>
              <sphereGeometry args={[0.02, 8, 6]} />
              <meshStandardMaterial color="#9ca3af" metalness={0.95} roughness={0.08} />
            </mesh>
          ))}
        </group>
      ))}

      {/* ══════ FIRE DOOR (front, lower) ══════ */}
      <mesh position={[0, 0.65, 1.155]}>
        <boxGeometry args={[1.15, 0.8, 0.04]} />
        <meshStandardMaterial color="#1c1917" metalness={0.9} roughness={0.2} />
      </mesh>
      <mesh position={[0, 0.65, 1.14]} ref={fireRef}>
        <boxGeometry args={[0.95, 0.6, 0.02]} />
        <meshStandardMaterial
          color={fireColor} emissive={fireColor}
          emissiveIntensity={t * 2.5} roughness={0.5} metalness={0.1}
        />
      </mesh>
      {/* hinges */}
      {[0.35, 0.95].map((y) => (
        <mesh key={`hinge-${y}`} position={[-0.55, y, 1.17]}>
          <cylinderGeometry args={[0.022, 0.022, 0.08, 8]} />
          <meshStandardMaterial color="#292524" metalness={0.95} roughness={0.1} />
        </mesh>
      ))}
      {/* latch handle */}
      <group position={[0.48, 0.65, 1.18]}>
        <mesh><boxGeometry args={[0.035, 0.2, 0.035]} /><meshStandardMaterial color="#1c1917" metalness={0.9} roughness={0.15} /></mesh>
        <mesh position={[0, 0, 0.035]} rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[0.012, 0.012, 0.06, 8]} />
          <meshStandardMaterial color="#44403c" metalness={0.9} roughness={0.15} />
        </mesh>
      </group>

      {/* ══════ SIGHT GLASS (viewing port) ══════ */}
      <mesh position={[-0.7, 1.55, 1.106]}>
        <torusGeometry args={[0.13, 0.026, 12, 24]} />
        <meshStandardMaterial color="#292524" metalness={0.92} roughness={0.12} />
      </mesh>
      <mesh position={[-0.7, 1.55, 1.11]} ref={sightRef}>
        <circleGeometry args={[0.1, 24]} />
        <meshStandardMaterial
          color={fireColor} emissive={new THREE.Color(1, 0.35, 0.08)}
          emissiveIntensity={t * 2} transparent opacity={0.85} side={THREE.DoubleSide}
        />
      </mesh>

      {/* ══════ TWO-STAGE CHIMNEY ══════ */}
      <mesh position={[0.3, 2.15, -0.4]}>
        <cylinderGeometry args={[0.46, 0.46, 0.07, 16]} />
        <meshStandardMaterial color="#1f2937" metalness={0.9} roughness={0.15} />
      </mesh>
      <mesh position={[0.3, 2.65, -0.4]}>
        <cylinderGeometry args={[0.35, 0.4, 0.95, 16]} />
        <meshStandardMaterial color="#4b5563" metalness={0.78} roughness={0.28} />
      </mesh>
      <mesh position={[0.3, 3.15, -0.4]}>
        <cylinderGeometry args={[0.4, 0.4, 0.06, 16]} />
        <meshStandardMaterial color="#374151" metalness={0.9} roughness={0.15} />
      </mesh>
      <mesh position={[0.3, 3.65, -0.4]}>
        <cylinderGeometry args={[0.3, 0.35, 0.95, 16]} />
        <meshStandardMaterial color="#6b7280" metalness={0.7} roughness={0.35} />
      </mesh>
      {/* rain cap */}
      <mesh position={[0.3, 4.22, -0.4]}>
        <coneGeometry args={[0.45, 0.22, 16]} />
        <meshStandardMaterial color="#4b5563" metalness={0.8} roughness={0.2} />
      </mesh>
      {/* cap support rods */}
      {[0, 90, 180, 270].map((deg) => {
        const r = (deg * Math.PI) / 180;
        return (
          <mesh key={`cr-${deg}`} position={[0.3 + Math.cos(r) * 0.28, 4.18, -0.4 + Math.sin(r) * 0.28]}>
            <cylinderGeometry args={[0.011, 0.011, 0.25, 6]} />
            <meshStandardMaterial color="#9ca3af" metalness={0.85} roughness={0.2} />
          </mesh>
        );
      })}

      {/* ══════ PIPE CONNECTION STUBS ══════ */}
      {/* steam out (upper right) */}
      <group position={[1.4, 1.6, 0]}>
        <mesh rotation={[0, 0, Math.PI / 2]}>
          <cylinderGeometry args={[0.11, 0.11, 0.35, 12]} />
          <meshStandardMaterial color="#4b5563" metalness={0.85} roughness={0.22} />
        </mesh>
        <mesh position={[0.18, 0, 0]} rotation={[0, 0, Math.PI / 2]}>
          <cylinderGeometry args={[0.17, 0.17, 0.05, 12]} />
          <meshStandardMaterial color="#374151" metalness={0.92} roughness={0.12} />
        </mesh>
      </group>
      {/* feed water in (lower right) */}
      <group position={[1.4, 0.6, 0.4]}>
        <mesh rotation={[0, 0, Math.PI / 2]}>
          <cylinderGeometry args={[0.08, 0.08, 0.35, 12]} />
          <meshStandardMaterial color="#4b5563" metalness={0.85} roughness={0.22} />
        </mesh>
        <mesh position={[0.18, 0, 0]} rotation={[0, 0, Math.PI / 2]}>
          <cylinderGeometry args={[0.13, 0.13, 0.05, 12]} />
          <meshStandardMaterial color="#374151" metalness={0.92} roughness={0.12} />
        </mesh>
      </group>

      {/* ══════ GAUGE + INDICATOR ══════ */}
      <group position={[0.7, 1.75, 1.11]}>
        <mesh><cylinderGeometry args={[0.09, 0.09, 0.025, 16]} rotation={[Math.PI / 2, 0, 0]} /><meshStandardMaterial color="#1c1917" metalness={0.9} roughness={0.15} /></mesh>
        <mesh position={[0, 0, 0.018]}><circleGeometry args={[0.075, 20]} /><meshStandardMaterial color="#f5f5f4" roughness={0.8} side={THREE.DoubleSide} /></mesh>
      </group>
      <mesh position={[0.35, 2.1, 1.106]} ref={indicatorRef}>
        <sphereGeometry args={[0.032, 12, 8]} />
        <meshStandardMaterial color={indicatorColor} emissive={indicatorColor} emissiveIntensity={1.5} />
      </mesh>

      {/* ══════ NAMEPLATE ══════ */}
      <mesh position={[0, 2.0, 1.106]}>
        <boxGeometry args={[0.5, 0.15, 0.007]} />
        <meshStandardMaterial color="#b8860b" metalness={0.95} roughness={0.08} />
      </mesh>

      {/* ══════ LIGHTS ══════ */}
      <pointLight ref={glowRef} position={[0, 0.7, 2]} color={fireColor} distance={7} intensity={t * 4} />
      <pointLight position={[0, 3.5, -0.4]} color="#ff6b35" distance={3} intensity={t * 1} />
    </group>
  );
}
