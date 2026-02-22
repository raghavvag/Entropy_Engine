import * as THREE from "three";

/* ─────────────────────────────────────────────────────────
   FLOOR — Industrial Factory Floor
   Concrete slab with expansion joints, yellow/black hazard
   stripe perimeter border, raised equipment pads under
   furnace and turbine, and a central drain grate.
   ───────────────────────────────────────────────────────── */
export default function Floor() {
  const stripeColor = "#eab308";
  const concreteColor = "#1e293b";

  return (
    <group>
      {/* ══════ MAIN CONCRETE SLAB ══════ */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0.5, -0.35, 0]} receiveShadow>
        <planeGeometry args={[24, 18]} />
        <meshStandardMaterial color={concreteColor} metalness={0.15} roughness={0.88} />
      </mesh>

      {/* subtle grid */}
      <gridHelper args={[24, 48, "#1a2236", "#1a2236"]} position={[0.5, -0.34, 0]} />

      {/* ══════ EXPANSION JOINTS (dark lines) ══════ */}
      {[-3, 0, 3, 6].map((x) => (
        <mesh key={`ej-x-${x}`} rotation={[-Math.PI / 2, 0, 0]} position={[x, -0.338, 0]}>
          <planeGeometry args={[0.025, 18]} />
          <meshBasicMaterial color="#0b1120" />
        </mesh>
      ))}
      {[-4, 0, 4].map((z) => (
        <mesh key={`ej-z-${z}`} rotation={[-Math.PI / 2, 0, Math.PI / 2]} position={[0.5, -0.338, z]}>
          <planeGeometry args={[0.025, 24]} />
          <meshBasicMaterial color="#0b1120" />
        </mesh>
      ))}

      {/* ══════ HAZARD STRIPE BORDER (yellow/black) ══════ */}
      {/* front */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0.5, -0.335, 8.85]}>
        <planeGeometry args={[23.8, 0.25]} />
        <meshStandardMaterial color={stripeColor} roughness={0.7} metalness={0.1} />
      </mesh>
      {/* back */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0.5, -0.335, -8.85]}>
        <planeGeometry args={[23.8, 0.25]} />
        <meshStandardMaterial color={stripeColor} roughness={0.7} metalness={0.1} />
      </mesh>
      {/* left */}
      <mesh rotation={[-Math.PI / 2, 0, Math.PI / 2]} position={[-11.25, -0.335, 0]}>
        <planeGeometry args={[17.5, 0.25]} />
        <meshStandardMaterial color={stripeColor} roughness={0.7} metalness={0.1} />
      </mesh>
      {/* right */}
      <mesh rotation={[-Math.PI / 2, 0, Math.PI / 2]} position={[12.25, -0.335, 0]}>
        <planeGeometry args={[17.5, 0.25]} />
        <meshStandardMaterial color={stripeColor} roughness={0.7} metalness={0.1} />
      </mesh>
      {/* black dashes on the yellow stripes */}
      {Array.from({ length: 30 }).map((_, i) => (
        <mesh key={`hd-${i}`} rotation={[-Math.PI / 2, 0, Math.PI / 4]} position={[-11 + i * 0.8, -0.333, 8.85]}>
          <planeGeometry args={[0.12, 0.35]} />
          <meshBasicMaterial color="#111827" />
        </mesh>
      ))}

      {/* ══════ EQUIPMENT PADS (slightly raised) ══════ */}
      {/* furnace pad */}
      <mesh position={[-4.5, -0.32, 0]}>
        <boxGeometry args={[4, 0.04, 3.5]} />
        <meshStandardMaterial color="#263040" roughness={0.85} metalness={0.2} />
      </mesh>
      {/* turbine pad */}
      <mesh position={[5, -0.32, 0]}>
        <boxGeometry args={[6, 0.04, 3.2]} />
        <meshStandardMaterial color="#263040" roughness={0.85} metalness={0.2} />
      </mesh>

      {/* ══════ DRAIN GRATE (center) ══════ */}
      <mesh position={[0.5, -0.33, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[0.6, 0.6]} />
        <meshStandardMaterial color="#111827" roughness={0.9} metalness={0.3} />
      </mesh>
      {/* grate bars */}
      {[-0.2, -0.1, 0, 0.1, 0.2].map((x) => (
        <mesh key={`gb-${x}`} position={[0.5 + x, -0.325, 0]} rotation={[-Math.PI / 2, 0, 0]}>
          <planeGeometry args={[0.02, 0.58]} />
          <meshStandardMaterial color="#374151" metalness={0.9} roughness={0.15} />
        </mesh>
      ))}
    </group>
  );
}
