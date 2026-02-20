import * as THREE from "three";

export default function Floor() {
  return (
    <group>
      {/* Ground plane */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[1, -0.3, 0]} receiveShadow>
        <planeGeometry args={[30, 30]} />
        <meshStandardMaterial
          color="#0f172a"
          metalness={0.3}
          roughness={0.8}
        />
      </mesh>

      {/* Grid helper for industrial feel */}
      <gridHelper
        args={[30, 60, "#1e293b", "#1e293b"]}
        position={[1, -0.29, 0]}
      />
    </group>
  );
}
