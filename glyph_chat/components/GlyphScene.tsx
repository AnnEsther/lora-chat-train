'use client';

import { useRef, useCallback, useEffect } from 'react';
import { Canvas, useLoader, useFrame } from '@react-three/fiber';
import { OrbitControls, Environment } from '@react-three/drei';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import { MTLLoader } from 'three/examples/jsm/loaders/MTLLoader.js';
import * as THREE from 'three';
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib';

const SCALE = 0.45;

// Prefix asset paths with basePath so they resolve correctly under /glyph in production
const BASE = process.env.NEXT_PUBLIC_BASE_PATH ?? '';

interface GlyphModelProps {
  onSpun: () => void;
  chatOpen: boolean;
}

// Minimum azimuthal angle change (radians) per frame to count as "still moving"
const STILL_THRESHOLD = 0.0002;
// Number of consecutive still frames before we consider rotation finished
const STILL_FRAMES_REQUIRED = 8;

function GlyphModel({ onSpun, chatOpen }: GlyphModelProps) {
  // didSpin: model was rotated during the current drag gesture
  const didSpin = useRef(false);
  // pointerDown: user is actively holding the mouse/touch down
  const pointerDown = useRef(false);
  // waitingForStop: pointer released, now waiting for damping to settle
  const waitingForStop = useRef(false);
  // hasTriggered: onSpun already fired for this open/close cycle — don't fire again
  const hasTriggered = useRef(false);
  // Track azimuthal angle between frames to detect when rotation stops
  const lastAzimuth = useRef<number | null>(null);
  const stillFrameCount = useRef(0);
  const controlsRef = useRef<OrbitControlsImpl>(null);
  const groupRef = useRef<THREE.Group>(null);

  const materials = useLoader(MTLLoader, `${BASE}/assets/Glyph_B_Baking.mtl`);
  const obj = useLoader(OBJLoader, `${BASE}/assets/Glyph_B_Baking.obj`, (loader) => {
    materials.preload();
    (loader as OBJLoader).setMaterials(materials);
  });

  useEffect(() => {
    if (!obj) return;

    obj.scale.setScalar(SCALE);

    const box = new THREE.Box3().setFromObject(obj);
    const center = new THREE.Vector3();
    box.getCenter(center);
    obj.position.set(-center.x, -center.y, -center.z);

    const pinkMaterial = new THREE.MeshPhysicalMaterial({
      color: '#823a88',
      emissive: '#5a1f60',
      emissiveIntensity: 0.15,
      transparent: true,
      opacity: 0.6,
      roughness: 0.12,             // wider specular lobe — no collapsed white spike
      metalness: 0.0,
      reflectivity: 0.5,           // 50% as requested
      specularColor: new THREE.Color('#b060dd'), // tint specular highlights violet
      specularIntensity: 0.6,      // moderate specular so iridescence wins at most angles
      envMapIntensity: 2.0,
      ior: 1.5,
      thickness: 0.5,
      iridescence: 1.0,
      iridescenceIOR: 1.8,
      iridescenceThicknessRange: [200, 800],
      side: THREE.DoubleSide,
    });

    obj.traverse((child) => {
      if ((child as THREE.Mesh).isMesh) {
        (child as THREE.Mesh).material = pinkMaterial;
      }
    });
  }, [obj]);

  // When the modal closes, reset all gates so the next drag can reopen it
  useEffect(() => {
    if (!chatOpen) {
      didSpin.current = false;
      pointerDown.current = false;
      waitingForStop.current = false;
      hasTriggered.current = false;
      lastAzimuth.current = null;
      stillFrameCount.current = 0;
    }
  }, [chatOpen]);

  // OrbitControls "start" — pointer is now down
  const handleStart = useCallback(() => {
    pointerDown.current = true;
    // If user grabs again mid-coast, cancel pending trigger
    waitingForStop.current = false;
    stillFrameCount.current = 0;
  }, []);

  // During drag — record that rotation happened
  const handleChange = useCallback(() => {
    if (pointerDown.current) {
      didSpin.current = true;
    }
  }, []);

  // OrbitControls "end" — pointer released, start watching for coast to settle
  const handleEnd = useCallback(() => {
    pointerDown.current = false;
    if (didSpin.current && !hasTriggered.current) {
      waitingForStop.current = true;
      stillFrameCount.current = 0;
      lastAzimuth.current = null;
    }
  }, []);

  // Each frame: if waiting for stop, check if azimuthal angle has settled
  useFrame(() => {
    if (!waitingForStop.current || hasTriggered.current) return;
    const controls = controlsRef.current;
    if (!controls) return;

    const azimuth = controls.getAzimuthalAngle();

    if (lastAzimuth.current === null) {
      lastAzimuth.current = azimuth;
      return;
    }

    const delta = Math.abs(azimuth - lastAzimuth.current);
    lastAzimuth.current = azimuth;

    if (delta < STILL_THRESHOLD) {
      stillFrameCount.current += 1;
    } else {
      stillFrameCount.current = 0;
    }

    if (stillFrameCount.current >= STILL_FRAMES_REQUIRED) {
      waitingForStop.current = false;
      hasTriggered.current = true;
      didSpin.current = false;
      onSpun();
    }
  });

  return (
    <>
      <group ref={groupRef}>
        <primitive object={obj} />
      </group>
      <OrbitControls
        ref={controlsRef}
        enableZoom={false}
        enablePan={false}
        minPolarAngle={Math.PI / 2}
        maxPolarAngle={Math.PI / 2}
        onStart={handleStart}
        onChange={handleChange}
        onEnd={handleEnd}
      />
    </>
  );
}

interface GlyphSceneProps {
  onSpun: () => void;
  chatOpen: boolean;
}

export default function GlyphScene({ onSpun, chatOpen }: GlyphSceneProps) {
  return (
    <Canvas
      gl={{ alpha: true, antialias: true }}
      style={{ width: '100vw', height: '100vh', background: '#0a0a0a' }}
      camera={{ position: [0, 0, 12], fov: 45 }}
    >
      {/* Ambient — soft base, no specular contribution */}
      <ambientLight intensity={0.7} />

      {/* Rim light — violet feeds iridescence on edges */}
      <pointLight position={[-5, 2, -5]} intensity={2.5} color="#8833ff" />

      {/* Fill — low-intensity pink, shapes the form without white spikes */}
      <pointLight position={[4, -3, 4]} intensity={0.5} color="#cc66aa" />

      {/* sunset HDR has warm/coloured highlights — no bright white panels */}
      <Environment preset="sunset" />

      <GlyphModel onSpun={onSpun} chatOpen={chatOpen} />
    </Canvas>
  );
}
