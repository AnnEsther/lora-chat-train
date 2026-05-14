# Glyph Scene

## Purpose
Renders the Glyph_B_Baking OBJ 3D model in a full-viewport Three.js canvas. Exposes an `onSpun` callback that fires the first time the user drags (rotates) the model, which is used to open the chat modal.

## Usage / API

```tsx
<GlyphScene onSpun={() => setChatOpen(true)} chatOpen={chatOpen} />
```

| Prop | Type | Description |
|------|------|-------------|
| `onSpun` | `() => void` | Called when the user finishes a drag (mouse released) and the model was rotated during that drag |
| `chatOpen` | `boolean` | When it transitions to `false` (modal closed), both spin gates reset so the next completed drag reopens the modal |

- Model files must exist at `public/assets/Glyph_B_Baking.obj` and `public/assets/Glyph_B_Baking.mtl`.
- Uses `@react-three/fiber` `<Canvas>` filling the viewport (`width: 100vw, height: 100vh`).
- Uses `@react-three/drei` `<OrbitControls>` for drag-to-rotate; `enableZoom` and `enablePan` are disabled. `minPolarAngle` and `maxPolarAngle` are both fixed at `Math.PI / 2` to allow only horizontal (Y-axis) rotation.
- `OBJLoader` + `MTLLoader` from `three/examples/jsm` load the model via HTTP from `/assets/`. The MTL has no image textures — all styling is applied via a custom `MeshStandardMaterial` after load.
- After loading, a `Box3` bounding box is computed and the model is repositioned to world origin (true centering, not a manual offset).
- Material: `MeshPhysicalMaterial` — pink, semi-transparent, reflective (`color: #ff80b0`, `opacity: 0.5`, `roughness: 0.08`, `reflectivity: 1.0`, `envMapIntensity: 2.5`, `ior: 1.5`, `side: DoubleSide`). `<Canvas gl={{ alpha: true }}>` required for transparency. `<Environment preset="studio">` provides the env map that the reflections sample from.
- Scale is applied via `obj.scale.setScalar(SCALE)` inside `useEffect` before the `Box3` computation so the bounding-box centering is accurate at render scale.
- Camera at `position: [0, 0, 12]`, model center zeroed to world origin → model midpoint is exactly at camera eye level (y=0).
- Lights: `ambientLight` (0.6) + `directionalLight` front-top (0.9, `#ffe8f0` pinkish-white) + `pointLight` rim-back (`#8833ff` violet, 2.5) + `pointLight` fill-bottom (`#e0ccff` soft violet, 0.4). All direct lights are tinted away from pure white to prevent blown-out white hotspots on the near-zero-roughness surface.

## Files

- `components/GlyphScene.tsx` — main component
- `public/assets/Glyph_B_Baking.obj` — geometry
- `public/assets/Glyph_B_Baking.mtl` — material

## Changelog

| Date | Change |
|------|--------|
| 2026-05-13 | Initial implementation |
| 2026-05-13 | Lock rotation to Y-axis only (minPolarAngle = maxPolarAngle = π/2); center model via Box3 bounding box; replace MTL material with emissive MeshStandardMaterial; improved 4-light rig |
| 2026-05-13 | Switch to MeshPhysicalMaterial for glassy pink transparency; apply scale before Box3 so centering is accurate; camera y=0 aligns with model center for true eye-level placement; pink rim light |
| 2026-05-13 | Material tuned: opacity=0.5, reflectivity=1.0, envMapIntensity=2.5, roughness=0.08; added drei Environment preset="studio" for real env-map reflections; removed transmission (was competing with reflectivity) |
| 2026-05-13 | Color changed to hot pink (#ff007f), opacity increased to 0.6 |
| 2026-05-13 | Color changed to #823a88 (deep purple-magenta), emissive updated to #5a1f60 |
| 2026-05-13 | Option D anti-blowout: roughness 0.05→0.12; reflectivity 1.0→0.5; specularColor=#b060dd, specularIntensity=0.6; removed directionalLight; env switched studio→sunset; fill light recoloured #cc66aa |
| 2026-05-13 | opacity 0.6→0.3 (50% reduction); fix chat not opening: added pointerDown ref tracked via onStart, handleChange only sets didSpin while pointer is down, preventing damping coast frames from cancelling waitingForStop |
| 2026-05-13 | Added iridescence (iridescence=1.0, iridescenceIOR=1.8, thicknessRange=[200,800]nm) for violet-to-rainbow shimmer; violet rim light (#8833ff) feeds iridescent edges |
| 2026-05-13 | Fix white hotspots: tinted all direct lights away from pure white (#ffe8f0 key, #e0ccff fill), reduced key intensity 1.5→0.9, fill intensity 0.8→0.4; fix modal not reopening: added chatOpen prop, hasSpun ref resets to false when modal closes |
| 2026-05-13 | Chat opens only after drag ends and mouse is released: onChange sets didSpin ref; onEnd fires onSpun only if didSpin=true and hasTriggered=false; both refs reset when modal closes |
| 2026-05-13 | Fix chat opening before model stops spinning: onEnd now sets waitingForStop flag; useFrame polls azimuthalAngle each frame and only fires onSpun after STILL_FRAMES_REQUIRED (8) consecutive frames below STILL_THRESHOLD (0.0002 rad) |
