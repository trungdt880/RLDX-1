# Phase 1.4 — robosuite merge + tracking de-risk: PASS

Ran `robosuite_merge_check.py` (bypasses the Robot/controller wrapper): wrap ALLEX
`robot.xml` in a minimal `MujocoXMLModel` (triggers `robot0_` prefix + `add_prefix`
rewrite of `<equality>` joint refs), merge into `EmptyArena`, drive `sim.data.ctrl`
with a replayed npz motion.

## Result
- Merge + prefix PRESERVE: nu=48 (all `<position>`), njnt=60, neq=12 (couplings intact).
- Tracking inside merged+prefixed model == standalone 0.5c: arm 0.043°, hand 0.374°,
  neck 0.035°, waist 0.046° RMS.  ⇒ passthrough-via-position-actuators works in robosuite.

## robot.xml adaptations made (ALLEX.xml -> fork assets/robots/allex/robot.xml)
1. geom groups remapped to robosuite convention: visual 2->1, collision 3->0
   (robosuite `_add_default_name_filter` only accepts group {None,0=col,1=vis}).
2. meshes: symlink `allex/meshes -> ~/workspace/allex_model/meshes`; rewrote 131
   `file="X.stl"` -> `file="meshes/X.stl"` (robosuite `resolve_asset_dependency`
   joins robot.xml folder + file, IGNORING `<compiler meshdir>`); removed abs meshdir.

## Findings carried forward
- Phase 3 MOUNTING: mount ALLEX elevated (~0.9m) — EmptyArena floor at z=0 vs ALLEX
  base at origin causes floor penetration -> contact shoves arms (13deg). Not a physics bug.
- Phase 1 env option: force `integrator=implicitfast` (robosuite arena defaults to Euler);
  ALLEX's stiff kp position-PD wants it.
