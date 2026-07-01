# ALLEX → RoboCasa — Phase 0 / 0.5 GO/NO-GO Gate

**Decision: GO to Phase 1** (build the RoboCasa `AllexRobot` + env), with torque omitted
and three caveats logged. Deliverable remains a *validated harness* — **no success-rate claim.**

## Gate inputs (all offline, against the real `RLDX-1-MT-ALLEX` checkpoint)

| Check | Result | Evidence |
|---|---|---|
| **Phase 0 — contract frozen** | ✅ | `contract/allex_contract.{md,py}` self-test PASS; groups/dims/normalization/action-abs/memory/torque-optional all resolved from checkpoint JSONs + RLDX source |
| **Joint ordering** | ✅ STRONGLY SUPPORTED | `verify_joint_order.py`: 500/500 hand-permutations worse; 0 gross arm/neck/waist violations |
| **0.5c — MuJoCo coupling/tracking** | ✅ PASS | `checks/coupling_check.py` both motions: arms sub-deg→~2° RMS, hands ~1–5° RMS, equality couplings resolve, stable |
| **0.5a — action-sanity** | ✅ PASS | `checks/action_sanity.py`: action (40,48) finite, **0 out-of-limit** on step0 (both poses), state-responsive (Δ=0.175 rad) |
| **0.5b — torque sensitivity** | ✅ GO (omit) | `checks/torque_sensitivity*.txt`: omit=mask0 is in-distribution (0.3 dropout); surrogate-OOD risk 0.118 rad avoided by omitting |

## Rollout configuration decided by the gate
- **Action:** absolute joint targets, 48-dim × 40-horizon, fed 1:1 to ALLEX position actuators (no IK). Group order + per-joint order frozen in `allex_contract.py`.
- **State:** absolute joint angles, 6 groups, percentile (q01/q99) min-max normalization.
- **Video:** single `camera_ego_left`, 4-frame ego history at strides [-6,-4,-2,0], 256².
- **Torque:** **OMIT** (physics_mask=0). Do NOT feed a MuJoCo torque surrogate (OOD).
- **Memory:** server-buffered; send `reset_memory` on episode reset.

## Caveats carried into Phase 1+ (not blockers; honest limits of a harness)
1. **Joint ordering** is cross-check-verified, not byte-exact. Obtain `real_allex` `meta/modality.json`
   before trusting any live rollout *number*.
2. **Torque mode mismatch**: sim runs mask=0 (trained-but-different from real mask=1 deployment).
3. **Domain gap** (visual + dynamics) is the dominant unknown and is the reason we make no
   success-rate claim. First bring-up is graded on closed-loop stability + sane motion only.

Artifacts: `run_scripts/eval/robocasa_allex/{contract,checks}/`.
