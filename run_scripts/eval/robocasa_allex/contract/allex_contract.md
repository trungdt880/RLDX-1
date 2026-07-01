# ALLEX ⇄ RLDX-1-MT-ALLEX — Frozen Observation/Action Contract (Phase 0)

Authoritative spec of what the `RLDX-1-MT-ALLEX` checkpoint (`RLWRLD/RLDX-1-MT-ALLEX`,
registry key `rldx_1_midtrain_allex`, embodiment tag **`general_embodiment`**) consumes
and emits. The RoboCasa sim robot, key-converter, and rollout client **must** match this
exactly. `allex_contract.py` is the executable form (constants + assertions + normalize);
import it rather than re-deriving anything. Everything here is derived from the checkpoint's
own `processor/{statistics,processor_config,embodiment_id}.json` + `config.json` and RLDX
source, cross-checked by `verify_joint_order.py`.

## 1. Embodiment / dims
- Tag: `general_embodiment` (index 0 in `embodiment_id.json`). The ALLEX config
  (`rldx/configs/data/midtrain_allex_data_config.py`) registers under it. This checkpoint's
  `general_embodiment` stats have **15-DOF hands** → they are ALLEX's (openarm/other configs
  sharing the tag have 6-DOF hands).
- `REAL_DIM = 48`; padded to `max_state_dim = max_action_dim = 64`; `action_horizon = 40`.

## 2. State/action group concat order + dims — **VERIFIED**
Source: `processor_config.json → processor_kwargs.modality_configs.general_embodiment.state.modality_keys`
(== action keys). The list order **is** the flat-vector concat order:

| idx range | group | dim |
|---|---|---|
| 0–6   | `left_arm_joints`   | 7  |
| 7–21  | `left_hand_joints`  | 15 |
| 22–23 | `neck_joints`       | 2  |
| 24–30 | `right_arm_joints`  | 7  |
| 31–45 | `right_hand_joints` | 15 |
| 46–47 | `waist_joints`      | 2  |

Torque/physics uses the same grouping/order with `_effort` suffixes (`left_arm_effort`, …).

## 3. Intra-group joint ordering — **STRONGLY SUPPORTED** (see caveat)
No in-repo per-joint name list exists for `general_embodiment`. Ordering below = the per-group
slice of the WIRobotics npz `joint_names` order, **cross-checked** by `verify_joint_order.py`
(each channel's `[q01,q99]` percentile band must fit inside that joint's MJCF limit):
**500/500 random hand-permutations were strictly worse (uniquely best); 0 gross violations on
arms/neck/waist.** Full per-joint list is frozen in `allex_contract.py::JOINT_NAMES`. Summary:
- arms (each 7): `Shoulder_Pitch, Shoulder_Roll, Shoulder_Yaw, Elbow, Wrist_Yaw, Wrist_Roll, Wrist_Pitch`
- hands (each 15): `Thumb_Yaw, Thumb_CMC, Thumb_MCP, Index_ABAD, Index_MCP, Index_PIP, Middle_ABAD, Middle_MCP, Middle_PIP, Ring_{ABAD,MCP,PIP}, Little_{ABAD,MCP,PIP}`
- neck (2): `Neck_Pitch, Neck_Yaw` · waist (2): `Waist_Yaw, Waist_Lower_Pitch`

**Caveat / residual:** two right-hand PIP channels (`R_Middle_PIP`, `R_Ring_PIP`) have wide/negative
percentile bands that *no* permutation fixes → a real-data quirk, not a mis-ordering. Minor MCP
q99 overshoots (~0.1–0.22 rad) are real-vs-sim ROM slack. **Byte-exact confirmation still requires
the `real_allex` `meta/modality.json`.** This bar is acceptable for a harness (no success-rate claim);
get the dataset's modality.json before trusting any live rollout *number*.

## 4. Units, signs, action semantics
- State & action are **absolute joint angles in radians** (matches ALLEX `<position>` actuators).
- **Action representation is ABSOLUTE for all 6 groups** (`midtrain_allex_data_config.py`
  ActionConfig `rep=ABSOLUTE, type=NON_EEF`). The global `use_relative_action=True` is a
  data-pipeline default; `_decode` (`rldx/policy/policy_runtime.py:442-461`) only inverts
  *state-relative* (DELTA) groups, so **ALLEX action output = absolute joint targets**, fed
  1:1 to the position actuators (NO IK, unlike GR1's WHOLE_BODY_IK). Compare openarm_inspire,
  which mixes DELTA arms + ABSOLUTE hands — proof the rep is per-group, not global.

## 5. Normalization — percentile min-max, clipped
`use_percentiles=True`, `clip_outliers=True`, `apply_sincos_state_encoding=False`.
Per channel (impl: `rldx/data/state_action/state_action_processor.py` `_compute_normalization_parameters`
→ min:=q01, max:=q99; `rldx/data/utils.py` normalize/unnormalize min-max):
- forward:  `n = clip( 2*(x - q01)/(q99 - q01) - 1, -1, 1)`
- inverse:  `x = (clip(n,-1,1)+1)/2 * (q99 - q01) + q01`
- degenerate channels (`q99==q01`) map to 0 and are non-invertible (RLDX does the same).
Stats live in `allex_stats.json['general_embodiment'][{state,action,torque}][group]`.
Use `allex_contract.py::normalize/denormalize` — do not hand-roll.

## 6. 48→64 zero-pad
Real 48-vector is zero-padded to 64 (`max_*_dim=64`); model output sliced back to 48 via the
`general_embodiment` real_dim from norm stats (`rldx/policy/policy_runtime.py:~280-320`). Client
works in the **48-dim real space**; padding is internal to the processor/runtime.

## 7. Video
- Single key **`camera_ego_left`** (mono ego). Model context = **4 frames** at action-step
  strides **[-6, -4, -2, 0]** (`video_length=4`, `video_stride=2`;
  `rldx/experiment/features/video.py:33` → `{(i-3)*2 for i in range(4)}`).
- RoboCasa GROOT preprocessing renders/resizes ego frames (robocasa convention 256²); the RLDX
  processor (Qwen3-VL vision) handles final resize. Confirm exact expected input res at serve time.

## 8. Torque / physics — **OPTIONAL (trained-supported), omit for first rollout**
- `physics_keys=['torque']`, `physics_dims=[48]`, **`allow_missing_physics=True`**.
- `rldx/model/core/processing_rldx.py:564-573`: with `allow_missing_physics=True` the processor
  ALWAYS emits `physics` + `physics_mask` — mask **1.0 if torque present, 0.0 if omitted**. The
  model was trained with this masking, so **omitting torque is in-distribution**, not OOD.
- ⇒ Phase 2 need not emit torque for a first rollout (send none → mask 0). Phase 0.5b measures
  whether a MuJoCo torque surrogate *helps* vs the mask-0 path; the mask-0 path is the safe default.

## 9. Memory + video window (VERIFIED at serve time — Phase 4)
Two SEPARATE mechanisms, do not conflate:
- **Video window is CLIENT-assembled.** The loaded sim-policy-wrapper advertises video
  `delta_indices=[-6,-4,-2,0]` (T=4) and strictly asserts `video.shape[1]==4`
  (`rldx/policy/rldx_policy.py:362`). So the CLIENT must send `video.camera_ego_left` as
  `(B,4,H,W,C)` — the 4-frame strided window assembled from its own ring buffer of past ego
  frames (retain >=8 sim frames; sample at offsets -6,-4,-2,0). It is NOT a single frame that
  the server expands. (Corrects an earlier draft of this section.)
- **Cognition memory is SERVER-side.** `memory_length=4`, `n_cog_tokens=64`,
  `memory_n_cog_tokens=16`: the server buffers cognition tokens across calls. Client must send
  `reset_memory=[True]` on episode reset (both in `reset(options=...)` and on the first
  `get_action`) — wire format in `rldx/policy/step_request.py`.

## 10. Language
Key `annotation.human.task_description`; `formalize_language=True`. Flat client sends the task
string; wrapper nests to `[[str]]` shape (B,1).

---
### Regenerate / validate
```
/home/thor/RLDX-1/.venv/bin/python run_scripts/eval/robocasa_allex/contract/allex_contract.py        # self-test: normalize round-trip + order assertion
/home/thor/RLDX-1/.venv/bin/python run_scripts/eval/robocasa_allex/contract/verify_joint_order.py     # ordering cross-check vs MJCF limits
```
Both must print PASS / "STRONGLY SUPPORTED". `allex_stats.json` is the `general_embodiment` slice
of the checkpoint `processor/statistics.json`.
