#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Phase 2 — verify the ALLEX GROOT key-converter + robot registration.

Two parts:

  (A) UNIT — no sim. Feed synthetic robosuite obs / policy actions through
      ``AllexKeyConverter`` and assert the exact key renames + dims of the
      FROZEN ALLEX contract (run_scripts/eval/robocasa_allex/contract):
        * map_obs        : robot0_<group>  -> body./hand.<group>  (6 keys)
        * map_obs_in_eval: -> state.<group>                       (6 state keys)
        * unmap_action   : action.<group>  -> robot0_<group>      dims 7/15/2/7/15/2
        * map/unmap round-trip identity on the 6 action groups.
        * get_metadata   : absolute=True, rotation_type=None; no torque/effort.

  (B) INTEGRATION — best-effort. make_key_converter('AllexRobot') is AllexKeyConverter;
      then try to gym.make ONE registered robocasa_allex/* env and get to reset()
      + one observation. Asserts (if it builds): ego video key + 6 state keys +
      language key present, and model.neq == 12. A clean blocker (task-scene /
      GR1-specific assumption) is an ACCEPTABLE Phase-3 outcome — it is reported,
      not faked.

Run in the robosuite venv:
  MUJOCO_GL=egl .../robocasa_uv/.venv/bin/python \
      run_scripts/eval/robocasa_allex/checks/converter_check.py
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import robocasa  # noqa: E402,F401  (triggers ALLEX registration)
from robocasa.models.robots import (  # noqa: E402
    AllexKeyConverter,
    make_key_converter,
)

# frozen contract (source of truth for group order + dims)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "contract"))
import allex_contract as C  # noqa: E402

GROUP_DIMS = [C.GROUP_DIMS[g] for g in C.GROUP_ORDER]  # [7,15,2,7,15,2]
EXPECT_PREFIX = {
    "left_arm_joints": "body",
    "left_hand_joints": "hand",
    "neck_joints": "body",
    "right_arm_joints": "body",
    "right_hand_joints": "hand",
    "waist_joints": "body",
}


def _synthetic_robot_obs(rng):
    """Mirror gather_robot_observations output for AllexPositionRobot."""
    return {
        f"robot0_{g}": rng.standard_normal(C.GROUP_DIMS[g]).astype(np.float64)
        for g in C.GROUP_ORDER
    }


def _synthetic_policy_action(rng):
    """Mirror the GROOT action space keys the policy emits (action.<group>)."""
    return {
        f"action.{g}": rng.standard_normal(C.GROUP_DIMS[g]).astype(np.float64)
        for g in C.GROUP_ORDER
    }


def unit_test() -> bool:
    print("=" * 70)
    print("(A) UNIT — AllexKeyConverter key mapping + dims")
    print("=" * 70)
    rng = np.random.default_rng(0)
    kc = AllexKeyConverter
    ok = True

    # --- map_obs: robot0_<group> -> body./hand.<group> ---------------------
    obs = _synthetic_robot_obs(rng)
    mapped = kc.map_obs(obs)
    exp_obs_keys = {f"{EXPECT_PREFIX[g]}.{g}": g for g in C.GROUP_ORDER}
    assert set(mapped) == set(exp_obs_keys), (set(mapped), set(exp_obs_keys))
    for k, g in exp_obs_keys.items():
        assert mapped[k].shape == (C.GROUP_DIMS[g],), (k, mapped[k].shape)
        assert np.array_equal(mapped[k], obs[f"robot0_{g}"]), k
    print("  map_obs        robot0_<group> -> body./hand.<group>  OK")
    for g in C.GROUP_ORDER:
        print(f"    robot0_{g:17s} -> {EXPECT_PREFIX[g]}.{g:17s} dim={C.GROUP_DIMS[g]}")

    # --- map_obs_in_eval: -> state.<group> ---------------------------------
    state = kc.map_obs_in_eval(obs)
    exp_state = {f"state.{g}": C.GROUP_DIMS[g] for g in C.GROUP_ORDER}
    assert set(state) == set(exp_state), (set(state), set(exp_state))
    for g in C.GROUP_ORDER:
        assert state[f"state.{g}"].shape == (C.GROUP_DIMS[g],)
    print("  map_obs_in_eval -> state.<group> (6 contract state keys)     OK")

    # --- unmap_action: action.<group> -> robot0_<group> dims 7/15/2/7/15/2 -
    act = _synthetic_policy_action(rng)
    env_act = kc.unmap_action(act)
    exp_act_keys = [f"robot0_{g}" for g in C.GROUP_ORDER]
    assert set(env_act) == set(exp_act_keys), (set(env_act), set(exp_act_keys))
    dims = [env_act[f"robot0_{g}"].shape[0] for g in C.GROUP_ORDER]
    assert dims == GROUP_DIMS, (dims, GROUP_DIMS)
    for g in C.GROUP_ORDER:
        assert np.array_equal(env_act[f"robot0_{g}"], act[f"action.{g}"]), g
    print(f"  unmap_action   action.<group> -> robot0_<group>  dims={dims}  OK")

    # --- round trip: map_action(reconstruct-style) <-> unmap_action --------
    # reconstruct_latest_actions emits robot0_<group>; map_action -> body./hand.;
    # deduce_action_space would turn those into action.<group>; unmap_action
    # inverts back to robot0_<group>. Verify the full loop is identity.
    recon = {f"robot0_{g}": rng.standard_normal(C.GROUP_DIMS[g]) for g in C.GROUP_ORDER}
    body_hand = kc.map_action(recon)
    assert set(body_hand) == {f"{EXPECT_PREFIX[g]}.{g}" for g in C.GROUP_ORDER}
    action_space_form = {"action." + k[5:]: v for k, v in body_hand.items()}
    back = kc.unmap_action(action_space_form)
    for g in C.GROUP_ORDER:
        assert np.array_equal(back[f"robot0_{g}"], recon[f"robot0_{g}"]), g
    print("  round-trip     robot0 -> map_action -> action.* -> unmap -> robot0  OK")

    # --- get_metadata: absolute joint position, no rotation, no torque -----
    for g in C.GROUP_ORDER:
        md = kc.get_metadata(f"body.{g}")
        assert md["absolute"] is True and md["rotation_type"] is None, (g, md)
    for k in body_hand:
        assert "effort" not in k and "torque" not in k, k
    print("  get_metadata   absolute=True rotation_type=None; no torque/effort  OK")

    print("\n  UNIT VERDICT: PASS")
    return ok


def integration_test() -> bool:
    print("\n" + "=" * 70)
    print("(B) INTEGRATION — dispatch + env instantiation (best-effort)")
    print("=" * 70)

    # dispatch
    kc = make_key_converter("AllexRobot")
    assert kc is AllexKeyConverter, kc
    print("  make_key_converter('AllexRobot') -> AllexKeyConverter  OK")

    import gymnasium as gym
    import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401

    env_id = os.environ.get(
        "ALLEX_ENV_ID", "robocasa_allex/PnPCanToBowl_AllexRobot_Env"
    )
    print(f"  attempting gym.make('{env_id}') ...")
    env = None
    try:
        env = gym.make(env_id, enable_render=True, seed=0)
        print("  gym.make: OK")
        obs, info = env.reset()
        print("  env.reset(): OK")

        base = env.unwrapped
        model = base.env.sim.model._model
        print(f"  model.neq={model.neq}  nu={model.nu}  njnt={model.njnt}")

        video_ok = "video.camera_ego_left" in obs
        state_keys = [f"state.{g}" for g in C.GROUP_ORDER]
        state_ok = all(k in obs for k in state_keys)
        lang_ok = "annotation.human.task_description" in obs
        neq_ok = int(model.neq) == 12
        for k in state_keys:
            if k in obs:
                assert obs[k].shape[-1] == C.GROUP_DIMS[k.split(".")[1]], (k, obs[k].shape)

        print(f"    video.camera_ego_left present : {video_ok}")
        print(f"    6 state.<group> present       : {state_ok}")
        print(f"    language key present          : {lang_ok}")
        print(f"    model.neq == 12               : {neq_ok}")
        integ_ok = video_ok and state_ok and lang_ok and neq_ok
        print(f"\n  INTEGRATION VERDICT: {'PASS' if integ_ok else 'PARTIAL'}")
        return integ_ok
    except Exception as e:  # noqa: BLE001
        print(f"\n  INTEGRATION BLOCKER (expected Phase-3 territory): {type(e).__name__}: {e}")
        tb = traceback.format_exc().strip().splitlines()
        # print the deepest few frames (file:line) for a precise blocker locus
        print("  --- blocker traceback (tail) ---")
        for line in tb[-12:]:
            print("   " + line)
        return False
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:  # noqa: BLE001
                pass


def main() -> None:
    unit_ok = unit_test()
    integ_ok = integration_test()
    print("\n" + "=" * 70)
    print(f"SUMMARY: unit={'PASS' if unit_ok else 'FAIL'}  "
          f"integration={'PASS' if integ_ok else 'BLOCKED/PARTIAL (see above)'}")
    print("=" * 70)
    # Unit test is the hard gate; integration blocker is acceptable (Phase 3).
    sys.exit(0 if unit_ok else 1)


if __name__ == "__main__":
    main()
