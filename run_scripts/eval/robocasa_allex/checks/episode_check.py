#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Phase 3 — full-episode harness check for the ALLEX RoboCasa env.

Goal: prove a gripperless ALLEX env runs a FULL closed-loop-shaped rollout
(>= 150 steps) through ``env.step(action)`` WITHOUT crashing, now that the
success/reward path is gripper-optional.

What it does:
  * gym.make a registered ``robocasa_allex/*`` env, reset().
  * Replay a REAL npz motion (``hello.npz``), mapped to the 6-group GROOT action
    dict (``action.<group>`` via ``allex_contract.JOINT_NAMES``), through
    ``env.step`` for >= STEPS frames.

Per-step assertions (whole episode, no exception allowed):
  * obs carries the full ALLEX contract: ``video.camera_ego_left`` + 6
    ``state.<group>`` + ``annotation.human.task_description``.
  * reward is a finite float; terminated/truncated are bools; info is a dict
    with a ``success`` field (harness stub -> may be False for a gripperless
    robot; NO success claim is made).
  * ``model.neq == 12`` throughout (equality couplings survive hard_reset).
  * sim stays finite (no NaN/inf in qpos/qvel).
  * the ego frame is a REAL render (pixel variance > threshold), so a policy
    would get a usable ego view.

Run in the robosuite venv:
  MUJOCO_GL=egl .../robocasa_uv/.venv/bin/python \
      run_scripts/eval/robocasa_allex/checks/episode_check.py
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
import robocasa.utils.gym_utils.gymnasium_groot  # noqa: E402,F401

# frozen contract (group order + intra-group joint names)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "contract"))
import allex_contract as C  # noqa: E402

MOTION = Path.home() / "workspace/allex_model/examples/motions/hello.npz"
ENV_ID = os.environ.get("ALLEX_ENV_ID", "robocasa_allex/PnPCanToBowl_AllexRobot_Env")
STEPS = int(os.environ.get("ALLEX_STEPS", "150"))
CKPT_EVERY = 30
EGO_VAR_MIN = 1.0  # variance of uint8 ego frame; a real render is >> this


def _load_motion():
    z = np.load(MOTION, allow_pickle=True)
    names = [str(x) for x in z["joint_names"]]
    q = z["q"].astype(np.float64)
    return names, q


def _action_for_frame(names, frame) -> dict:
    """Map one npz frame -> GROOT policy action dict (action.<group>)."""
    name2val = dict(zip(names, frame))
    return {
        f"action.{g}": np.array(
            [name2val[j] for j in C.JOINT_NAMES[g]], dtype=np.float64
        )
        for g in C.GROUP_ORDER
    }


def main() -> None:
    import gymnasium as gym

    print("=" * 72)
    print(f"Phase 3 episode_check — {ENV_ID}  ({STEPS} steps)")
    print("=" * 72)

    names, q = _load_motion()
    print(f"[motion] hello.npz frames={q.shape[0]} joints={q.shape[1]}")

    env = gym.make(ENV_ID, enable_render=True, seed=0)
    obs, info = env.reset()
    base = env.unwrapped
    model = base.env.sim.model._model
    raw_data = base.env.sim.data._data
    print(f"[reset] OK  model.neq={model.neq} nu={model.nu} njnt={model.njnt}")
    print(f"[reset] robot arms={base.env.robots[0].arms} "
          f"gripperless={len(base.env.robots[0].arms) == 0}")

    state_keys = [f"state.{g}" for g in C.GROUP_ORDER]
    contract_keys = ["video.camera_ego_left", *state_keys,
                     "annotation.human.task_description"]

    def check_obs(o, tag):
        for k in contract_keys:
            assert k in o, f"{tag}: missing obs key {k}"
        for k in state_keys:
            assert o[k].shape[-1] == C.GROUP_DIMS[k.split(".")[1]], (k, o[k].shape)
        ego = np.asarray(o["video.camera_ego_left"])
        assert ego.ndim == 3 and ego.shape[-1] == 3, ego.shape
        return float(np.var(ego)), (float(ego.min()), float(ego.max()))

    v0, r0 = check_obs(obs, "reset")
    print(f"[reset] ego frame shape={np.asarray(obs['video.camera_ego_left']).shape} "
          f"var={v0:.2f} range={r0}")

    ok = True
    ego_vars = [v0]
    rewards = []
    n = len(q)
    try:
        for t in range(STEPS):
            frame = q[t % n]  # loop the motion if shorter than STEPS
            action = _action_for_frame(names, frame)
            obs, reward, terminated, truncated, info = env.step(action)

            # --- shape / type contract ---
            assert isinstance(reward, (int, float, np.floating)), type(reward)
            assert np.isfinite(reward), reward
            assert isinstance(terminated, (bool, np.bool_)), type(terminated)
            assert isinstance(truncated, (bool, np.bool_)), type(truncated)
            assert isinstance(info, dict) and "success" in info, info
            rewards.append(float(reward))

            # --- obs contract + ego render ---
            var, rng = check_obs(obs, f"step{t}")
            ego_vars.append(var)
            assert var > EGO_VAR_MIN, (
                f"step{t}: ego frame degenerate (var={var:.4f} <= {EGO_VAR_MIN}); "
                "camera returned a constant/blank image"
            )

            # --- structural invariants ---
            assert int(model.neq) == 12, f"step{t}: model.neq={model.neq} != 12"
            assert np.isfinite(raw_data.qpos).all(), f"step{t}: NaN/inf in qpos"
            assert np.isfinite(raw_data.qvel).all(), f"step{t}: NaN/inf in qvel"

            if (t + 1) % CKPT_EVERY == 0 or t == 0:
                print(f"  [step {t + 1:>3}] reward={reward:.3f} "
                      f"term={bool(terminated)} trunc={bool(truncated)} "
                      f"success={info['success']} neq={model.neq} "
                      f"ego_var={var:.1f} qpos_finite=True")
            if terminated or truncated:
                print(f"  [step {t + 1}] episode ended early "
                      f"(term={bool(terminated)} trunc={bool(truncated)}) "
                      f"-> resetting to continue to {STEPS} steps")
                obs, info = env.reset()
    except Exception:  # noqa: BLE001
        ok = False
        print("\n  EPISODE CRASHED:")
        for line in traceback.format_exc().strip().splitlines()[-14:]:
            print("   " + line)
    finally:
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass

    print("-" * 72)
    if rewards:
        print(f"[summary] steps_run={len(rewards)} "
              f"reward[min/max/mean]={min(rewards):.3f}/{max(rewards):.3f}/"
              f"{np.mean(rewards):.3f}")
    print(f"[summary] ego_var[min/mean/max]="
          f"{min(ego_vars):.2f}/{np.mean(ego_vars):.2f}/{max(ego_vars):.2f} "
          f"(threshold>{EGO_VAR_MIN})")
    full = ok and len(rewards) >= STEPS and min(ego_vars) > EGO_VAR_MIN
    verdict = "PASS" if full else "FAIL"
    print(f"\nVERDICT [Phase 3 full-episode harness]: {verdict}")
    print(f"  ran {len(rewards)}/{STEPS} steps with no exception: {ok}")
    print(f"  model.neq==12 throughout, sim finite, ego non-degenerate: "
          f"{full}")
    sys.exit(0 if full else 1)


if __name__ == "__main__":
    main()
