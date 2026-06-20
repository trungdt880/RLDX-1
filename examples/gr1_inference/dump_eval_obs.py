#!/usr/bin/env python
"""Dump ONE GR1 table-top observation from the sim, for offline inference.

Builds the SAME env stack the evaluator uses (GrootRoboCasaEnv ->
MultiStepWrapper -> SyncVectorEnv), resets once, and saves:
  * sample_obs.pkl   - the exact flat obs dict the policy client sends
  * sample_frame.png - the ego_view frame (for the pure-HF infer path)

Run with the ROBOCASA eval venv (NOT the training .venv), same interpreter
the eval script uses:

    rldx/eval/sim/robocasa-gr1-tabletop-tasks/robocasa_uv/.venv/bin/python \
        examples/gr1_inference/dump_eval_obs.py \
        --env-name gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env

Then infer with the main env:

    uv run python examples/gr1_inference/infer_gr1.py \
        --model-path RLWRLD/RLDX-1-FT-GR1 --sample sample_obs.pkl
"""

import argparse
import pickle

import gymnasium as gym
import numpy as np
from PIL import Image

from rldx.eval.rollout_policy import get_robocasa_env_fn
from rldx.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--env-name",
        default="gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-pkl", default="sample_obs.pkl")
    p.add_argument("--out-png", default="sample_frame.png")
    args = p.parse_args()

    def make():
        base = get_robocasa_env_fn(args.env_name, seed=args.seed)()
        # GR1 modality config: video + state both use delta_indices=[0] -> T=1.
        # n_action_steps only matters for stepping; 16 matches action_horizon.
        return MultiStepWrapper(
            base,
            video_delta_indices=np.array([0]),
            state_delta_indices=np.array([0]),
            n_action_steps=16,
            max_episode_steps=720,
        )

    # SyncVectorEnv(n=1) adds the leading batch dim B=1, matching the rollout.
    env = gym.vector.SyncVectorEnv([make])
    obs, _ = env.reset(seed=args.seed)

    print("=== dumped observation (flat sim format) ===")
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            print(f"{k:45s} shape={v.shape} dtype={v.dtype}")
        else:
            print(f"{k:45s} {type(v).__name__}: {v}")

    with open(args.out_pkl, "wb") as f:
        pickle.dump(obs, f)
    print(f"\nsaved obs -> {args.out_pkl}")

    # Pull the ego_view frame (B, T, H, W, 3) -> (H, W, 3) for a visual sanity check.
    cam = next((k for k in obs if k.startswith("video.ego_view")), None)
    if cam is not None:
        frame = np.asarray(obs[cam])[0, 0]
        Image.fromarray(frame).save(args.out_png)
        print(f"saved frame -> {args.out_png}")

    env.close()


if __name__ == "__main__":
    main()
