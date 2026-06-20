#!/usr/bin/env python
"""Real-robot deployment stub for RLDX-1 on the Fourier GR1 (arms+waist+hands).

Adapted from run_scripts/deploy/droid_deploy.py, stripped to the GR1 contract:
  * 5 state/action groups: left_arm(7) left_hand(6) right_arm(7) right_hand(6) waist(3)
  * single egocentric camera  -> video key "ego_view"
  * language key "annotation.human.coarse_action", prefixed "unlocked_waist: "
  * actions are ABSOLUTE joint targets (radians for arms/waist, hand DOF for hands)

There is NO GR1 hardware SDK in this repo. All robot I/O is behind the
`GR1Robot` interface below — implement its 4 methods against your stack
(fourier-grx / ROS / your controller). A `DummyGR1Robot` is provided so you can
run the FULL control loop end-to-end (model + chunking + pacing) with no robot.

Run (main training .venv, needs GPU + checkpoint):

    uv run python examples/gr1_inference/gr1_deploy.py \
        --model-path RLWRLD/RLDX-1-FT-GR1 \
        --instruction "pick the cup and place it in the drawer" \
        --dummy-robot            # drop this flag once GR1Robot is implemented
"""

import argparse
import time
from collections import deque

import numpy as np
from PIL import Image

# import side effect registers RLDX into HF Auto* registries. keep explicit.
import rldx  # noqa: F401
from rldx.data.embodiment_tags import EmbodimentTag
from rldx.policy.rldx_policy import RLDXPolicy

# GR1 ArmsAndWaist + Fourier hands. Order is informational; the model is keyed
# by name (modality_cfg), not position. Source: rldx/configs/data/gr1_config.py
# + robocasa key-converter (robot0_left/right=7, *_gripper=6, torso=3).
GR1_STATE_DIMS = {
    "left_arm": 7,
    "left_hand": 6,
    "right_arm": 7,
    "right_hand": 6,
    "waist": 3,
}


# ----------------------------------------------------------------------------
# Hardware interface — IMPLEMENT THIS against your GR1 stack.
# ----------------------------------------------------------------------------
class GR1Robot:
    """Contract between RLDX-1 and your GR1 hardware. Implement all 4 methods."""

    def get_ego_frame(self) -> np.ndarray:
        """Return the egocentric RGB frame as uint8 (H, W, 3). Any size; we resize."""
        raise NotImplementedError

    def get_state(self) -> dict[str, np.ndarray]:
        """Return current proprioception, one float32 array per group.

        REQUIRED keys + dims (must match the model's state modality):
            left_arm (7,)  left_hand (6,)  right_arm (7,)  right_hand (6,)  waist (3,)
        Arms/waist in radians, hands in the Fourier hand's DOF units — SAME units
        the training data used (absolute joint positions).
        """
        raise NotImplementedError

    def apply_action(self, action: dict[str, np.ndarray]) -> None:
        """Command ONE timestep. `action` has the same 5 keys, each shape (D,).

        Values are ABSOLUTE joint targets (model action rep = ABSOLUTE). Send
        them to your position controller. Do NOT integrate/accumulate.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Move robot to a safe home pose before a new episode."""
        raise NotImplementedError


class DummyGR1Robot(GR1Robot):
    """No-op robot so the control loop runs without hardware. Prints commands."""

    def __init__(self, image_size: int = 256, verbose: bool = False):
        self.image_size = image_size
        self.verbose = verbose

    def get_ego_frame(self) -> np.ndarray:
        return np.random.randint(0, 256, (self.image_size, self.image_size, 3), dtype=np.uint8)

    def get_state(self) -> dict[str, np.ndarray]:
        return {k: np.zeros(d, dtype=np.float32) for k, d in GR1_STATE_DIMS.items()}

    def apply_action(self, action: dict[str, np.ndarray]) -> None:
        if self.verbose:
            print("  cmd:", {k: np.round(v, 3) for k, v in action.items()})

    def reset(self) -> None:
        print("[dummy] reset to home pose")


# ----------------------------------------------------------------------------
# Observation builder — nested format RLDXPolicy.get_action expects.
# ----------------------------------------------------------------------------
def build_observation(
    frames: np.ndarray,            # (T, H, W, 3) uint8 — already at target size
    state: dict[str, np.ndarray],  # per-group (D,) float32
    instruction: str,
    *,
    video_key: str,
    state_keys: list[str],
    language_key: str,
    waist_prefix: bool,
) -> dict:
    if waist_prefix and not instruction.startswith("unlocked_waist:"):
        instruction = f"unlocked_waist: {instruction}"
    return {
        "video": {video_key: frames[None].astype(np.uint8)},          # (1, T, H, W, 3)
        "state": {k: state[k].reshape(1, 1, -1).astype(np.float32) for k in state_keys},
        "language": {language_key: [[instruction]]},                   # (B=1, T=1)
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="RLWRLD/RLDX-1-FT-GR1")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--instruction", required=True)
    p.add_argument("--control-hz", type=float, default=20.0, help="GR1 datasets are freq20")
    p.add_argument("--open-loop-horizon", type=int, default=16, help="steps per chunk before re-query")
    p.add_argument("--max-timesteps", type=int, default=720)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--no-waist-prefix", action="store_true", help="disable 'unlocked_waist: ' prefix")
    p.add_argument("--deactivate-memory", action="store_true")
    p.add_argument("--dummy-robot", action="store_true", help="run loop with DummyGR1Robot")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    # 1. Load policy (model + processor + per-embodiment heads + norm stats).
    policy = RLDXPolicy(
        embodiment_tag=EmbodimentTag.GENERAL_EMBODIMENT,
        model_path=args.model_path,
        device=args.device,
        strict=True,
        deactivate_memory=args.deactivate_memory,
    )

    # 2. Read the contract FROM the checkpoint — never hardcode. If you trained a
    #    variant (different cams/joints), this auto-adapts.
    modality_cfg = policy.get_modality_config()
    video_keys = modality_cfg["video"].modality_keys          # GR1: ["ego_view"]
    state_keys = modality_cfg["state"].modality_keys          # GR1: 5 groups
    video_T = len(modality_cfg["video"].delta_indices)        # GR1: 1
    language_key = policy.language_key                         # GR1: annotation.human.coarse_action
    assert len(video_keys) == 1, f"GR1 stub assumes one camera, got {video_keys}"
    video_key = video_keys[0]

    print("=== GR1 deploy ===")
    print(f"  checkpoint : {args.model_path}")
    print(f"  video key  : {video_key}  (T={video_T})")
    print(f"  state keys : {state_keys}")
    print(f"  language   : {language_key}  (prefix={'on' if not args.no_waist_prefix else 'off'})")
    print(f"  memory     : {policy.use_memory}")
    print(f"  control    : {args.control_hz} Hz, open-loop {args.open_loop_horizon}")

    robot: GR1Robot = DummyGR1Robot(args.image_size, args.verbose) if args.dummy_robot else _connect_real_robot()

    # 3. Episode setup.
    robot.reset()
    session_id = f"gr1_{int(time.time())}"   # stable per episode -> memory isolation
    is_first_inference = True
    period = 1.0 / args.control_hz

    # Frame history (GR1 T=1 -> length-1; general for T>1 checkpoints).
    frame_hist = deque(maxlen=video_T)

    chunk = None                  # (open_loop_horizon, total_dim) not used; we keep per-key
    chunk_keys = None
    step_in_chunk = 0

    for t in range(args.max_timesteps):
        loop_start = time.time()

        # --- sense ---
        frame = robot.get_ego_frame()
        if frame.shape[:2] != (args.image_size, args.image_size):
            frame = np.asarray(Image.fromarray(frame).resize((args.image_size, args.image_size)))
        frame_hist.append(frame)
        while len(frame_hist) < video_T:        # warm up history on first frames
            frame_hist.append(frame)
        state = robot.get_state()

        # --- infer a new chunk when the previous one is exhausted ---
        if chunk is None or step_in_chunk >= args.open_loop_horizon:
            step_in_chunk = 0
            obs = build_observation(
                np.stack(frame_hist), state, args.instruction,
                video_key=video_key, state_keys=state_keys,
                language_key=language_key, waist_prefix=not args.no_waist_prefix,
            )
            options = {"reset_memory": [is_first_inference], "session_ids": [session_id]}
            t0 = time.time()
            action_dict, _ = policy.get_action(obs, options=options)   # {grp: (1,16,D)} physical units
            is_first_inference = False
            if args.verbose:
                print(f"[t={t}] inference {1000*(time.time()-t0):.1f} ms")
            chunk = action_dict
            chunk_keys = list(action_dict.keys())

        # --- act: send timestep `step_in_chunk` of the chunk ---
        cmd = {k: chunk[k][0, step_in_chunk] for k in chunk_keys}      # each (D,)
        robot.apply_action(cmd)
        step_in_chunk += 1

        # --- pace to control frequency ---
        dt = time.time() - loop_start
        if dt < period:
            time.sleep(period - dt)

    print("done.")


def _connect_real_robot() -> GR1Robot:
    """Wire up your real GR1 here, e.g.:

        from your_gr1_sdk import GR1Client
        class RealGR1(GR1Robot): ...
        return RealGR1(GR1Client(...))
    """
    raise NotImplementedError(
        "Implement GR1Robot for your hardware and return it here, "
        "or pass --dummy-robot to test the loop without a robot."
    )


if __name__ == "__main__":
    main()
