#!/usr/bin/env python
"""Minimal GR1 table-top inference, HuggingFace-style.

Load the RLDX-1 GR1 checkpoint once, feed it ONE observation (a saved frame +
robot state + a language instruction), get back a 16-step action chunk.

Run with the MAIN training env (uv .venv), NOT the robocasa eval venv:

    uv run python examples/gr1_inference/infer_gr1.py \
        --model-path RLWRLD/RLDX-1-FT-GR1 \
        --image examples/gr1_inference/sample_frame.png \
        --instruction "pick the cup and place it in the drawer"

Or replay an exact observation dumped from the evaluator (see
dump_eval_obs.py) so the inputs match the sim bit-for-bit:

    uv run python examples/gr1_inference/infer_gr1.py \
        --model-path RLWRLD/RLDX-1-FT-GR1 --sample sample_obs.pkl
"""

import argparse
import pickle

import numpy as np

# IMPORTANT: importing rldx registers RLDXConfig/RLDX/RLDXProcessor into the HF
# Auto* registries as an import side effect. The policy wrapper imports it too,
# but keep this here so the contract is explicit.
import rldx  # noqa: F401
from rldx.data.embodiment_tags import EmbodimentTag
from rldx.policy.rldx_policy import RLDXPolicy

# GR1 ArmsAndWaist + Fourier hands joint layout.
# Source of truth: rldx/configs/data/gr1_config.py (modality keys) and the
# robocasa key-converter (dims). Order here is just for building a dummy state.
STATE_DIMS = {
    "left_arm": 7,
    "left_hand": 6,
    "right_arm": 7,
    "right_hand": 6,
    "waist": 3,
}


def build_nested_obs(image: np.ndarray, instruction: str, state: dict | None = None):
    """Build the nested observation dict RLDXPolicy.get_action expects.

    Contract (single inference, batch B=1, history T=1 because every GR1
    modality uses delta_indices=[0]):
        video[key]    -> uint8   (B, T, H, W, 3)
        state[key]    -> float32 (B, T, D)
        language[key] -> list[list[str]]  shape (B, T)

    The GR1 sim prefixes the prompt with "unlocked_waist: " during training
    (GR1ArmsAndWaist branch of gymnasium_groot.py). Match it at inference.
    """
    # (H, W, 3) uint8 -> (1, 1, H, W, 3)
    video = image[None, None].astype(np.uint8)

    if state is None:
        # No real proprioception on hand -> zeros. Fine for a smoke test; for a
        # real rollout you MUST pass the robot's current joint state.
        state = {k: np.zeros((d,), dtype=np.float32) for k, d in STATE_DIMS.items()}

    return {
        "video": {"ego_view": video},
        "state": {
            k: state[k].reshape(1, 1, -1).astype(np.float32) for k in STATE_DIMS
        },
        "language": {
            "annotation.human.coarse_action": [[f"unlocked_waist: {instruction}"]]
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="RLWRLD/RLDX-1-FT-GR1")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--image", help="PNG/JPG egocentric frame (any size, gets resized to 256)")
    p.add_argument("--instruction", default="pick up the object and place it")
    p.add_argument("--sample", help="pickled obs dumped from the evaluator (overrides --image)")
    args = p.parse_args()

    # 1. Load model + processor + norm stats. One call. Same as AutoModel under
    #    the hood (policy_loader does AutoConfig/AutoModel.from_pretrained with
    #    torch_dtype=bfloat16) plus the processor and embodiment-specific heads.
    policy = RLDXPolicy(
        embodiment_tag=EmbodimentTag.GENERAL_EMBODIMENT,
        model_path=args.model_path,
        device=args.device,
        strict=True,
    )

    # 2. Build ONE observation.
    if args.sample:
        # Replay an exact eval observation. It is in the FLAT sim format
        # (keys like "state.left_arm", "video.ego_view_pad_res256_freq20",
        # "annotation.human.coarse_action"), so route it through the sim wrapper
        # which converts flat<->nested for us.
        from rldx.policy.rldx_policy import RLDXSimPolicyWrapper

        with open(args.sample, "rb") as f:
            obs = pickle.load(f)
        policy = RLDXSimPolicyWrapper(policy, strict=False)
    else:
        # Pure HuggingFace-style: one image off disk + an instruction.
        from PIL import Image

        img = np.asarray(Image.open(args.image).convert("RGB"))
        # 256x256 is what the GR1 datasets use; the processor also resizes, but
        # matching here keeps things obvious.
        if img.shape[:2] != (256, 256):
            img = np.asarray(Image.fromarray(img).resize((256, 256)))
        obs = build_nested_obs(img, args.instruction)

    # 3. Infer. Returns (action_dict, info_dict). Actions are ALREADY
    #    denormalized into physical units (radians / hand DOF), 16 steps ahead.
    action, info = policy.get_action(obs)

    # 4. Inspect. Keys depend on path: nested -> "left_arm"; sim wrapper ->
    #    "action.left_arm". Both carry shape (B=1, T=16, D).
    print("=== action chunk ===")
    for k, v in action.items():
        print(f"{k:28s} shape={v.shape} dtype={v.dtype}")

    # First action to send to the robot RIGHT NOW = timestep 0 of the chunk.
    arm_key = "left_arm" if "left_arm" in action else "action.left_arm"
    print("\nfirst left_arm command (step 0):", action[arm_key][0, 0])
    np.savez("gr1_action_chunk.npz", **{k: v for k, v in action.items()})
    print("\nsaved full chunk -> gr1_action_chunk.npz")


if __name__ == "__main__":
    main()
