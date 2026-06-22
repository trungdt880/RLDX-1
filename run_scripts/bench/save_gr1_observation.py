#!/usr/bin/env python3
"""Capture one REAL GR-1 Tabletop observation from the live eval pipeline.

Stands up the same policy server the eval uses (RLDXPolicy + sim wrapper, sdpa
attention), but wraps ``get_action`` so the FIRST observation the mujoco client
sends is pickled to disk, then served normally. Run the ordinary rollout client
against it for a couple of steps and you get a genuine, in-distribution
observation dict (real camera frame + proprioceptive state + task language) to
replay for benchmarking / demo — no mujoco needed afterwards.

Usage (server side, main venv):
    RLDX_ATTN_IMPL=sdpa OBS_DUMP_PATH=run_scripts/bench/gr1_sample_obs.pkl \
        .venv/bin/python run_scripts/bench/save_gr1_observation.py \
        --model-path RLWRLD/RLDX-1-FT-GR1 --port 20111
"""
import os
import pickle
import sys

import tyro

from rldx.data.embodiment_tags import EmbodimentTag
from rldx.policy.rldx_policy import RLDXPolicy, RLDXSimPolicyWrapper
from rldx.policy.server_client import PolicyServer


def main(
    model_path: str = "RLWRLD/RLDX-1-FT-GR1",
    host: str = "127.0.0.1",
    port: int = 20111,
):
    dump_path = os.environ.get("OBS_DUMP_PATH", "run_scripts/bench/gr1_sample_obs.pkl")
    os.makedirs(os.path.dirname(dump_path) or ".", exist_ok=True)

    policy = RLDXPolicy(
        model_path=model_path,
        embodiment_tag=EmbodimentTag.GENERAL_EMBODIMENT,
        device="cuda",
    )
    wrapper = RLDXSimPolicyWrapper(policy, strict=True)

    # Wrap get_action so the first real observation is persisted, then continue
    # serving normally so the client's episode keeps running.
    _orig = wrapper.get_action
    state = {"saved": False}

    def capturing_get_action(observation, options=None):
        if not state["saved"]:
            with open(dump_path, "wb") as f:
                pickle.dump({"observation": observation, "options": options}, f)
            state["saved"] = True
            keys = sorted(observation.keys())
            print(f"[CAPTURE] saved real observation -> {dump_path}", flush=True)
            print(f"[CAPTURE] observation keys: {keys}", flush=True)
            sys.stdout.flush()
        return _orig(observation, options)

    # Bind BEFORE constructing the server: PolicyServer captures
    # ``policy.get_action`` by reference at register time.
    wrapper.get_action = capturing_get_action

    server = PolicyServer(policy=wrapper, host=host, port=port)
    print(f"[CAPTURE] server ready on {host}:{port}, dump -> {dump_path}", flush=True)
    server.run()


if __name__ == "__main__":
    tyro.cli(main)
