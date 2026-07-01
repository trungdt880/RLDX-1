#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# ============================================================================
# ALLEX <-> RoboCasa: closed-loop rollout client  (Phase 4 capstone)
# ============================================================================
#
# WHAT THIS IS
# ------------
# A standalone policy client that drives the gripperless ALLEX RoboCasa env
# (Phases 0-3) with the `RLDX-1-MT-ALLEX` checkpoint served over ZeroMQ. It runs
# INSIDE the robosuite sim venv (torch 2.5 / mujoco) and speaks only the on-the-
# wire protocol to a separate model-server process (torch 2.9 / GPU) -- the two
# torch worlds cannot share a venv. See the from-scratch kitchen client for the
# annotated protocol reference; this file is the ALLEX-specialized sibling.
#
# THE DEAL, PER THE FROZEN CONTRACT (run_scripts/eval/robocasa_allex/contract)
# ---------------------------------------------------------------------------
#   video : one mono ego cam `camera_ego_left`, uint8 (B,T,H,W,3).
#           server buffers the 4-frame [-6,-4,-2,0] memory; the client sends the
#           current frame(s) per the server-advertised video delta_indices (==[0]).
#   state : 6 joint groups `state.<group>`, float32 (B,T,D), absolute radians.
#   lang  : key `annotation.human.task_description`, list[str] (B,).
#   action: 6 groups `action.<group>`, (B,40,D) absolute joint targets. The GROOT
#           env.step() calls key_converter.unmap_action() internally
#           (gymnasium_groot.py:137), so we hand it the `action.<group>` dict
#           straight -- NO conversion to robot0_* on the client side.
#   memory: send reset_memory=[True] on the first call of each episode.
#
# GRADING: closed-loop STABILITY, not task success. The checkpoint is real-data-
# trained only; sim success is expected ~0 and is NOT claimed here.
#
# RUN (robosuite venv, headless):
#   MUJOCO_GL=egl <sim-venv-python> allex_rollout_client.py \
#       --host 127.0.0.1 --port 20250 --max-steps 80 --exec-horizon 16 \
#       --video-dir run_scripts/eval/robocasa_allex/rollout
# ============================================================================
from __future__ import annotations

import argparse
from collections import deque
import io
import os
import sys
import time
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
import zmq

# frozen contract (group order, dims, joint names + MJCF limits for sanity)
sys.path.insert(0, str(Path(__file__).resolve().parent / "contract"))
import allex_contract as C  # noqa: E402


# ----------------------------------------------------------------------------
# 1. WIRE LAYER  (ZeroMQ REQ + msgpack, numpy hook mirroring the server)
# ----------------------------------------------------------------------------
class MsgpackZmqClient:
    def __init__(self, host: str, port: int, timeout_ms: int = 120_000):
        self.ctx = zmq.Context.instance()
        self.host, self.port, self.timeout_ms = host, port, timeout_ms
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(f"tcp://{host}:{port}")

    @staticmethod
    def _encode(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            buf = io.BytesIO()
            np.save(buf, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
        return obj

    @staticmethod
    def _decode(obj: dict) -> Any:
        if not isinstance(obj, dict):
            return obj
        if "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        if "__ModalityConfig_class__" in obj:
            return obj["as_json"]
        return obj

    def call(self, endpoint: str, data: dict | None = None, requires_input: bool = True) -> Any:
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data or {}
        self.sock.send(msgpack.packb(request, default=self._encode))
        try:
            raw = self.sock.recv()
        except zmq.error.Again:
            raise RuntimeError(
                f"No reply from tcp://{self.host}:{self.port} within "
                f"{self.timeout_ms} ms. Is the server up and past model-load?"
            )
        reply = msgpack.unpackb(raw, object_hook=self._decode)
        if isinstance(reply, dict) and "error" in reply:
            raise RuntimeError(f"Server error: {reply['error']}")
        return reply

    def ping(self) -> Any:
        return self.call("ping", requires_input=False)

    def get_modality_config(self) -> dict:
        return self.call("get_modality_config", requires_input=False)

    def reset(self, options: dict | None = None) -> Any:
        return self.call("reset", {"options": options})

    def get_action(self, observation: dict, options: dict | None = None) -> tuple:
        action, info = self.call("get_action", {"observation": observation, "options": options})
        return action, info


# ----------------------------------------------------------------------------
# 2. CONTRACT read from the server (never hardcode; adapt to the loaded model)
# ----------------------------------------------------------------------------
class ModelContract:
    def __init__(self, mc: dict):
        v, s, a, lang = (mc[k] for k in ("video", "state", "action", "language"))
        self.video_keys = list(v["modality_keys"])
        self.video_delta = list(v["delta_indices"])
        self.state_keys = list(s["modality_keys"])
        self.state_delta = list(s["delta_indices"])
        self.action_keys = list(a["modality_keys"])
        self.action_horizon = len(a["delta_indices"])
        self.language_keys = list(lang["modality_keys"])
        span = lambda d: (max(d) - min(d) + 1)  # noqa: E731
        self.history_len = max(span(self.video_delta), span(self.state_delta)) + 1

    def describe(self) -> str:
        return (
            f"  video   : keys={self.video_keys} delta={self.video_delta}\n"
            f"  state   : keys={self.state_keys} delta={self.state_delta}\n"
            f"  action  : keys={self.action_keys} horizon={self.action_horizon}\n"
            f"  language: keys={self.language_keys}\n"
            f"  -> retain {self.history_len} past sim frames"
        )


# ----------------------------------------------------------------------------
# 3. OBSERVATION ASSEMBLY  (ring buffer + temporal stack + batch dim)
# ----------------------------------------------------------------------------
# For ALLEX the sim obs keys already MATCH the model modality keys 1:1:
#   video.camera_ego_left  and  state.<group>  -- no translation dict needed.
# We keep a short ring buffer and sample it at the server-advertised delta
# indices (index convention: delta d -> ring[d-1], newest d=0 -> ring[-1]).
class ObsHistory:
    def __init__(self, contract: ModelContract):
        self.c = contract
        self.buf: deque[dict] = deque(maxlen=contract.history_len)

    def reset(self, first_obs: dict) -> None:
        self.buf.clear()
        for _ in range(self.c.history_len):
            self.buf.append(first_obs)

    def push(self, obs: dict) -> None:
        self.buf.append(obs)

    def _gather(self, key: str, deltas: list[int]) -> np.ndarray:
        return np.stack([self.buf[d - 1][key] for d in deltas], axis=0)

    def build_request(self, task_text: str) -> dict:
        c, obs = self.c, {}
        for cam in c.video_keys:                                   # camera_ego_left
            stacked = self._gather(f"video.{cam}", c.video_delta)  # (T,H,W,C) uint8
            obs[f"video.{cam}"] = stacked[None].astype(np.uint8, copy=False)
        for g in c.state_keys:                                     # left_arm_joints, ...
            stacked = self._gather(f"state.{g}", c.state_delta)    # (T,D)
            obs[f"state.{g}"] = stacked[None].astype(np.float32, copy=False)
        for lk in c.language_keys:                                 # annotation.human.task_description
            obs[lk] = [str(task_text)]
        return obs


# ----------------------------------------------------------------------------
# 4. ACTION DECODING  (chunk -> per-step GROOT action dict)
# ----------------------------------------------------------------------------
# The server returns action.<group> shaped (B=1, T=40, D). To execute step t we
# take batch 0, time t of every key -> {action.<group>: (D,)}, which is exactly
# what the GROOT env.step expects (it unmap_action's internally).
def action_chunk_to_steps(action: dict, n_steps: int) -> list[dict]:
    horizon = min(n_steps, *(np.asarray(v).shape[1] for v in action.values()))
    return [{k: np.asarray(v)[0, t] for k, v in action.items()} for t in range(horizon)]


# --- joint-limit sanity: flat 48-vec bounds from the frozen MJCF/percentile ---
def _flat_action_48(step: dict) -> np.ndarray:
    """Concatenate the 6 action.<group> arrays in frozen GROUP_ORDER -> (48,)."""
    return np.concatenate([np.asarray(step[f"action.{g}"]).ravel() for g in C.GROUP_ORDER])


# ----------------------------------------------------------------------------
# 5. ENV
# ----------------------------------------------------------------------------
def make_env(env_name: str, seed: int):
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import gymnasium as gym
    import robocasa  # noqa: F401  (triggers ALLEX robot + key-converter registration)
    import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401  (registers robocasa_allex/*)

    return gym.make(env_name, enable_render=True, seed=seed)


# ----------------------------------------------------------------------------
# THE CLOSED LOOP
# ----------------------------------------------------------------------------
def run_episode(env, client, contract, *, exec_horizon, max_steps, session_id,
                video_frames):
    history = ObsHistory(contract)
    obs, _ = env.reset()
    history.reset(obs)

    base = env.unwrapped
    model = base.env.sim.model._model
    raw_data = base.env.sim.data._data

    task_text = obs[contract.language_keys[0]]
    print(f"    task: {task_text!r}")
    print(f"    reset: model.neq={model.neq} nu={model.nu} njnt={model.njnt}")

    # fresh server-side memory for this episode
    client.reset(options={"reset_memory": [True], "session_ids": [session_id]})

    steps, first_call = 0, True
    latencies, rewards = [], []
    max_abs_action = 0.0
    n_calls = 0
    stable = True
    while steps < max_steps:
        request = history.build_request(task_text)
        options = {"reset_memory": [first_call], "session_ids": [session_id]}
        t0 = time.perf_counter()
        action, _info = client.get_action(request, options=options)
        dt = time.perf_counter() - t0
        latencies.append(dt)
        n_calls += 1
        first_call = False

        chunk = action_chunk_to_steps(action, exec_horizon)
        # action magnitude read on the first executed step of the chunk
        a48 = _flat_action_48(chunk[0])
        max_abs_action = max(max_abs_action, float(np.max(np.abs(a48))))

        chunk_reward = []
        for act in chunk:
            obs, reward, terminated, truncated, info = env.step(act)
            history.push(obs)
            steps += 1
            chunk_reward.append(float(reward))
            rewards.append(float(reward))

            # save ego frame for the rollout video
            if video_frames is not None:
                video_frames.append(np.asarray(obs["video.camera_ego_left"]).copy())

            # --- stability invariants (the actual grading criteria) ---
            fin_r = np.isfinite(reward)
            fin_q = np.isfinite(raw_data.qpos).all() and np.isfinite(raw_data.qvel).all()
            neq_ok = int(model.neq) == 12
            if not (fin_r and fin_q and neq_ok):
                stable = False
                print(f"    !! step{steps}: STABILITY BREAK "
                      f"finite_reward={fin_r} finite_sim={fin_q} neq={model.neq}")
            if terminated or truncated or steps >= max_steps:
                break

        print(f"  [call {n_calls:>2} | step {steps:>3}] "
              f"lat={dt*1e3:7.1f}ms reward={np.mean(chunk_reward):+.4f} "
              f"term={bool(terminated)} trunc={bool(truncated)} "
              f"neq={model.neq} |a|max={np.max(np.abs(a48)):.3f}rad "
              f"finite={bool(np.isfinite(raw_data.qpos).all())}")
        if terminated or truncated:
            break

    return {
        "steps": steps,
        "n_calls": n_calls,
        "latencies": latencies,
        "rewards": rewards,
        "max_abs_action": max_abs_action,
        "stable": stable,
    }


def main():
    p = argparse.ArgumentParser(description="ALLEX RoboCasa closed-loop rollout client.")
    p.add_argument("--env-name", default="robocasa_allex/PnPCanToBowl_AllexRobot_Env")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=20250)
    p.add_argument("--n-episodes", type=int, default=1)
    p.add_argument("--exec-horizon", type=int, default=16,
                   help="Chunk steps to execute open-loop before re-querying (of 40).")
    p.add_argument("--max-steps", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--video-dir", default=None, help="If set, write an mp4 of the ego view.")
    args = p.parse_args()

    client = MsgpackZmqClient(args.host, args.port)
    print(f"[client] ping tcp://{args.host}:{args.port} -> {client.ping()}")
    contract = ModelContract(client.get_modality_config())
    print("[client] model contract:")
    print(contract.describe())

    print(f"[client] creating env {args.env_name}")
    env = make_env(args.env_name, args.seed)

    all_ok = True
    for ep in range(args.n_episodes):
        video_frames = [] if args.video_dir else None
        print(f"\n[client] ===== episode {ep} =====")
        res = run_episode(
            env, client, contract,
            exec_horizon=args.exec_horizon, max_steps=args.max_steps,
            session_id=f"{args.env_name}-ep{ep}", video_frames=video_frames,
        )
        lat = np.array(res["latencies"])
        print(f"\n[client] episode {ep} summary:")
        print(f"    steps run       : {res['steps']}")
        print(f"    server calls    : {res['n_calls']}")
        print(f"    latency ms      : mean={lat.mean()*1e3:.1f} "
              f"min={lat.min()*1e3:.1f} max={lat.max()*1e3:.1f}")
        print(f"    reward          : mean={np.mean(res['rewards']):+.4f} "
              f"(success NOT claimed; ~0 expected)")
        print(f"    max |action|    : {res['max_abs_action']:.3f} rad")
        print(f"    STABLE (finite + neq==12 all steps): {res['stable']}")
        all_ok = all_ok and res["stable"] and res["steps"] >= 1

        if args.video_dir and video_frames:
            _write_video(args.video_dir, ep, video_frames)

    env.close()
    print(f"\n[client] CLOSED-LOOP BRING-UP: {'PASS (stable harness)' if all_ok else 'FAIL'}")
    sys.exit(0 if all_ok else 1)


def _write_video(video_dir: str, ep: int, frames: list) -> None:
    os.makedirs(video_dir, exist_ok=True)
    out = os.path.join(video_dir, f"allex_rollout_ep{ep}.mp4")
    try:
        import imageio.v2 as imageio
        imageio.mimsave(out, [f.astype(np.uint8) for f in frames], fps=20)
        print(f"    video           : {out}  ({len(frames)} frames)")
    except Exception as e:  # noqa: BLE001
        # non-fatal: video is a nice-to-have
        npy = os.path.join(video_dir, f"allex_rollout_ep{ep}.npy")
        np.save(npy, np.stack(frames))
        print(f"    video encode failed ({e}); saved frames -> {npy}")


if __name__ == "__main__":
    main()
