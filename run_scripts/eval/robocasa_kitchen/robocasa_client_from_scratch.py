#!/usr/bin/env python3
# ============================================================================
# RLDX-1 <-> RoboCasa: a from-scratch policy client (teaching edition)
# ============================================================================
#
# WHAT THIS IS
# ------------
# A *complete, standalone* example of driving a RoboCasa kitchen task with the
# RLDX-1 model, written from first principles so you can see every moving part.
# It deliberately does NOT import any `rldx` code except nothing at all -- it
# only speaks the on-the-wire protocol to a running RLDX-1 inference server.
# That keeps the two halves cleanly separated and lets this file run inside the
# lightweight robocasa sim venv (which has gymnasium/mujoco/robosuite but NOT
# the heavy model stack).
#
# THE BIG PICTURE (why two processes?)
# ------------------------------------
# The model is big (multi-billion-param VLA) and wants a modern CUDA/torch +
# transformers stack. The simulator (robosuite/mujoco/numba) wants an *older*
# numpy and a different torch. Those two dependency worlds do not fit in one
# venv. So RLDX-1 uses a client/server split:
#
#     +-------------------------+        ZeroMQ (TCP) + msgpack        +------------------------+
#     |   THIS PROCESS (client) |  ----  observation request  ----->  |   SERVER (model)       |
#     |   - robosuite / mujoco  |                                     |   run_rldx_server.py   |
#     |   - builds observations |  <----  action chunk reply  ------  |   - RLDX-1 on GPU      |
#     |   - steps the sim       |                                     |   - returns actions    |
#     +-------------------------+                                     +------------------------+
#
# The server is started separately, e.g.:
#
#     RLDX_PATCHEMBED_FP32=1 RLDX_ATTN_IMPL=sdpa CUDA_VISIBLE_DEVICES=0 \
#       uv run python rldx/eval/run_rldx_server.py \
#         --model-path RLWRLD/RLDX-1-FT-ROBOCASA \
#         --embodiment-tag GENERAL_EMBODIMENT \
#         --use-sim-policy-wrapper --host 127.0.0.1 --port 20200
#
# Then this client connects to 127.0.0.1:20200 and runs episodes.
#
# THE FIVE THINGS A CLIENT MUST GET RIGHT
# ---------------------------------------
#   1. WIRE FORMAT   : ZeroMQ REQ socket + msgpack, with a custom hook that
#                      (de)serializes numpy arrays. Must match the server byte
#                      for byte (see MsgpackZmqClient below).
#   2. THE CONTRACT  : ask the server `get_modality_config()` -- it tells you
#                      which camera/state/action keys it wants, and the temporal
#                      window (delta_indices) for each. NEVER hardcode this; read
#                      it at runtime so the client adapts to the loaded model.
#   3. KEY MAPPING   : the sim's observation keys are not always the model's
#                      keys (e.g. sim `video.res256_image_side_0` == model
#                      `left_view`). One small dict does the translation. THIS is
#                      the main thing you edit to add a new arm / new cameras.
#   4. TEMPORAL STACK: the model wants a short history of frames, sampled at the
#                      delta_indices (e.g. [-6,-4,-2,0] = "now and 3 strided past
#                      frames"). You keep a ring buffer of past sim obs and stack.
#   5. ACTION CHUNK  : the model returns a *chunk* of N future actions in one
#                      call (here N=16). You execute some/all of them open-loop,
#                      then ask again. This is what makes 0.16 Hz inference usable.
#
# RUN IT (headless -- the default, works over SSH)
# ------------------------------------------------
#   <sim-venv-python> robocasa_client_from_scratch.py \
#       --env-name robocasa_panda_omron/TurnOnMicrowave_PandaOmron_Env \
#       --host 127.0.0.1 --port 20200 --n-episodes 1
#
# where <sim-venv-python> is
#   rldx/eval/sim/robocasa/robocasa_uv/.venv/bin/python
#
# WATCH IT LIVE (GUI window -- only at a PHYSICAL monitor on the machine)
# ----------------------------------------------------------------------
# The RoboCasa gym env never opens a window (it builds robosuite with
# has_renderer=False and only renders cameras *offscreen* via EGL -- those
# offscreen frames are the model's observations). To actually *see* the robot
# we attach MuJoCo's built-in "passive viewer" to the same physics state. It
# opens its own GLFW window and does NOT disturb the EGL camera pipeline, so
# the model still gets its observations exactly as in headless mode.
#
# This requires a real display, so it works only when you are sitting at the
# machine's monitor (or via VNC) -- NOT over plain SSH. Add `--viewer`:
#
#   # at the physical console (a desktop session is running on :0 or :1)
#   export DISPLAY=:0            # or :1 -- whichever your session is
#   MUJOCO_GL=egl \              # keep EGL: cameras stay offscreen; the
#                                # viewer uses its own GLFW context
#   <sim-venv-python> robocasa_client_from_scratch.py \
#       --env-name robocasa_panda_omron/TurnOnMicrowave_PandaOmron_Env \
#       --host 127.0.0.1 --port 20200 --n-episodes 1 --viewer
#
# Notes:
#   * Close the window to abort the current episode early.
#   * Inference is slow (~0.3 s/chunk), so motion comes in 16-step bursts --
#     that is the model thinking, not a render stall.
#   * If no window appears: confirm DISPLAY is set and you are on the console,
#     not SSH. If GLFW errors under `MUJOCO_GL=egl`, unset MUJOCO_GL (with a
#     display, MuJoCo defaults to GLFW, which can still render cameras too).
#   * Only `--viewer` and the env touch the GUI; the headless path above is
#     unchanged, so leaving `--viewer` off behaves exactly as before.
# ============================================================================

from __future__ import annotations

import argparse
from collections import deque
import io
import os
from typing import Any

import msgpack
import numpy as np
import zmq


# ----------------------------------------------------------------------------
# 1. THE WIRE LAYER
# ----------------------------------------------------------------------------
# The server (rldx/policy/server_client.py :: PolicyServer) listens on a ZeroMQ
# REP socket. Every request is one msgpack blob:
#
#     {"endpoint": "<name>", "data": {<kwargs for the handler>}}
#
# and the reply is one msgpack blob (the handler's return value). msgpack has no
# native numpy type, so both sides agree on a tiny custom encoding:
#
#     numpy array  ->  {"__ndarray_class__": True, "as_npy": <bytes of np.save>}
#
# We MUST replicate that exactly, or the server cannot read our arrays (and we
# cannot read its action arrays). This is a faithful re-implementation of the
# server's MsgSerializer -- kept here so the client has zero `rldx` imports.
# ----------------------------------------------------------------------------
class MsgpackZmqClient:
    def __init__(self, host: str, port: int, timeout_ms: int = 60_000):
        self.ctx = zmq.Context.instance()
        self.host, self.port, self.timeout_ms = host, port, timeout_ms
        self._connect()

    def _connect(self) -> None:
        self.sock = self.ctx.socket(zmq.REQ)
        # RCVTIMEO makes a hung server raise instead of blocking forever.
        self.sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(f"tcp://{self.host}:{self.port}")

    # --- numpy <-> msgpack hooks (must mirror the server) -------------------
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
        # The server also sends ModalityConfig objects wrapped like this; we do
        # not need the class, just the plain dict of fields inside "as_json".
        if "__ModalityConfig_class__" in obj:
            return obj["as_json"]
        return obj

    # --- one round-trip RPC -------------------------------------------------
    def call(self, endpoint: str, data: dict | None = None, requires_input: bool = True) -> Any:
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data or {}
        self.sock.send(msgpack.packb(request, default=self._encode))
        try:
            raw = self.sock.recv()
        except zmq.error.Again:
            raise RuntimeError(
                f"No reply from server tcp://{self.host}:{self.port} within "
                f"{self.timeout_ms} ms. Is the server up and past model-load?"
            )
        reply = msgpack.unpackb(raw, object_hook=self._decode)
        # The server reports handler exceptions as {"error": "..."}.
        if isinstance(reply, dict) and "error" in reply:
            raise RuntimeError(f"Server error: {reply['error']}")
        return reply

    # --- the three endpoints we use -----------------------------------------
    def ping(self) -> Any:
        return self.call("ping", requires_input=False)

    def get_modality_config(self) -> dict:
        return self.call("get_modality_config", requires_input=False)

    def reset(self, options: dict | None = None) -> Any:
        return self.call("reset", {"options": options})

    def get_action(self, observation: dict, options: dict | None = None) -> tuple:
        # The server returns a 2-element list [action_dict, info_dict]; msgpack
        # gives us a list, we keep it as a tuple for clarity.
        action, info = self.call("get_action", {"observation": observation, "options": options})
        return action, info


# ----------------------------------------------------------------------------
# 2. THE CONTRACT  (parsed from get_modality_config)
# ----------------------------------------------------------------------------
# We ask the server what it wants instead of hardcoding it. A model fine-tuned
# for a different robot would answer differently, and this client would adapt.
# ----------------------------------------------------------------------------
class ModelContract:
    def __init__(self, modality_config: dict):
        v, s, a, lang = (modality_config[k] for k in ("video", "state", "action", "language"))
        self.video_keys: list[str] = list(v["modality_keys"])      # e.g. ['left_view', ...]
        self.video_delta: list[int] = list(v["delta_indices"])     # e.g. [-6,-4,-2,0]
        self.state_keys: list[str] = list(s["modality_keys"])
        self.state_delta: list[int] = list(s["delta_indices"])     # e.g. [0]
        self.action_keys: list[str] = list(a["modality_keys"])
        self.action_horizon: int = len(a["delta_indices"])         # e.g. 16
        self.language_keys: list[str] = list(lang["modality_keys"])

        # How many *consecutive* past sim frames we must retain so that the
        # most-negative delta index is reachable. For video [-6,-4,-2,0] that is
        # 0-(-6)+1 = 7 frames. We keep one extra (matches the reference wrapper).
        span = lambda d: (max(d) - min(d) + 1)  # noqa: E731
        self.history_len: int = max(span(self.video_delta), span(self.state_delta)) + 1

    def describe(self) -> str:
        return (
            f"  video  : keys={self.video_keys} delta={self.video_delta}\n"
            f"  state  : keys={self.state_keys} delta={self.state_delta}\n"
            f"  action : keys={self.action_keys} horizon={self.action_horizon}\n"
            f"  language: keys={self.language_keys}\n"
            f"  -> must retain {self.history_len} past sim frames"
        )


# ----------------------------------------------------------------------------
# 3. KEY MAPPING  (sim observation key  ->  model modality key)
# ----------------------------------------------------------------------------
# The RoboCasa GrootRoboCasaEnv emits long camera names; the model uses short
# canonical names. State / language keys happen to match 1:1 for this robot, so
# only the cameras need translating.
#
# *** ADDING A NEW ARM / NEW CAMERAS STARTS HERE ***
# If your new robot exposes a wrist cam under, say, "video.res256_image_wrist_1",
# and the model was trained to call it "wrist_view", add that pair below. If a
# state key differs, add a STATE map the same way. That is the whole change on
# the client side; everything else is schema-driven.
# ----------------------------------------------------------------------------
CANONICAL_VIDEO_TO_SIM = {
    "left_view": "video.res256_image_side_0",
    "right_view": "video.res256_image_side_1",
    "wrist_view": "video.res256_image_wrist_0",
    # "ego_view": "video.ego_view_pad_res256_freq20",  # (mobile-manip / GR-1)
}


def sim_key_for_video(canonical: str) -> str:
    """Translate a model camera name to the sim's observation key."""
    # Fall back to "video.<canonical>" so a model whose key already matches the
    # sim (a future, cleanly-named robot) works with no map entry.
    return CANONICAL_VIDEO_TO_SIM.get(canonical, f"video.{canonical}")


# ----------------------------------------------------------------------------
# 4. OBSERVATION ASSEMBLY  (ring buffer + temporal stacking + batch dim)
# ----------------------------------------------------------------------------
# The server (running with --use-sim-policy-wrapper, strict validation ON)
# requires, for every get_action call:
#   video.<key> : uint8,   shape (B, T_video, H, W, C),  C == 3
#   state.<key> : float32, shape (B, T_state, D)
#   <lang key>  : list[str] of length B
# Here B (batch) == 1 because we run a single environment.
#
# T_video / T_state come from the delta_indices. We sample the ring buffer at
# those (strided, non-positive) offsets. Index convention -- copied from the
# reference MultiStepWrapper -- is "delta d maps to ring position d-1", so the
# newest frame (d=0) is ring[-1].
# ----------------------------------------------------------------------------
class ObsHistory:
    def __init__(self, contract: ModelContract):
        self.contract = contract
        self.buf: deque[dict] = deque(maxlen=contract.history_len)

    def reset(self, first_obs: dict) -> None:
        # At t=0 we have no past, so pad the whole window with the first frame.
        self.buf.clear()
        for _ in range(self.contract.history_len):
            self.buf.append(first_obs)

    def push(self, obs: dict) -> None:
        self.buf.append(obs)

    def _gather(self, sim_key: str, delta_indices: list[int]) -> np.ndarray:
        # ring[d-1] for each delta d, then stack on a new time axis -> (T, ...)
        frames = [self.buf[d - 1][sim_key] for d in delta_indices]
        return np.stack(frames, axis=0)

    def build_request(self, task_text: str) -> dict:
        c = self.contract
        obs: dict[str, Any] = {}

        # --- cameras: (T,H,W,C) -> add batch -> (1,T,H,W,C), keep uint8 -------
        for cam in c.video_keys:
            sim_key = sim_key_for_video(cam)
            stacked = self._gather(sim_key, c.video_delta)            # (T,H,W,C) uint8
            obs[f"video.{cam}"] = stacked[None].astype(np.uint8, copy=False)

        # --- state: (T,D) -> (1,T,D), float32 --------------------------------
        for sk in c.state_keys:
            sim_key = f"state.{sk}"                                   # 1:1 for this robot
            stacked = self._gather(sim_key, c.state_delta)            # (T,D) float32
            obs[sk if sk.startswith("state.") else f"state.{sk}"] = (
                stacked[None].astype(np.float32, copy=False)
            )

        # --- language: a single string per batch element ----------------------
        # delta is [0] (just "now"); the server wants list[str] of length B==1.
        for lk in c.language_keys:
            obs[lk] = [str(task_text)]

        return obs


# ----------------------------------------------------------------------------
# 5. ACTION DECODING  (model action chunk  ->  per-step sim action dict)
# ----------------------------------------------------------------------------
# The server returns, for each model action key, an array shaped (B, T, D) where
# T == action_horizon (16). The sim's action_space is a Dict with the SAME keys
# (action.end_effector_position, ...). To execute step t we take batch 0, time t
# of every key. The sim's internal key_converter handles squashing the
# continuous gripper_close/control_mode outputs into its Discrete slots.
# ----------------------------------------------------------------------------
def action_chunk_to_steps(action: dict, n_steps: int) -> list[dict]:
    # action[key] : (B=1, T, D). Drop batch -> (T, D), then slice per step.
    steps = []
    horizon = min(n_steps, *(v.shape[1] for v in action.values()))
    for t in range(horizon):
        steps.append({key: np.asarray(arr)[0, t] for key, arr in action.items()})
    return steps


# ----------------------------------------------------------------------------
# OPTIONAL LIVE GUI  (MuJoCo passive viewer)
# ----------------------------------------------------------------------------
# Reach past the gym wrappers to the raw mujoco.MjModel / MjData that robosuite
# is simulating, and hand them to MuJoCo's passive viewer. "Passive" means the
# viewer only *displays* -- we still drive physics via env.step() and call
# viewer.sync() to refresh the window. Resolve the handles AFTER reset(), since
# RoboCasa rebuilds the scene (new layout/objects) on every reset.
def open_viewer(env):
    import mujoco.viewer  # lazy: only imported when --viewer is used
    sim = env.unwrapped.env.sim          # robosuite MjSim
    model = sim.model._model             # mujoco.MjModel
    data = sim.data._data                # mujoco.MjData
    return mujoco.viewer.launch_passive(model, data)


# ----------------------------------------------------------------------------
# THE CLOSED LOOP
# ----------------------------------------------------------------------------
def run_episode(env, client, contract, *, exec_horizon, max_steps, session_id, verbose,
                show_viewer=False):
    """One episode. Returns (success: bool, n_steps: int)."""
    history = ObsHistory(contract)

    obs, _ = env.reset()
    history.reset(obs)
    # Open the GUI window now -- after reset(), so it binds the freshly-built
    # scene. None in headless mode (the default).
    viewer = open_viewer(env) if show_viewer else None
    # The task instruction is part of the observation in RoboCasa.
    task_text = obs[contract.language_keys[0]]
    if verbose:
        print(f"    task: {task_text!r}")

    # Tell the server this is a fresh episode (clears any per-session memory the
    # model keeps). reset_memory is a per-batch-element mask; we have B==1.
    client.reset(options={"session_ids": [session_id]})

    success, steps, first_call = False, 0, True
    try:
        while steps < max_steps:
            # (a) assemble the observation the model expects, from our ring buffer.
            request = history.build_request(task_text)

            # (b) ask the model for an action chunk. We pass reset_memory=True only
            #     on the first call of the episode (matches the reference rollout).
            options = {"reset_memory": [first_call], "session_ids": [session_id]}
            # breakpoint()
            action, _info = client.get_action(request, options=options)
            first_call = False

            # (c) execute the first `exec_horizon` actions of the chunk, open-loop,
            #     feeding each resulting frame back into the history buffer.
            for act in action_chunk_to_steps(action, exec_horizon):
                obs, _reward, terminated, truncated, info = env.step(act)
                history.push(obs)
                steps += 1

                # Refresh the GUI window (no-op when headless).
                if viewer is not None:
                    viewer.sync()
                    if not viewer.is_running():   # user closed the window
                        return success, steps

                # RoboCasa reports task success in info["success"] (may be bool or
                # array). Once true, the task is solved -- we can stop.
                s = info.get("success", False)
                success = success or bool(np.any(s))
                if success or terminated or truncated or steps >= max_steps:
                    break
            if success or terminated or truncated:
                break
    finally:
        if viewer is not None:
            viewer.close()

    return success, steps


def make_env(env_name: str, seed: int):
    # Imported lazily so `--help` works without a full sim install, and so the
    # heavy robosuite import only happens when we actually run.
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import gymnasium as gym
    import robocasa  # noqa: F401  (side-effect: base registration)
    import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401  (registers robocasa_panda_omron/*)

    return gym.make(env_name, enable_render=True, seed=seed)


def main():
    p = argparse.ArgumentParser(description="From-scratch RLDX-1 client for RoboCasa.")
    p.add_argument("--env-name", default="robocasa_panda_omron/TurnOnMicrowave_PandaOmron_Env")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=20200)
    p.add_argument("--n-episodes", type=int, default=1)
    p.add_argument("--exec-horizon", type=int, default=16,
                   help="How many actions of each 16-step chunk to execute before re-querying.")
    p.add_argument("--max-steps", type=int, default=720, help="Max sim steps per episode.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--viewer", action="store_true",
                   help="Open a live MuJoCo window. Needs a physical display "
                        "(monitor/VNC), not SSH. See the header docstring.")
    args = p.parse_args()

    # 1. connect + 2. read the contract -------------------------------------
    client = MsgpackZmqClient(args.host, args.port)
    print(f"[client] pinging server tcp://{args.host}:{args.port} ...")
    print(f"[client] ping -> {client.ping()}")
    contract = ModelContract(client.get_modality_config())
    print("[client] model contract:")
    print(contract.describe())

    # 3. build the env -------------------------------------------------------
    print(f"[client] creating env {args.env_name}")
    env = make_env(args.env_name, args.seed)

    # 4. run episodes --------------------------------------------------------
    successes = []
    for ep in range(args.n_episodes):
        session_id = f"{args.env_name}-ep{ep}"
        ok, steps = run_episode(
            env, client, contract,
            exec_horizon=args.exec_horizon,
            max_steps=args.max_steps,
            session_id=session_id,
            verbose=args.verbose,
            show_viewer=args.viewer,
        )
        successes.append(ok)
        print(f"[client] episode {ep}: success={ok} steps={steps}")

    env.close()
    print(f"[client] success rate: {np.mean(successes):.2f}  ({sum(successes)}/{len(successes)})")


if __name__ == "__main__":
    main()
