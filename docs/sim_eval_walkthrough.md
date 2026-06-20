# Simulator evaluation — process & data-flow walkthrough

Deep dive into what actually runs when you launch a sim eval: which file is
the entry point, how many processes spin up, what each one does, and how a
single observation becomes an action and gets executed.

This is the "trace it end to end" companion to
[`evaluation.md`](evaluation.md) (which is the how-to-run reference) and
[`architecture.md`](architecture.md) (what the model computes). GR1 tabletop
is used as the running example; every other sim benchmark follows the same
two-process shape, only the env id, venv path, and embodiment differ.

---

## 1. Entry point

One bash script orchestrates everything:
[`run_scripts/eval/gr1_tabletop/eval_gr1.sh`](../run_scripts/eval/gr1_tabletop/eval_gr1.sh).

Defaults: `MODEL_PATH=RLWRLD/RLDX-1-FT-GR1`, `PORT=20100`, `N_EPISODES=50`,
24 tabletop tasks, `n_envs=1`, `n_action_steps=16`, `max_episode_steps=720`.

What it does, in order (`eval_gr1.sh`):

1. Launches the **model server** once in the background (`:54`).
2. `sleep 30` (`:62`) — wait for the checkpoint to load before any client
   connects (cold load can take minutes; warm cache ~30 s).
3. Loops over the 24 tasks (`:64`), launching one **rollout client** per task,
   sequentially, each pointed at the same server.
4. `trap ... kill` (`:60`) tears the server down on exit.

---

## 2. Process map

```
eval_gr1.sh  (bash orchestrator)
   │
   ├─► PROC A — SERVER     uv run python rldx/eval/run_rldx_server.py        [main .venv, GPU]
   │      holds the RLDX-1 checkpoint, answers policy queries over ZeroMQ
   │      started ONCE, lives across all 24 tasks
   │
   └─► PROC B — CLIENT     robocasa_uv/.venv/bin/python rldx/eval/rollout_policy.py   [robocasa venv]
          drives the MuJoCo sim, sends observations, applies action chunks
          one per task, 24 sequential; with n_envs=1 the sim runs IN-PROCESS
                    │
              ZeroMQ TCP :20100  (msgpack-serialized, REQ/REP)
```

**At any instant: 2 processes** — one server, one client. The server is
persistent; the client is respawned 24 times (once per task).

### Why two processes (and two venvs)

The MuJoCo / robosuite stack and the `torch==2.7 + flash-attn` training stack
have conflicting pins, so they cannot share an environment. Splitting them
across processes — server in the main `.venv/`, client in the per-sim
`robocasa_uv/.venv/` — lets each keep its own dependency tree. It also isolates
the GPU (only the server touches CUDA) and lets one loaded model serve many
sequential sim clients without reloading.

| | PROC A — server | PROC B — client |
|---|---|---|
| File | `rldx/eval/run_rldx_server.py` | `rldx/eval/rollout_policy.py` |
| venv | main `.venv/` | `rldx/eval/sim/.../robocasa_uv/.venv/` |
| Owns | model, GPU, processor, norm stats | MuJoCo sim, video recording, CSV |
| Lifetime | once, all tasks | one per task (×24) |
| Imports `rldx.model`? | yes | no (only a thin ZMQ client) |
| Count | 1 | 1 (+ `n_envs` sim workers if async) |

---

## 3. PROC A — the model server

### Lifecycle (runs once)

`rldx/eval/run_rldx_server.py:main`:

1. `tyro.cli(ServerConfig)` parses flags (`:219`).
2. `RLDXPolicy(...)` (`:159`) → `PolicyLoader` does
   `AutoConfig` / `AutoModel.from_pretrained(torch_dtype=bfloat16)`, loads the
   processor, the per-embodiment encoder/decoder heads, and the **normalization
   statistics** baked into the checkpoint. Model goes to GPU, `.eval()`.
3. `RLDXSimPolicyWrapper(policy)` (`:204`, because `--use-sim-policy-wrapper`)
   — translates the *flat* sim observation layout to the *nested* layout the
   model expects, and the action chunk back. **Always on for sim eval.**
4. `PolicyServer(...).run()` (`:206`) binds a `zmq.REP` socket and blocks.

### Endpoints

Registered in `rldx/policy/server_client.py:96-104`: `ping`, `kill`,
`get_action`, `reset`, `get_modality_config`.

### Request loop (one query at a time)

`server_client.py:140-163` — strictly serial REQ/REP:

```python
message = self.socket.recv()                       # bytes
request = MsgSerializer.from_bytes(message)         # msgpack → dict; ndarrays via np.save blobs
endpoint = request.get("endpoint", "get_action")
result  = handler(**request.get("data", {}))        # → policy.get_action(observation, options)
self.socket.send(MsgSerializer.to_bytes(result))
```

Serialization is msgpack with a custom hook (`server_client.py:34-69`):
numpy arrays are `np.save`-d into a blob; bf16/fp16 torch tensors are up-cast
to float32 first so the wire stays lossless.

> New to ZeroMQ / TCP / ports / msgpack, or wondering how this differs from the
> WebSocket server (`run_rldx_server_pi.py`)? See
> [`inference_server.md` → Transport & serialization](inference_server.md#transport--serialization).

### `get_action` internals (the heart of it)

`policy.get_action` → `RLDXSimPolicyWrapper._get_action` (flat→nested,
`rldx_policy.py:427`) → `RLDXPolicy._get_action` → `PolicyRuntime.step`
(`rldx/policy/policy_runtime.py:107`). Four stages:

```
PolicyRuntime.step(request)                                   policy_runtime.py:107
  ├─ 1. _prepare_inputs        :164   unbatch B → VLAStepData → processor(...) → collate → bf16
  │        processor: aspect-area resize to 256², normalize state via ckpt
  │        norm stats, Qwen3-VL tokenize (image + text)
  ├─ 2. _inject_rtc_prefix     :223   no-op unless Real-Time Chunking enabled (GR1 eval: off)
  ├─ 3. _run_inference         :368   autocast(bf16) + inference_mode → model.get_action(**collated)
  └─ 4. _decode                :442   normalized action → physical units (denormalize via norm stats)
       returns (action_dict, {})
```

Stage 3 calls into the model proper, `RLDX.get_action`
(`rldx/model/core/rldx.py:1117`):

```
RLDX.get_action                                              rldx.py:1117
  ├─ prepare_input(inputs)                                  split backbone vs action inputs
  ├─ backbone(backbone_inputs)                              Qwen3-VL-8B → (vlm_hidden, cognition tokens)
  ├─ [memory]  _apply_memory_inference(...)                 only if use_memory (GR1 eval: off)
  └─ action_model.get_action(...)  →  get_action_with_features   rldx.py:738 / :490
        ├─ actions = randn(B, horizon=16, action_dim)       initial noise            :603
        ├─ timesteps = [0, 1/4, 2/4, 3/4, 1.0]              num_inference_timesteps=4 :616
        └─ for each step:  Euler integrate the flow         loop                      :642
              v = MSAT(state+noised-action stream, VL stream, [physics])   _dit_forward :623
              actions += dt * v                              :644
        returns {"action_pred": (B, 16, action_dim)}        normalized               :731
```

So one `get_action` = one Qwen3-VL forward + 4 MSAT (flow-matching) forwards,
producing a **16-step action chunk** in the model's normalized space. Stage 4
denormalizes it back to physical joint units before it goes on the wire.

Output dict keys are the GR1 modality groups, each shape `(B, 16, D)`:
`left_arm(7) left_hand(6) right_arm(7) right_hand(6) waist(3)`.

---

## 4. PROC B — the rollout client

### Lifecycle (once per task)

`rldx/eval/rollout_policy.py:run_rldx_sim_policy` (`:759`):

1. `get_embodiment_tag_from_env_name` maps a `gr1_unified/...` env to
   `GENERAL_EMBODIMENT` (`:771`).
2. `create_rldx_sim_policy(...)` (`:779`): because `--policy-client-host/port`
   are set, this returns a **`PolicyClient`** (`server_client.py:177`) — a thin
   ZMQ REQ stub. No model is loaded in this process.
3. `policy.get_modality_config()` (`:783`) is an RPC to the server; the returned
   `video` / `state` delta indices configure the `MultiStepWrapper`.
4. `run_rollout_gymnasium_policy(...)` (`:807`) builds the env and runs the loop.

### Env wrapper stack

Built inner → outer in `create_eval_env`:

```
GrootRoboCasaEnv            gymnasium_groot.py    MuJoCo sim; renders ego cam (process_img → 256²);
                                                  maps robot0_left→state.left_arm etc.;
                                                  language = "unlocked_waist: <task>"
  └─ VideoRecordingWrapper  rollout_policy.py:309 writes per-episode .mp4; deterministic reset seed
       └─ MultiStepWrapper  rollout_policy.py:322 stacks obs history (T); executes 16-action chunks
            └─ SyncVectorEnv(n=1)  :420            adds batch dim B=1; runs the env in-process
```

For `n_envs > 1`, the outermost layer is `AsyncVectorEnv(context="spawn")`
(`:422`) instead — see §6.

### Rollout loop

`run_rollout_gymnasium_policy` (`rollout_policy.py:438-496`):

```python
observations, _ = env.reset(); policy.reset()                       # :438  (reset is an RPC)
while completed_episodes < n_episodes:
    options = {"reset_memory": is_first_step, "session_ids": session_ids}   # :469
    actions, _ = policy.get_action(observations, options=options)   # :491  RPC → server → 16-step chunk
    observations, rewards, terms, truncs, infos = env.step(actions) # :496  MultiStepWrapper runs all 16
    # accumulate success/length; on episode end → record mp4 + CSV row + reset
```

The key timing fact: `policy.get_action` is **one network round-trip per
16 sim steps**. The server computes the whole chunk; `MultiStepWrapper.step`
plays it out against MuJoCo. Open-loop chunking, server-side.

---

## 5. MultiStepWrapper mechanics

`rldx/eval/sim/wrapper/multistep_wrapper.py`. Two jobs: assemble the
observation *history* the model wants, and execute action *chunks*.

### Observation history (the input side)

- Holds a ring buffer `self.obs = deque(maxlen=max_steps_needed + 1)`
  (`:184`). `max_steps_needed` is derived from the delta indices
  (`get_max_steps_needed`, `:220`).
- On `reset` the buffer is pre-filled with copies of the first obs (`:255`) so
  the very first inference has a full window.
- `_get_obs(video_delta_indices, state_delta_indices)` (`:329`) gathers the
  requested past frames. The delta indices are **0-indexed from the latest**,
  so the code uses `delta_indices - 1` to index the deque (`:348`), e.g.
  `[-4,-3,-2,-1,0]` → grabs the 5 most recent frames, latest last.
- For GR1, video and state both use `delta_indices=[0]` → **T=1**, the single
  latest frame/state. Language always takes only the latest (`:360`).
- Result per key: `np.stack(...)` along a new horizon axis → `(T,) + shape`.
  `SyncVectorEnv` then prepends `B` → the `(B, T, ...)` the policy validates.

### Action chunk execution (the output side)

`step(action)` (`:267`) receives the full chunk, values shaped
`(n_action_steps,) + action_shape`:

```python
for step in range(self.n_action_steps):       # n_action_steps = 16
    act = {key: value[step, :] for key, value in action.items()}   # one timestep
    if self.done[-1]: break                   # stop early on termination
    obs, reward, done, truncated, info = super().step(act)   # one MuJoCo step
    self.obs.append(obs)                       # feed the history buffer
# return the freshly-assembled history obs, aggregated reward/done, stacked info
```

So 1 chunk in → up to 16 MuJoCo steps → 1 new history obs out, plus
`terminate_on_success` early-out (`:323`). That single returned obs is what
the next loop iteration sends back to the server.

---

## 6. n_envs > 1 (parallel sim)

`eval_gr1.sh` uses `n_envs=1` (in-process `SyncVectorEnv`). Raising it switches
to `AsyncVectorEnv(context="spawn")` (`rollout_policy.py:422`), which forks
**`n_envs` sim worker subprocesses**, each running one full MuJoCo env. The
client collates their observations into a batched `(B=n_envs, ...)` and sends a
single batched RPC; the server batches inference across all of them. Each env
gets its own `session_id` (`:456`) so memory/RTC state stays isolated per
stream.

Process count then: `1 server + 1 client + n_envs sim workers`.

---

## 7. Full step, traced round-trip

```
[CLIENT] GrootRoboCasaEnv: render ego cam → process_img → 256²
         map robot0_left→state.left_arm(7), torso→state.waist(3), grippers→hands(6)
         language = "unlocked_waist: <task>"
[CLIENT] MultiStepWrapper stacks T=1 → SyncVectorEnv adds B=1 → flat obs dict
[CLIENT] PolicyClient.get_action → msgpack pack → ZMQ send :20100
   ───────────────────────────── TCP ─────────────────────────────►
[SERVER] recv → unpack → RLDXSimPolicyWrapper flat→nested
         PolicyRuntime: processor → Qwen3-VL → MSAT ×4 (flow matching) → 16-chunk → denormalize
         RLDXSimPolicyWrapper nested→flat
[SERVER] msgpack pack → ZMQ send
   ◄───────────────────────────── TCP ─────────────────────────────
[CLIENT] unpack → {action.left_arm:(1,16,7), action.waist:(1,16,3), ...}
[CLIENT] MultiStepWrapper.step: unmap action.left_arm→robot0_left,
         execute up to 16 MuJoCo steps → success flag + next history obs
```

---

## 8. Outputs

Per task, under `output_final/gr1_tabletop/<TAG>/<task>/`:

- `eval.log` — stdout/stderr for that task.
- `simulation_results.csv` — one row per episode (success / reward / steps /
  video filename), written by `_update_prediction_csv` (`rollout_policy.py:551`).
- `*.mp4` — one video per episode (`VideoRecordingWrapper`).
- Final success rate printed at the end of the run (`rollout_policy.py:898`).

See [`evaluation.md`](evaluation.md#collecting-results) for tally snippets.

---

## 9. File reference

| Concern | File:line |
|---|---|
| Orchestrator | `run_scripts/eval/gr1_tabletop/eval_gr1.sh` |
| Server entry | `rldx/eval/run_rldx_server.py:142` |
| ZMQ server loop + serialization | `rldx/policy/server_client.py:137`, `:34` |
| ZMQ client | `rldx/policy/server_client.py:177` |
| Inference pipeline (4 stages) | `rldx/policy/policy_runtime.py:107` |
| Flat ↔ nested translation | `rldx/policy/rldx_policy.py:248` (`RLDXSimPolicyWrapper`) |
| Model forward | `rldx/model/core/rldx.py:1117` |
| Flow-matching sampler | `rldx/model/core/rldx.py:490` (`get_action_with_features`) |
| Rollout entry | `rldx/eval/rollout_policy.py:759` |
| Env factory | `rldx/eval/rollout_policy.py:101` (`get_robocasa_env_fn`) |
| Env wrappers | `rldx/eval/rollout_policy.py:263` (`create_eval_env`) |
| Rollout loop | `rldx/eval/rollout_policy.py:438` |
| MultiStepWrapper | `rldx/eval/sim/wrapper/multistep_wrapper.py:139` |
| GR1 env / obs mapping | `external_dependencies/robocasa-gr1-tabletop-tasks/robocasa/utils/gym_utils/gymnasium_groot.py` |
| GR1 modality config | `rldx/configs/data/gr1_config.py` |
```
