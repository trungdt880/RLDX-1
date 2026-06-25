# SPDX-License-Identifier: Apache-2.0
"""Save incoming camera frames from a policy's observations to disk.

Shared by the debug obs-server and the real inference server (opt-in via
--save-frames). Saves every ``video.*`` frame as a PNG (npy fallback if Pillow
is missing). Tolerant of unbatched shapes so it works before any shape adapter.
"""
from pathlib import Path
from typing import Any

import numpy as np

try:
    from PIL import Image

    _HAVE_PIL = True
except ImportError:
    _HAVE_PIL = False


def save_video_frames(observation: dict[str, Any], outdir: Path, step: int,
                      batch0_only: bool = True) -> int:
    """Save each ``video.*`` frame (batch 0, all timesteps) under outdir. Returns count.

    Accepts (B,T,H,W,C), (T,H,W,C) or (H,W,C); higher rank is squeezed from the front.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    n = 0
    for key, v in observation.items():
        if not (isinstance(key, str) and key.startswith("video.") and isinstance(v, np.ndarray)):
            continue
        arr = v
        while arr.ndim > 5:
            arr = arr[0]
        if arr.ndim == 3:
            arr = arr[None, None]
        elif arr.ndim == 4:
            arr = arr[None]
        B, T = arr.shape[0], arr.shape[1]
        cam = key.replace("video.", "")
        n_batch = 1 if batch0_only else B
        for b in range(n_batch):
            for t in range(T):
                frame = arr[b, t]
                suffix = f"_b{b}" if n_batch > 1 else ""
                fname = outdir / f"step{step:06d}_{cam}{suffix}_t{t}.png"
                if (_HAVE_PIL and frame.dtype == np.uint8 and frame.ndim == 3
                        and frame.shape[-1] in (1, 3)):
                    img = frame[..., 0] if frame.shape[-1] == 1 else frame
                    Image.fromarray(img).save(fname)
                else:
                    np.save(fname.with_suffix(".npy"), frame)
                n += 1
    return n


class FrameSavingPolicy:
    """Wraps a policy; saves video frames from each get_action observation."""

    def __init__(self, policy: Any, outdir: str = "./server_frames",
                 every: int = 1, max_saves: int | None = None):
        self.policy = policy
        self.outdir = Path(outdir)
        self.every = max(1, int(every))
        self.max_saves = max_saves
        self._calls = 0
        self._saved = 0

    def get_action(self, observation: dict[str, Any], options: dict[str, Any] | None = None):
        if (self._calls % self.every == 0
                and (self.max_saves is None or self._saved < self.max_saves)):
            try:
                n = save_video_frames(observation, self.outdir, self._calls)
                if n:
                    self._saved += 1
                    print(f"[frame-saver] step {self._calls}: saved {n} frame(s) "
                          f"-> {self.outdir}/step{self._calls:06d}_*")
            except Exception as e:  # never let logging break inference
                print(f"[frame-saver] WARNING: could not save frames: {e}")
        self._calls += 1
        return self.policy.get_action(observation, options)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.policy, name)
