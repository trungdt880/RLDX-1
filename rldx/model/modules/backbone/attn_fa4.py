"""FlashAttention-4 (CuTeDSL) attention backend for the Qwen3-VL backbone.

FA4 (``pip install flash-attn-4``, import path ``flash_attn.cute``) is the
CuTeDSL rewrite of FlashAttention. It JIT-compiles a per-arch kernel at
runtime and supports Blackwell **including Jetson Thor** — its dispatch
(``flash_attn/cute/interface.py``) groups compute capability 11 (``sm_110``,
Thor) with 10 (datacenter ``sm_100``) and runs the Sm100 kernel. Requires
CUDA 13.

Why a custom attention function instead of just naming it
``"flash_attention_4"``:

* transformers special-cases any ``attn_implementation`` that *starts with*
  ``"flash_attention"`` and force-imports it through its string-keyed
  ``lazy_import_flash_attention`` at load time (see
  ``PreTrainedModel._check_and_adjust_attn_implementation``). That importer
  only knows ``flash_attention_2`` / ``flash_attention_3`` and would raise on
  ``"flash_attention_4"``. So the *registered* key here is :data:`FA4_ATTN_IMPL`
  (``"rldx_fa4"``), which deliberately does **not** start with
  ``"flash_attention"``. Users still select it with the friendly
  ``RLDX_ATTN_IMPL=flash_attention_4`` (see :mod:`rldx...backbone.adapter`),
  which the adapter maps onto this key.
* We still want flash-style masking (a 2D padding mask or ``None``, never a 4D
  causal mask), so we also register the key in ``ALL_MASK_ATTENTION_FUNCTIONS``
  pointing at transformers' own ``flash_attention_mask``.

The heavy lifting (transpose, un/pad, varlen cu_seqlens, causal handling) is
reused from transformers' ``_flash_attention_forward`` by passing FA4's funcs
as the ``implementation`` object (the "kernels fallback" branch of
``_lazy_imports`` simply does ``getattr(implementation, "flash_attn_func")``).

Importing this module is an idempotent side effect: it registers the backend
iff ``flash_attn.cute`` is importable. On non-Thor / FA4-less envs the import
is a no-op and the key is simply unavailable.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import torch


# Registered attention key. MUST NOT start with "flash_attention" (see module
# docstring). This is the value written into ``config._attn_implementation``.
FA4_ATTN_IMPL = "rldx_fa4"

# Friendly aliases a user may pass via ``RLDX_ATTN_IMPL`` to select FA4.
FA4_PUBLIC_ALIASES = frozenset({"flash_attention_4", "fa4", "flash_attn_4", FA4_ATTN_IMPL})

# Set True once registration succeeds (flash_attn.cute present).
FA4_AVAILABLE = False


def fa4_is_supported_here() -> bool:
    """True iff FA4 is importable AND the current GPU arch is in FA4's
    supported compute-capability set (8.x/9.x/10.x/11.x/12.x — Thor is 11.x).

    Cheap, side-effect-free probe used by the adapter's arch guard so an
    unsupported host falls back to SDPA instead of crashing at first forward.
    """
    if not FA4_AVAILABLE or not torch.cuda.is_available():
        return False
    major = torch.cuda.get_device_capability()[0]
    # FA4 cute interface asserts ``arch // 10 in [8, 9, 10, 11, 12]``.
    return major in (8, 9, 10, 11, 12)


# --------------------------------------------------------------------------- #
# FA4 cute func wrappers.
#
# These isolate the *pre-release* FA4 API surface to one place. FA4's public
# signatures are FA2-compatible per upstream docs
# (``flash_attn_func(q, k, v, causal=...)``), but FA4 is pre-release
# (``4.0.0b18``) — if a positional/kw name drifts, fix it HERE only. The thin
# wrappers also give transformers a stable, introspectable signature so its
# ``_lazy_define_process_function`` correctly decides which optional kwargs
# (dropout/window/softcap/...) the kernel supports — for FA4 it supports none
# of those, only ``causal`` + ``softmax_scale``, which are always forwarded.
# --------------------------------------------------------------------------- #
def _fa4_dense(query, key, value, *, softmax_scale=None, causal=False, **_ignored):
    """Dense (no-padding) attention. q/k/v: (batch, seqlen, nheads, headdim)."""
    from flash_attn.cute import flash_attn_func

    return flash_attn_func(query, key, value, softmax_scale=softmax_scale, causal=causal)


def _fa4_varlen(
    query,
    key,
    value,
    *,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    softmax_scale=None,
    causal=False,
    **_ignored,
):
    """Variable-length attention. q/k/v: (total_tokens, nheads, headdim)."""
    from flash_attn.cute import flash_attn_varlen_func

    return flash_attn_varlen_func(
        query,
        key,
        value,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
    )


# Object handed to transformers as the flash "implementation" — its
# ``_lazy_imports`` kernels-fallback branch does
# ``getattr(implementation, "flash_attn_func")`` / ``flash_attn_varlen_func``.
_FA4_IMPL = SimpleNamespace(flash_attn_func=_fa4_dense, flash_attn_varlen_func=_fa4_varlen)


def fa4_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Drop-in replacement for transformers' ``flash_attention_forward`` that
    routes through FlashAttention-4.

    Same calling convention as every entry in ``ALL_ATTENTION_FUNCTIONS``:
    q/k/v arrive transposed as ``(batch, num_heads, seq, head_dim)``. We hand
    everything to transformers' ``_flash_attention_forward`` but inject FA4's
    funcs via ``implementation=_FA4_IMPL``, so all the un/pad + varlen +
    cu_seqlens plumbing is reused verbatim.
    """
    from transformers.modeling_flash_attention_utils import _flash_attention_forward

    # ``_flash_attention_forward`` expects non-transposed (batch, seq, h, d).
    seq_len = query.shape[2]
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    # Honor an explicit is_causal (vision passes False); else the module's flag
    # (text decoder is causal). Default True is the safe causal-LM fallback.
    is_causal = kwargs.pop("is_causal", None)
    if is_causal is None:
        is_causal = getattr(module, "is_causal", True)

    attn_output = _flash_attention_forward(
        query,
        key,
        value,
        attention_mask,
        query_length=seq_len,
        is_causal=is_causal,
        dropout=dropout,
        softmax_scale=scaling,
        sliding_window=sliding_window,
        softcap=softcap,
        # FA4 is post-FA2.1; top-left masking is the legacy FA<2.1 quirk.
        use_top_left_mask=False,
        implementation=_FA4_IMPL,
        **kwargs,
    )
    return attn_output, None


def _register() -> None:
    """Register the FA4 backend + its mask builder, iff FA4 is importable.

    Idempotent and import-safe: silently no-ops when ``flash_attn.cute`` is
    absent (non-Thor / FA4-less environments), leaving ``FA4_ATTN_IMPL``
    simply unregistered.
    """
    global FA4_AVAILABLE
    try:
        import flash_attn.cute  # noqa: F401  (presence probe only)
    except Exception:
        return

    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, flash_attention_mask
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS.register(FA4_ATTN_IMPL, fa4_attention_forward)
    # Mask is built exactly as for flash_attention_2 (2D padding mask / None),
    # never a 4D causal mask — FA4 applies causality internally.
    ALL_MASK_ATTENTION_FUNCTIONS.register(FA4_ATTN_IMPL, flash_attention_mask)
    FA4_AVAILABLE = True


_register()
