"""Sec 7 / Algorithm 1 Stages 1-3 (spec): the three-stage allocation
pipeline, run once per layer-head pair immediately after prefill.

Stage 4 (TriZone packing, Algorithm 1's final stage) is out of scope for
this phase -- see spec Sec 14, decision 1. This module stops at producing
the per-token V bit-widths and per-channel K bit-widths; packing them into
TriZone storage is Phase 2.
"""

from dataclasses import dataclass

import torch

from .mckp import mckp_bisect
from .weights import bennett_sigma, channel_weight_k, token_weight_v


@dataclass
class AllocationResult:
    """Output of RDKVAllocator.allocate for one (layer, head) pair."""

    b_v: torch.Tensor  # (T,) per-token V bit-widths, hardware set {0,2,4,8,16}
    b_k: torch.Tensor  # (d,) per-channel K bit-widths, hardware set {0,2,4,8,16}
    kept_tokens: torch.Tensor  # long tensor of token indices where b_v > 0
    w_t: torch.Tensor  # (T,) raw token weights (Eq. 2)
    w_c: torch.Tensor  # (d,) raw channel weights (Eq. 4)


class RDKVAllocator:
    """Orchestrates Stage 1 (weighting) -> Stage 2 (V allocation) ->
    Stage 3 (K allocation on kept tokens only), per spec Sec 7.

    Uses a fixed sigma_u = 1 for all units by default (uniform dynamic
    range assumption) since Phase 1 has no real per-unit dynamic-range
    estimation wired in yet; pass sigma_v / sigma_k to override.
    """

    def allocate(
        self,
        attn: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        b_tok: float,
        sigma_v: torch.Tensor | None = None,
        sigma_k: torch.Tensor | None = None,
        mckp_kwargs: dict | None = None,
    ) -> AllocationResult:
        """attn: (n_queries, T) post-softmax attention weights. q, k: (T, d).
        b_tok: per-head budget in FP16-equivalent tokens (spec Sec 7).
        """
        mckp_kwargs = mckp_kwargs or {}
        T = attn.shape[1]
        d = q.shape[1]

        # Stage 1: weight computation
        w_t = token_weight_v(attn)
        w_c = channel_weight_k(q, k)
        if sigma_v is None:
            sigma_v = torch.ones_like(w_t)
        if sigma_k is None:
            sigma_k = torch.ones_like(w_c)

        # Stage 2: V-side token allocation. B_head = 2*b_tok*d*16 (Sec 7);
        # B_V = B_K = B_head/2; B_bar_V = B_V/d in summed-bit-width units.
        b_head = 2.0 * b_tok * d * 16.0
        b_v_budget = b_head / 2.0
        b_v_bar = b_v_budget / d
        target_avg_v = b_v_bar / T
        b_v, _ = mckp_bisect(w_t, sigma_v, target_avg_bits=target_avg_v, **mckp_kwargs)
        kept_tokens = torch.nonzero(b_v > 0, as_tuple=True)[0]

        # Stage 3: K-side channel allocation, denominator rescaled by
        # |T_kept| (Sec 7). If every token was evicted, there is nothing
        # left to allocate K bits for; return an all-zero K allocation
        # rather than dividing by zero.
        n_kept = kept_tokens.shape[0]
        if n_kept == 0:
            b_k = torch.zeros(d, dtype=torch.long)
        else:
            b_k_budget = b_head / 2.0
            k_avg = b_k_budget / (n_kept * d)
            b_k, _ = mckp_bisect(w_c, sigma_k, target_avg_bits=k_avg, **mckp_kwargs)

        return AllocationResult(b_v=b_v, b_k=b_k, kept_tokens=kept_tokens, w_t=w_t, w_c=w_c)
