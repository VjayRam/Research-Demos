"""Eq. (7) (spec Sec 8): packed-decode output decomposition.

    o_tau = sum_{t in T_kept \\ T_V16} a_{tau,t} * v_hat_t   (Zone A, quantized V)
          + sum_{t in T_V16}           a_{tau,t} * v_t       (Zone B, FP16 retained)
          + sum_{t in T_new}           a_{tau,t} * v_t       (Zone C, FP16 new)

This module is the NATIVE reference implementation: it dequantizes Zone A's
K rows into a real tensor before computing attention scores. This is
intentional here (correctness reference for Task 12's kernel to be checked
against) but is exactly what the fused kernel (rdkv.kernel.fused_decode)
must NOT do -- the whole point of Sec 8's algebraic rewrite is to never
materialize a dequantized FP16 K tile in the fast path.
"""

import torch

from .trizone import PackedCache


def packed_decode(
    packed: PackedCache,
    q_tau: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    sqrt_d: float,
) -> torch.Tensor:
    """q_tau: (d,) query for this decode step. k_new, v_new: (n_new, d) --
    Zone C, the new tokens generated since the last packing (n_new >= 1,
    the current step's own token at minimum). Returns o_tau: (d,).
    """
    d = q_tau.shape[0]
    n_kept = packed.zone_a_k.shape[0]

    # Reconstruct Zone A's K rows (native reference -- see module docstring
    # for why this is NOT how the fused kernel does it).
    if n_kept > 0:
        k_zone_a_dequant = packed.zone_a_k.float() * packed.k_scale + packed.k_zero_point
        # Undo the channel permutation applied at packing time so scores
        # align with q_tau's original channel order.
        inv_perm = torch.argsort(packed.k_channel_perm)
        k_zone_a = k_zone_a_dequant[:, inv_perm]
    else:
        k_zone_a = torch.empty(0, d)

    # Zone A's V rows: concatenate the {2,4,8}-bit sub-segments in the
    # dict's iteration order. NOTE: this concatenation order does NOT match
    # ascending-original-token-index order overall (tokens interleave across
    # bit-widths), so the K rows fed into the Zone A(V) score computation
    # must be gathered in this SAME per-bit-width order -- not in
    # zone_a_k's ascending-kept-token-index order -- or K/V rows for
    # different tokens get paired together.
    zone_a_v_parts = []
    k_for_zone_a_v_parts = []
    for bits, seg in packed.zone_a_v.items():
        if seg.shape[0] == 0:
            continue
        zone_a_v_parts.append(seg)
        # kept_token_idx and zone_a_v_token_idx[bits] are both ascending, so
        # searchsorted recovers each token's row index into zone_a_k/k_zone_a.
        rows = torch.searchsorted(packed.kept_token_idx, packed.zone_a_v_token_idx[bits])
        k_for_zone_a_v_parts.append(k_zone_a[rows])
    v_zone_a = torch.cat(zone_a_v_parts, dim=0) if zone_a_v_parts else torch.empty(0, d)
    k_for_zone_a_v = (
        torch.cat(k_for_zone_a_v_parts, dim=0) if k_for_zone_a_v_parts else torch.empty(0, d)
    )
    n_16bit = packed.zone_b_v.shape[0]
    n_non16bit = n_kept - n_16bit
    # Zone B's K rows are still selected via the boolean mask over zone_a_k's
    # ascending-kept-token order -- zone_b_v is already in that same
    # ascending order (built as v[zone_b_token_idx] with zone_b_token_idx a
    # subsequence of kept), so this side of the split needs no realignment.
    non16_selector = _non16bit_row_selector(packed)

    scores_parts = []
    values_parts = []

    if v_zone_a.shape[0] > 0:
        scores_a = (q_tau @ k_for_zone_a_v.T) / sqrt_d
        scores_parts.append(scores_a)
        values_parts.append(v_zone_a)

    if packed.zone_b_v.shape[0] > 0:
        k_zone_b = k_zone_a[~non16_selector]
        scores_b = (q_tau @ k_zone_b.T) / sqrt_d
        scores_parts.append(scores_b)
        values_parts.append(packed.zone_b_v)

    scores_c = (q_tau @ k_new.T) / sqrt_d
    scores_parts.append(scores_c)
    values_parts.append(v_new)

    all_scores = torch.cat(scores_parts, dim=0)
    all_values = torch.cat(values_parts, dim=0)
    all_weights = torch.softmax(all_scores, dim=0)

    return all_weights @ all_values


def _non16bit_row_selector(packed: PackedCache) -> torch.Tensor:
    """Boolean mask over zone_a_k's rows selecting the non-16-bit-V kept
    tokens (Zone A's K rows cover every kept token; only the subset with
    b_v != 16 pairs with Zone A's V sub-segments)."""
    return ~packed.zone_b_mask
