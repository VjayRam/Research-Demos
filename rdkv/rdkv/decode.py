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

    # Zone A's V rows: concatenate the {2,4,8}-bit sub-segments back in
    # kept-token order is not required for a decode-step sum (order-
    # independent softmax-weighted sum), so we just concatenate.
    zone_a_v_parts = [seg for seg in packed.zone_a_v.values() if seg.shape[0] > 0]
    v_zone_a = torch.cat(zone_a_v_parts, dim=0) if zone_a_v_parts else torch.empty(0, d)
    # k_zone_a rows correspond to ALL kept tokens (Zone A(K) covers every
    # kept token, per spec Sec 8's Zone A definition), while v_zone_a only
    # covers the non-16-bit subset -- so the K used for Zone A's V-weighted
    # sum must be restricted to the same non-16-bit token subset.
    n_16bit = packed.zone_b_v.shape[0]
    n_non16bit = n_kept - n_16bit
    # zone_a_k's row order is: all kept tokens in original packing order is
    # NOT guaranteed here since trizone.py builds zone_a_k from `kept`
    # directly (see Task 10) -- so we must select the same non-16-bit rows.
    # This selection mirrors trizone.py's zone_b_mask/zone_a_v construction.
    non16_selector = _non16bit_row_selector(packed)
    k_for_zone_a_v = k_zone_a[non16_selector]

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
