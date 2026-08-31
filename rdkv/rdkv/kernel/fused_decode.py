"""Fused Triton kernel for RDKV's packed decode (spec Sec 8).

Fuses Zone A's algebraic K dequantization directly into the attention
score computation:

    q_tau^T k_hat_t = sum_c (s_c * q_{tau,c}) * k_tilde_{t,c}  + bias

where k_hat_{t,c} = s_c*k_tilde_{t,c} + z_c is the per-channel affine
dequantization actually used by rdkv.trizone.pack_trizone (dequant =
quantized*scale + zero_point, with zero_point stored in the column's
original value units -- see trizone.py's k_scale/k_zero_point and
_affine_quantize_channel), so
q_tau^T k_hat_t = sum_c s_c*q_{tau,c}*k_tilde_{t,c} + sum_c q_{tau,c}*z_c.
The second term is a single per-query-head bias (using the UNSCALED
q_{tau,c}, since z_c is already in original value units, not
scale-normalized), computed once and added to every score -- never
requiring a materialized FP16 K tile for Zone A. This is the module
Task 12's structural-memory test
(test_fused_kernel_does_not_materialize_dequantized_k_tile) checks.

One program per decode step (the batch here is n_kept -- the whole point
is a single query attending over the packed cache), block over n_kept.

NOTE on Zone A(V) alignment: Zone A's V rows (packed.zone_a_v) are stored
grouped by bit-width (dict keyed 2/4/8, iterated in that order), which is
NOT the same order as packed.zone_a_k's rows (plain ascending kept-token
order). rdkv.decode's native packed_decode fixes this by gathering K rows
per bit-width via torch.searchsorted(packed.kept_token_idx,
packed.zone_a_v_token_idx[bits]) so K/V rows line up per original token.
Since zone_a_scores[i] here is in exact 1:1 correspondence with
zone_a_k's rows (scores[i] is the score for the token at
kept_token_idx[i]), the same realignment is applied to the *scores*
tensor below, mirroring rdkv.decode.packed_decode's fix exactly.
"""

import torch
import triton
import triton.language as tl

_BLOCK_N = 128


@triton.jit
def _fused_score_kernel(
    q_scaled_ptr, k_tilde_ptr, bias_ptr, scores_ptr,
    stride_k_row,
    N, D: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """scores[n] = sum_d q_scaled[d] * k_tilde[n, d] + bias, for a block of
    N (Zone A's kept-token) rows. q_scaled[d] = s_d * q_tau[d] is
    precomputed on the host (cheap, O(d)) so the kernel's inner loop is a
    a plain dot product against the still-quantized-integer k_tilde -- no
    K dequantization happens inside or outside this kernel. bias =
    sum_d q_tau[d] * z_d (UNSCALED q, since dequant = s*k_tilde + z uses
    z in original value units, not scale-normalized -- see module
    docstring)."""
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    mask_n = offs_n < N
    mask_d = offs_d < D
    mask_2d = mask_n[:, None] & mask_d[None, :]

    q_scaled = tl.load(q_scaled_ptr + offs_d, mask=mask_d, other=0.0)
    # k_tilde is loaded straight from the native quantized-integer buffer
    # (int64) -- the int->float cast happens HERE, inside the kernel, on
    # the loaded register tile only. No fp32/fp16 dequantized copy of the
    # (n_kept, d) tile is ever materialized in global memory.
    k_tilde_int = tl.load(
        k_tilde_ptr + offs_n[:, None] * stride_k_row + offs_d[None, :], mask=mask_2d, other=0
    )
    k_tilde = k_tilde_int.to(tl.float32)
    bias = tl.load(bias_ptr)

    raw_score = tl.sum(k_tilde * q_scaled[None, :], axis=1)
    score = raw_score + bias
    tl.store(scores_ptr + offs_n, score, mask=mask_n)


def _fused_zone_a_scores(q_tau: torch.Tensor, packed) -> torch.Tensor:
    """Computes Zone A's raw (pre-softmax, pre-sqrt_d) scores against ALL
    kept tokens' K rows, fused with dequantization per this module's
    docstring. Returns a (n_kept,) tensor, in zone_a_k's row order (i.e.
    scores[i] corresponds to the token at packed.kept_token_idx[i])."""
    d = q_tau.shape[0]
    n_kept = packed.zone_a_k.shape[0]
    device = q_tau.device

    if n_kept == 0:
        return torch.empty(0, device=device)

    # q must be permuted the same way K's channels were at packing time
    # (spec Sec 9: "permute q to match"), then pre-scaled by s_c.
    q_permuted = q_tau[packed.k_channel_perm]
    q_scaled = (q_permuted * packed.k_scale).contiguous()
    # bias = sum_c q_{tau,c} * z_c (UNSCALED q -- z_c is already in the
    # column's original value units, not scale-normalized; see module
    # docstring), a single per-query-head scalar, ADDED to every score.
    bias = (q_permuted * packed.k_zero_point).sum().reshape(1)

    # Keep zone_a_k in its native quantized-integer dtype -- the int->float
    # cast happens inside the Triton kernel (_fused_score_kernel), never
    # here on the host, so no materialized fp32/fp16 dequantized copy of
    # the (n_kept, d) tile is ever allocated.
    k_tilde = packed.zone_a_k.to(device).contiguous()
    scores = torch.empty(n_kept, device=device, dtype=torch.float32)
    block_d = triton.next_power_of_2(d)
    block_n = _BLOCK_N
    grid = (triton.cdiv(n_kept, block_n),)

    _fused_score_kernel[grid](
        q_scaled, k_tilde, bias, scores,
        k_tilde.stride(0),
        n_kept, D=d, BLOCK_N=block_n, BLOCK_D=block_d,
    )
    return scores


def fused_packed_decode(
    packed, q_tau: torch.Tensor, k_new: torch.Tensor, v_new: torch.Tensor, sqrt_d: float
) -> torch.Tensor:
    """GPU-only fused equivalent of rdkv.decode.packed_decode(..., backend="native").

    Zone A's scores are computed via _fused_zone_a_scores (no materialized
    dequantized K tile). Zone B (already FP16) and Zone C (new tokens) use
    ordinary matmuls, since they were never quantized in the first place --
    there is nothing to fuse for them.
    """
    d = q_tau.shape[0]
    device = q_tau.device
    # zone_a_scores[i] is the score for the token at packed.kept_token_idx[i]
    # -- same row order as packed.zone_a_k.
    zone_a_scores = _fused_zone_a_scores(q_tau, packed) / sqrt_d

    # Zone A(V): concatenate the {2,4,8}-bit sub-segments in the dict's
    # iteration order, same as rdkv.decode's native path. That order does
    # NOT match zone_a_scores' ascending-kept-token order overall (tokens
    # interleave across bit-widths), so the scores fed into the Zone A(V)
    # weighted sum must be gathered in this SAME per-bit-width order via
    # searchsorted against kept_token_idx -- mirroring rdkv.decode's fix
    # for k_for_zone_a_v, applied here to the precomputed scores tensor.
    zone_a_v_parts = []
    scores_for_zone_a_v_parts = []
    for bits, seg in packed.zone_a_v.items():
        if seg.shape[0] == 0:
            continue
        zone_a_v_parts.append(seg)
        rows = torch.searchsorted(packed.kept_token_idx, packed.zone_a_v_token_idx[bits])
        scores_for_zone_a_v_parts.append(zone_a_scores[rows])
    v_zone_a = torch.cat(zone_a_v_parts, dim=0) if zone_a_v_parts else torch.empty(0, d, device=device)
    scores_for_zone_a_v = (
        torch.cat(scores_for_zone_a_v_parts, dim=0)
        if scores_for_zone_a_v_parts
        else torch.empty(0, device=device)
    )

    scores_parts, values_parts = [], []
    if v_zone_a.shape[0] > 0:
        scores_parts.append(scores_for_zone_a_v)
        values_parts.append(v_zone_a)

    if packed.zone_b_v.shape[0] > 0:
        # Zone B's K rows are still selected via the boolean mask over
        # zone_a_scores' ascending-kept-token order -- zone_b_v is already
        # in that same ascending order (built as v[zone_b_token_idx] with
        # zone_b_token_idx a subsequence of kept), so this side of the
        # split needs no realignment (matches rdkv.decode's native path).
        scores_zone_b = zone_a_scores[packed.zone_b_mask]
        scores_parts.append(scores_zone_b)
        values_parts.append(packed.zone_b_v)

    scores_c = (q_tau @ k_new.T) / sqrt_d
    scores_parts.append(scores_c)
    values_parts.append(v_new)

    all_scores = torch.cat(scores_parts, dim=0)
    all_values = torch.cat(values_parts, dim=0)
    all_weights = torch.softmax(all_scores, dim=0)
    return all_weights @ all_values
