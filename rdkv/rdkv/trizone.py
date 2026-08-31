"""Algorithm 1 Stage 4 (spec Sec 8/9): TriZone packing.

Packs a prefill K/V cache plus an Sec 7 AllocationResult into three storage
zones:

  Zone A -- old cache: retained K rows from T_kept (packed and
    quantized, per-channel affine, per pack_trizone below), plus V rows
    with b_v in {2,4,8}, grouped into uniform-bit sub-segments.
    DISCLOSED GAP: Zone A(V)'s rows are grouped by target bit-width but
    NOT actually quantized or byte-packed -- they're stored at full
    float32 precision. Only Zone A(K) is really quantized. See
    PackedCache.zone_a_v's field comment and rdkv/README.md's Phase 2
    section for the same disclosure; real V quantization/byte-packing is
    follow-up work.
  Zone B -- FP16, retained: V rows with b_v == 16. Their K rows still
    live in Zone A (K bit-widths follow the independent per-channel
    allocation, not b_v).
  Zone C -- FP16, new decode tokens -- NOT produced here. Zone C grows
    one entry per decode step and is owned by the decode loop
    (rdkv.decode), not by this one-shot post-prefill packing step.

This is memory-layout bookkeeping: it runs in plain PyTorch on CPU or GPU
and requires no Triton kernel.
"""

from dataclasses import dataclass, field

import torch

from .pipeline import AllocationResult

_ZONE_A_V_BITS = (2, 4, 8)


@dataclass
class PackedCache:
    zone_a_v: dict[int, torch.Tensor]  # bit_width -> (n_tokens_at_this_bit, d) V rows grouped by
    # target bit-width -- NOT actually quantized or byte-packed yet (a
    # disclosed gap: values are full-precision float32, only bucketed by
    # which bit-width they were assigned; see rdkv/README.md's Phase 2
    # section and module docstring below for the follow-up-work note)
    zone_a_k: torch.Tensor  # (n_kept, d) quantized K rows, channel-permuted
    zone_b_v: torch.Tensor  # (n_16bit, d) FP16 V rows
    zone_b_token_idx: torch.Tensor  # original token indices of Zone B rows
    zone_b_mask: torch.Tensor  # (n_kept,) bool, True where the kept token is 16-bit (Zone B)
    kept_token_idx: torch.Tensor  # (n_kept,) ascending original token indices, one per zone_a_k row
    zone_a_v_token_idx: dict[int, torch.Tensor]  # bit_width -> original token indices for that zone_a_v segment, same row order as zone_a_v[bits]
    k_channel_perm: torch.Tensor  # (d,) permutation applied to K's channel axis
    k_scale: torch.Tensor  # (d,) per-channel affine quantization scale s_c
    k_zero_point: torch.Tensor  # (d,) per-channel affine quantization zero point z_c


def _affine_quantize_channel(col: torch.Tensor, bits: int) -> tuple[torch.Tensor, float, float]:
    """Per-channel affine (asymmetric) quantization: k_hat = s*(k_tilde - z).
    Returns (quantized_int_tensor, scale, zero_point) such that
    dequant = quantized * scale + zero_point approximately reconstructs col.
    """
    if bits <= 0 or col.numel() == 0:
        return torch.zeros_like(col, dtype=torch.int64), 1.0, 0.0
    lo, hi = col.min().item(), col.max().item()
    if hi - lo < 1e-12:
        return torch.zeros_like(col, dtype=torch.int64), 1.0, lo
    n_levels = 2**bits
    scale = (hi - lo) / (n_levels - 1)
    zero_point = lo
    quantized = torch.clamp(torch.round((col - zero_point) / scale), 0, n_levels - 1).long()
    return quantized, scale, zero_point


def pack_trizone(k: torch.Tensor, v: torch.Tensor, allocation: AllocationResult) -> PackedCache:
    """k, v: (T, d) prefill K/V cache for one (layer, head) pair."""
    d = k.shape[1]
    kept = allocation.kept_tokens
    b_v_kept = allocation.b_v[kept]

    # Zone B: V rows with b_v == 16.
    zone_b_mask = b_v_kept == 16
    zone_b_token_idx = kept[zone_b_mask]
    zone_b_v = v[zone_b_token_idx]

    # Zone A(V): kept, non-16-bit V rows, split into per-bit-width sub-segments.
    zone_a_v: dict[int, torch.Tensor] = {}
    zone_a_v_token_idx: dict[int, torch.Tensor] = {}
    for bits in _ZONE_A_V_BITS:
        mask = b_v_kept == bits
        idx = kept[mask]
        zone_a_v[bits] = v[idx]
        zone_a_v_token_idx[bits] = idx

    # Zone A(K): every kept token's K row, with channels permuted by b_k ascending
    # (spec Sec 9: "sort channels by b_c^K into segments; permute q to match").
    k_channel_perm = torch.argsort(allocation.b_k)
    k_kept_permuted = k[kept][:, k_channel_perm]  # (n_kept, d)

    b_k_sorted = allocation.b_k[k_channel_perm]
    k_scale = torch.ones(d, device=k.device)
    k_zero_point = torch.zeros(d, device=k.device)
    zone_a_k_int = torch.zeros_like(k_kept_permuted, dtype=torch.int64)
    for c in range(d):
        bits_c = int(b_k_sorted[c].item())
        quantized, scale, zero_point = _affine_quantize_channel(k_kept_permuted[:, c], bits_c)
        zone_a_k_int[:, c] = quantized
        k_scale[c] = scale
        k_zero_point[c] = zero_point

    return PackedCache(
        zone_a_v=zone_a_v,
        zone_a_k=zone_a_k_int,
        zone_b_v=zone_b_v,
        zone_b_token_idx=zone_b_token_idx,
        zone_b_mask=zone_b_mask,
        kept_token_idx=kept,
        zone_a_v_token_idx=zone_a_v_token_idx,
        k_channel_perm=k_channel_perm,
        k_scale=k_scale,
        k_zero_point=k_zero_point,
    )
