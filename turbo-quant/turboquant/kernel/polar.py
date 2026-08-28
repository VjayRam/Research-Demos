"""Fused Triton kernels for PolarQuant, one fused kernel per recursion level.

Each level pairs up coordinates, computes (radius, angle), and quantizes the
angle against that level's Lloyd-Max codebook in a single kernel -- matching
native's per-level computation but without materializing the intermediate
angle/radius tensors before quantization. Levels remain sequential (each
depends on the previous level's radii), matching native's control flow.
"""

import math

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

_TWO_PI = 2.0 * math.pi


@triton.jit
def _polar_quantize_level_kernel(
    v_ptr, centroids_ptr, angle_idx_ptr, radius_ptr,
    stride_v_row, stride_out_row,
    HALF: tl.constexpr, BLOCK_HALF: tl.constexpr, N_CENTROIDS: tl.constexpr, TWO_PI: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_HALF)
    mask = offs < HALF

    v0 = tl.load(v_ptr + row * stride_v_row + 2 * offs, mask=mask, other=0.0)
    v1 = tl.load(v_ptr + row * stride_v_row + 2 * offs + 1, mask=mask, other=0.0)

    radius = tl.sqrt(v0 * v0 + v1 * v1)
    angle = libdevice.atan2(v1, v0)
    angle = tl.where(angle < 0, angle + TWO_PI, angle)

    offs_c = tl.arange(0, N_CENTROIDS)
    centroids = tl.load(centroids_ptr + offs_c)

    diffs = tl.abs(angle[:, None] - centroids[None, :])
    idx = tl.argmin(diffs, axis=1)

    tl.store(angle_idx_ptr + row * stride_out_row + offs, idx, mask=mask)
    tl.store(radius_ptr + row * stride_out_row + offs, radius, mask=mask)


@triton.jit
def _polar_dequantize_level_kernel(
    angle_idx_ptr, radius_ptr, centroids_ptr, v_out_ptr,
    stride_in_row, stride_out_row,
    HALF: tl.constexpr, BLOCK_HALF: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_HALF)
    mask = offs < HALF

    idx = tl.load(angle_idx_ptr + row * stride_in_row + offs, mask=mask, other=0)
    angle = tl.load(centroids_ptr + idx, mask=mask, other=0.0)
    radius = tl.load(radius_ptr + row * stride_in_row + offs, mask=mask, other=0.0)

    v0 = radius * tl.math.cos(angle)
    v1 = radius * tl.math.sin(angle)

    tl.store(v_out_ptr + row * stride_out_row + 2 * offs, v0, mask=mask)
    tl.store(v_out_ptr + row * stride_out_row + 2 * offs + 1, v1, mask=mask)


def quantize_level(v: torch.Tensor, centroids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """v: (..., L). Returns (angle_indices, radius), each (..., L // 2)."""
    orig_shape = v.shape
    length = orig_shape[-1]
    half = length // 2
    v_flat = v.reshape(-1, length).contiguous()
    n = v_flat.shape[0]
    centroids = centroids.to(v.device).contiguous()
    n_centroids = centroids.shape[0]
    block_half = triton.next_power_of_2(half)

    angle_idx = torch.empty((n, half), dtype=torch.int32, device=v.device)
    radius = torch.empty((n, half), dtype=v.dtype, device=v.device)

    _polar_quantize_level_kernel[(n,)](
        v_flat, centroids, angle_idx, radius,
        v_flat.stride(0), angle_idx.stride(0),
        HALF=half, BLOCK_HALF=block_half, N_CENTROIDS=n_centroids, TWO_PI=_TWO_PI,
    )

    out_shape = (*orig_shape[:-1], half)
    return angle_idx.reshape(out_shape).long(), radius.reshape(out_shape)


def dequantize_level(angle_indices: torch.Tensor, radius: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """Inverse of quantize_level. angle_indices, radius: (..., L // 2). Returns (..., L)."""
    orig_shape = angle_indices.shape
    half = orig_shape[-1]
    idx_flat = angle_indices.reshape(-1, half).contiguous().to(radius.device).int()
    radius_flat = radius.reshape(-1, half).contiguous()
    n = idx_flat.shape[0]
    centroids = centroids.to(radius.device).contiguous()
    block_half = triton.next_power_of_2(half)

    out = torch.empty((n, 2 * half), dtype=radius.dtype, device=radius.device)

    _polar_dequantize_level_kernel[(n,)](
        idx_flat, radius_flat, centroids, out,
        idx_flat.stride(0), out.stride(0),
        HALF=half, BLOCK_HALF=block_half,
    )

    return out.reshape(*orig_shape[:-1], 2 * half)
