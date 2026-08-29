"""Fused Triton kernels for TurboQuant_mse (Algorithm 1).

Each program processes a block of BLOCK_M input vectors at once, using tl.dot
for the D x D rotation matmul (D is small enough -- 64/128 in this package --
that the whole rotation matrix fits in one resident tile; only the batch (M)
dimension is blocked/gridded). The previous version used one program per
vector with a Python-level `for j in range(D)` loop that did the matmul as D
serial O(D) reductions -- an O(D^2) serial anti-pattern that made the kernel
backend dramatically slower than native's single cuBLAS matmul. This version
keeps the whole computation in one Triton kernel (same fusion goal, no
materialized intermediates) but does the matmul as tl.dot the way Triton is
actually meant to run one.

Empirically (on this machine's triton-windows build), the default
compiler-picked num_warps for these small D x D tl.dot kernels was unstable
across constexpr specializations, and the best fixed value differs by kernel:

- `_mse_quantize_kernel` (has the N_CENTROIDS argmin loop on top of the
  tl.dot) was ~4-6x slower at D=128, N_CENTROIDS=2 (bits=1) than at
  N_CENTROIDS=4/8/16 under the compiler's default heuristic, purely from the
  warp-count pick, not extra work; num_warps=8 was the fastest, stable choice
  across every N_CENTROIDS at both D=64 and D=128.
- `_mse_dequantize_kernel` (no loop, just a gather + one tl.dot) was
  consistently ~4-5x slower than native at D=128 with num_warps=8 (or the
  compiler default), regardless of bits; num_warps=16 fixed it to ~native
  speed or better at every bits/D combination tested, while num_warps=8 was
  occasionally worse than the default (multi-ms spikes at some N_CENTROIDS).

Both were verified via `examples/run_perf_benchmark.py` to match or beat
native across every bits/head_dim config.

Note: `tl.dot` requires each operand dimension to be at least 16 elements.
`d < 16` (untested here, and not used by this package's default configs of
64/128) will fail at Triton compile time on `backend="kernel"`, unlike the
previous per-row-loop kernel, which had no such floor.
"""

import torch
import triton
import triton.language as tl

_BLOCK_M = 64


@triton.jit
def _mse_quantize_kernel(
    x_ptr, rotation_ptr, centroids_ptr, indices_ptr, norm_ptr,
    stride_x_row, stride_rot_row,
    M, D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr, N_CENTROIDS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < M
    mask_d = offs_d < D
    mask_2d = mask_m[:, None] & mask_d[None, :]

    x = tl.load(x_ptr + offs_m[:, None] * stride_x_row + offs_d[None, :], mask=mask_2d, other=0.0)
    norm = tl.sqrt(tl.sum(x * x, axis=1))
    norm_safe = tl.maximum(norm, 1e-12)
    unit = x / norm_safe[:, None]

    # rot_t[i, j] = rotation[j, i]  (i.e. rotation.T), loaded directly via pointer arithmetic
    rot_t = tl.load(
        rotation_ptr + offs_d[:, None] + offs_d[None, :] * stride_rot_row,
        mask=mask_d[:, None] & mask_d[None, :], other=0.0,
    )
    y = tl.dot(unit, rot_t, input_precision="ieee")  # (BLOCK_M, BLOCK_D) = unit @ rotation.T, matches native rotate()

    best_idx = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.int32)
    best_dist = tl.full((BLOCK_M, BLOCK_D), float("inf"), dtype=tl.float32)
    for c in range(N_CENTROIDS):
        centroid_c = tl.load(centroids_ptr + c)
        dist = tl.abs(y - centroid_c)
        better = dist < best_dist
        best_idx = tl.where(better, c, best_idx)
        best_dist = tl.where(better, dist, best_dist)

    tl.store(indices_ptr + offs_m[:, None] * D + offs_d[None, :], best_idx, mask=mask_2d)
    tl.store(norm_ptr + offs_m, norm, mask=mask_m)


@triton.jit
def _mse_dequantize_kernel(
    indices_ptr, norm_ptr, rotation_ptr, centroids_ptr, out_ptr,
    stride_rot_row, stride_out_row,
    M, D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < M
    mask_d = offs_d < D
    mask_2d = mask_m[:, None] & mask_d[None, :]

    idx = tl.load(indices_ptr + offs_m[:, None] * D + offs_d[None, :], mask=mask_2d, other=0)
    y_hat = tl.load(centroids_ptr + idx, mask=mask_2d, other=0.0)
    norm = tl.load(norm_ptr + offs_m, mask=mask_m, other=0.0)

    # rot[j, k] = rotation[j, k], loaded as-is (no transpose)
    rot = tl.load(
        rotation_ptr + offs_d[:, None] * stride_rot_row + offs_d[None, :],
        mask=mask_d[:, None] & mask_d[None, :], other=0.0,
    )
    x_hat = tl.dot(y_hat, rot, input_precision="ieee")  # (BLOCK_M, BLOCK_D) = y_hat @ rotation, matches native unrotate()

    out = x_hat * norm[:, None]
    tl.store(out_ptr + offs_m[:, None] * stride_out_row + offs_d[None, :], out, mask=mask_2d)


def quantize(
    x: torch.Tensor, rotation: torch.Tensor, centroids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused normalize + rotate + per-coordinate argmin. Returns (indices, norm)."""
    orig_shape = x.shape
    d = orig_shape[-1]
    x_flat = x.reshape(-1, d).contiguous()
    n = x_flat.shape[0]
    n_centroids = centroids.shape[0]
    block_d = triton.next_power_of_2(d)
    block_m = _BLOCK_M
    rotation = rotation.to(x.device).contiguous()
    centroids = centroids.to(x.device).contiguous()

    indices = torch.empty((n, d), dtype=torch.int32, device=x.device)
    norm = torch.empty((n,), dtype=x.dtype, device=x.device)

    grid = (triton.cdiv(n, block_m),)
    _mse_quantize_kernel[grid](
        x_flat, rotation, centroids, indices, norm,
        x_flat.stride(0), rotation.stride(0),
        n, D=d, BLOCK_M=block_m, BLOCK_D=block_d, N_CENTROIDS=n_centroids,
        num_warps=8,
    )

    return indices.reshape(*orig_shape).long(), norm.reshape(*orig_shape[:-1])


def dequantize(
    indices: torch.Tensor, norm: torch.Tensor, rotation: torch.Tensor, centroids: torch.Tensor
) -> torch.Tensor:
    """Fused centroid-lookup + unrotate + rescale."""
    orig_shape = indices.shape
    d = orig_shape[-1]
    indices_flat = indices.reshape(-1, d).contiguous().int()
    norm_flat = norm.reshape(-1).contiguous().to(indices.device)
    n = indices_flat.shape[0]
    block_d = triton.next_power_of_2(d)
    block_m = _BLOCK_M
    rotation = rotation.to(indices.device).contiguous()
    centroids = centroids.to(indices.device).contiguous()

    out = torch.empty((n, d), dtype=centroids.dtype, device=indices.device)

    grid = (triton.cdiv(n, block_m),)
    _mse_dequantize_kernel[grid](
        indices_flat, norm_flat, rotation, centroids, out,
        rotation.stride(0), out.stride(0),
        n, D=d, BLOCK_M=block_m, BLOCK_D=block_d,
        num_warps=16,
    )

    return out.reshape(*orig_shape)
