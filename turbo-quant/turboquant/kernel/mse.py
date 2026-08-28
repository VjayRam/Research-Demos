"""Fused Triton kernels for TurboQuant_mse (Algorithm 1).

Each kernel processes one input vector per program, keeping the rotation
matrix and centroid array resident for the whole kernel so no intermediate
(normalized vector, rotated vector, per-coordinate distance) tensor is ever
written to global memory -- only the final (indices, norm) or x_hat tensors.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _mse_quantize_kernel(
    x_ptr, rotation_ptr, centroids_ptr, indices_ptr, norm_ptr,
    stride_x_row, stride_rot_row,
    D: tl.constexpr, BLOCK_D: tl.constexpr, N_CENTROIDS: tl.constexpr,
):
    row = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    x = tl.load(x_ptr + row * stride_x_row + offs_d, mask=mask_d, other=0.0)
    norm = tl.sqrt(tl.sum(x * x, axis=0))
    norm_safe = tl.maximum(norm, 1e-12)
    unit = x / norm_safe

    offs_c = tl.arange(0, N_CENTROIDS)
    centroids = tl.load(centroids_ptr + offs_c)

    for j in range(D):
        rot_row = tl.load(rotation_ptr + j * stride_rot_row + offs_d, mask=mask_d, other=0.0)
        y_j = tl.sum(rot_row * unit, axis=0)
        diffs = tl.abs(y_j - centroids)
        best_idx = tl.argmin(diffs, axis=0)
        tl.store(indices_ptr + row * D + j, best_idx)

    tl.store(norm_ptr + row, norm)


@triton.jit
def _mse_dequantize_kernel(
    indices_ptr, norm_ptr, rotation_ptr, centroids_ptr, out_ptr,
    stride_rot_row, stride_out_row,
    D: tl.constexpr, BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    idx = tl.load(indices_ptr + row * D + offs_d, mask=mask_d, other=0)
    y_hat = tl.load(centroids_ptr + idx, mask=mask_d, other=0.0)
    norm = tl.load(norm_ptr + row)

    for k in range(D):
        rot_col = tl.load(rotation_ptr + offs_d * stride_rot_row + k, mask=mask_d, other=0.0)
        x_hat_k = tl.sum(rot_col * y_hat, axis=0)
        tl.store(out_ptr + row * stride_out_row + k, x_hat_k * norm)


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
    rotation = rotation.to(x.device).contiguous()
    centroids = centroids.to(x.device).contiguous()

    indices = torch.empty((n, d), dtype=torch.int32, device=x.device)
    norm = torch.empty((n,), dtype=x.dtype, device=x.device)

    _mse_quantize_kernel[(n,)](
        x_flat, rotation, centroids, indices, norm,
        x_flat.stride(0), rotation.stride(0),
        D=d, BLOCK_D=block_d, N_CENTROIDS=n_centroids,
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
    rotation = rotation.to(indices.device).contiguous()
    centroids = centroids.to(indices.device).contiguous()

    out = torch.empty((n, d), dtype=centroids.dtype, device=indices.device)

    _mse_dequantize_kernel[(n,)](
        indices_flat, norm_flat, rotation, centroids, out,
        rotation.stride(0), out.stride(0),
        D=d, BLOCK_D=block_d,
    )

    return out.reshape(*orig_shape)
