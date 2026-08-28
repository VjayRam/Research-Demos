"""Fused Triton kernels for TurboQuant_prod (Algorithm 2).

Reuses `kernel.mse` for the (bits-1)-bit MSE stage. Adds two fused kernels:
one for the residual -> QJL-projection -> sign-quantize step (used by
`quantize`), and one for the QJL correction -> add-to-x_hat step (used by
`dequantize`). `inner_product` uses a dedicated fused kernel that avoids
materializing the full rotated-y and QJL-projected-y intermediates.
"""

import torch
import triton
import triton.language as tl

from . import mse as kernel_mse  # noqa: F401  (re-exported for the mse_stage.quantize/dequantize dispatch)


@triton.jit
def _qjl_project_sign_kernel(
    x_ptr, x_hat_ptr, qjl_ptr, signs_ptr, residual_norm_ptr,
    stride_x_row, stride_xhat_row, stride_qjl_row,
    D: tl.constexpr, BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    x = tl.load(x_ptr + row * stride_x_row + offs_d, mask=mask_d, other=0.0)
    x_hat = tl.load(x_hat_ptr + row * stride_xhat_row + offs_d, mask=mask_d, other=0.0)
    residual = x - x_hat
    residual_norm = tl.sqrt(tl.sum(residual * residual, axis=0))
    tl.store(residual_norm_ptr + row, residual_norm)

    for j in range(BLOCK_D):
        if j < D:
            qjl_row = tl.load(qjl_ptr + j * stride_qjl_row + offs_d, mask=mask_d, other=0.0)
            proj_j = tl.sum(qjl_row * residual, axis=0)
            sign_j = tl.where(proj_j >= 0, 1.0, -1.0)
            tl.store(signs_ptr + row * D + j, sign_j)


@triton.jit
def _qjl_correct_kernel(
    x_hat_mse_ptr, signs_ptr, residual_norm_ptr, qjl_ptr, out_ptr,
    correction_scale,
    stride_xhat_row, stride_qjl_row, stride_out_row,
    D: tl.constexpr, BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    signs = tl.load(signs_ptr + row * D + offs_d, mask=mask_d, other=0.0)
    residual_norm = tl.load(residual_norm_ptr + row)
    x_hat_mse = tl.load(x_hat_mse_ptr + row * stride_xhat_row + offs_d, mask=mask_d, other=0.0)

    for k in range(BLOCK_D):
        if k < D:
            qjl_col = tl.load(qjl_ptr + offs_d * stride_qjl_row + k, mask=mask_d, other=0.0)
            correction_k = tl.sum(qjl_col * signs, axis=0) * residual_norm * correction_scale
            x_hat_mse_k = tl.sum(tl.where(offs_d == k, x_hat_mse, 0.0), axis=0)
            tl.store(out_ptr + row * stride_out_row + k, x_hat_mse_k + correction_k)


@triton.jit
def _prod_inner_product_kernel(
    y_ptr, indices_ptr, centroids_ptr, norm_ptr,
    qjl_ptr, signs_ptr, residual_norm_ptr,
    rotation_ptr, out_ptr,
    correction_scale,
    stride_y_row, stride_rot_row, stride_qjl_row,
    D: tl.constexpr, BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    y = tl.load(y_ptr + row * stride_y_row + offs_d, mask=mask_d, other=0.0)
    idx = tl.load(indices_ptr + row * D + offs_d, mask=mask_d, other=0)
    y_hat = tl.load(centroids_ptr + idx, mask=mask_d, other=0.0)
    signs = tl.load(signs_ptr + row * D + offs_d, mask=mask_d, other=0.0)
    norm = tl.load(norm_ptr + row)
    residual_norm = tl.load(residual_norm_ptr + row)

    term1_acc = 0.0
    term2_acc = 0.0
    for j in range(BLOCK_D):
        if j < D:
            rot_row = tl.load(rotation_ptr + j * stride_rot_row + offs_d, mask=mask_d, other=0.0)
            rotated_y_j = tl.sum(rot_row * y, axis=0)
            y_hat_j = tl.sum(tl.where(offs_d == j, y_hat, 0.0), axis=0)

            qjl_row = tl.load(qjl_ptr + j * stride_qjl_row + offs_d, mask=mask_d, other=0.0)
            y_proj_j = tl.sum(qjl_row * y, axis=0)
            signs_j = tl.sum(tl.where(offs_d == j, signs, 0.0), axis=0)

            term1_acc += y_hat_j * rotated_y_j
            term2_acc += y_proj_j * signs_j

    out = norm * term1_acc + residual_norm * correction_scale * term2_acc
    tl.store(out_ptr + row, out)


def quantize(x: torch.Tensor, mse_stage, qjl_matrix: torch.Tensor) -> dict:
    """mse_stage: a TurboQuantMSE instance already constructed with backend='kernel'."""
    orig_shape = x.shape
    d = orig_shape[-1]
    x_flat = x.reshape(-1, d).contiguous()
    n = x_flat.shape[0]
    block_d = triton.next_power_of_2(d)
    qjl_matrix = qjl_matrix.to(x.device).contiguous()

    indices, norm = mse_stage.quantize(x)
    x_hat = mse_stage.dequantize(indices, norm)
    x_hat_flat = x_hat.reshape(-1, d).contiguous()

    signs = torch.empty((n, d), dtype=x.dtype, device=x.device)
    residual_norm = torch.empty((n,), dtype=x.dtype, device=x.device)

    _qjl_project_sign_kernel[(n,)](
        x_flat, x_hat_flat, qjl_matrix, signs, residual_norm,
        x_flat.stride(0), x_hat_flat.stride(0), qjl_matrix.stride(0),
        D=d, BLOCK_D=block_d,
    )

    return {
        "indices": indices,
        "norm": norm,
        "qjl_signs": signs.reshape(*orig_shape),
        "residual_norm": residual_norm.reshape(*orig_shape[:-1]),
    }


def dequantize(compressed: dict, mse_stage, qjl_matrix: torch.Tensor, correction_scale: float) -> torch.Tensor:
    x_hat_mse = mse_stage.dequantize(compressed["indices"], compressed["norm"])
    orig_shape = x_hat_mse.shape
    d = orig_shape[-1]
    x_hat_mse_flat = x_hat_mse.reshape(-1, d).contiguous()
    signs_flat = compressed["qjl_signs"].reshape(-1, d).contiguous().to(x_hat_mse.device)
    residual_norm_flat = compressed["residual_norm"].reshape(-1).contiguous().to(x_hat_mse.device)
    n = x_hat_mse_flat.shape[0]
    block_d = triton.next_power_of_2(d)
    qjl_matrix = qjl_matrix.to(x_hat_mse.device).contiguous()

    out = torch.empty((n, d), dtype=x_hat_mse.dtype, device=x_hat_mse.device)

    _qjl_correct_kernel[(n,)](
        x_hat_mse_flat, signs_flat, residual_norm_flat, qjl_matrix, out,
        correction_scale,
        x_hat_mse_flat.stride(0), qjl_matrix.stride(0), out.stride(0),
        D=d, BLOCK_D=block_d,
    )

    return out.reshape(*orig_shape)


def inner_product(
    y: torch.Tensor, compressed: dict, mse_stage, qjl_matrix: torch.Tensor, correction_scale: float
) -> torch.Tensor:
    orig_shape = y.shape[:-1]
    d = y.shape[-1]
    y_flat = y.reshape(-1, d).contiguous()
    indices_flat = compressed["indices"].reshape(-1, d).contiguous().to(y.device).int()
    norm_flat = compressed["norm"].reshape(-1).contiguous().to(y.device)
    signs_flat = compressed["qjl_signs"].reshape(-1, d).contiguous().to(y.device)
    residual_norm_flat = compressed["residual_norm"].reshape(-1).contiguous().to(y.device)
    n = y_flat.shape[0]
    block_d = triton.next_power_of_2(d)
    qjl_matrix = qjl_matrix.to(y.device).contiguous()
    rotation = mse_stage.rotation.to(y.device).contiguous()
    centroids = mse_stage.codebook.centroids.to(y.device).contiguous()

    out = torch.empty((n,), dtype=y.dtype, device=y.device)

    _prod_inner_product_kernel[(n,)](
        y_flat, indices_flat, centroids, norm_flat,
        qjl_matrix, signs_flat, residual_norm_flat,
        rotation, out,
        correction_scale,
        y_flat.stride(0), rotation.stride(0), qjl_matrix.stride(0),
        D=d, BLOCK_D=block_d,
    )

    return out.reshape(*orig_shape)
