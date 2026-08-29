"""Fused Triton kernels for TurboQuant_prod (Algorithm 2).

Reuses `kernel.mse` for the (bits-1)-bit MSE stage. Adds two fused kernels:
one for the residual -> QJL-projection -> sign-quantize step (used by
`quantize`), and one for the QJL correction -> add-to-x_hat step (used by
`dequantize`). `inner_product` uses a dedicated fused kernel that avoids
materializing the full rotated-y and QJL-projected-y intermediates.

Each kernel processes a block of BLOCK_M input vectors per program and uses
tl.dot for the D x D matmuls (QJL projection / rotation), mirroring
`kernel.mse`'s fix: the previous version did these matmuls as a Python-level
`for j in range(BLOCK_D): if j < D:` loop of D serial O(D) reductions per
program, which made the kernel backend dramatically slower than native's
single cuBLAS matmul.

`inner_product`'s fused kernel originally held both the D x D rotation and
D x D QJL matrix tiles resident at once (two tl.dot calls sharing one
program). At D=128 that overflows this GPU's shared memory (measured:
163840 bytes required vs a 101376-byte hardware limit) and made *every*
num_warps value raise `triton.runtime.errors.OutOfResources` -- a real crash,
not just a slowdown, and one the existing tests never caught because
`test_prod_kernel_inner_product_matches_native` only exercises D=64. Fixed by
splitting it into two single-matrix kernels (`_prod_inner_product_term1_kernel`,
`_prod_inner_product_term2_kernel`), each resident with only one D x D tile,
summed on the host afterward.

Known limitation: at D=128, `quantize`/`dequantize`'s two-D×D-matrix-per-block
shared-memory footprint is already at this GPU's per-block limit, so latency
here is 1.2-2.7x slower than native (still correct, just not equal-or-better
latency) -- documented as an accepted exception in the design spec's "Known
Limitations" section rather than silently absorbed. D=64 fully meets the
same-or-better requirement.

Like `kernel.mse`, num_warps is pinned per kernel rather than left to the
compiler's default heuristic -- empirically, that heuristic was wildly
unstable across constexpr specializations on this machine's triton-windows
build (single-digit-ms regressions from the "wrong" pick), and the best value
differs by kernel: `_qjl_project_sign_kernel`/`_qjl_correct_kernel` (single
tl.dot, no extra loop) do best at num_warps=16; the term1/term2 inner_product
kernels (single tl.dot, gather loads) also use num_warps=16. See
`kernel/mse.py`'s module docstring for the measured quantize-side details.
"""

import torch
import triton
import triton.language as tl

from . import mse as kernel_mse  # noqa: F401  (re-exported for the mse_stage.quantize/dequantize dispatch)

_BLOCK_M = 64


@triton.jit
def _qjl_project_sign_kernel(
    x_ptr, x_hat_ptr, qjl_ptr, signs_ptr, residual_norm_ptr,
    stride_x_row, stride_xhat_row, stride_qjl_row,
    M, D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < M
    mask_d = offs_d < D
    mask_2d = mask_m[:, None] & mask_d[None, :]

    x = tl.load(x_ptr + offs_m[:, None] * stride_x_row + offs_d[None, :], mask=mask_2d, other=0.0)
    x_hat = tl.load(x_hat_ptr + offs_m[:, None] * stride_xhat_row + offs_d[None, :], mask=mask_2d, other=0.0)
    residual = x - x_hat
    residual_norm = tl.sqrt(tl.sum(residual * residual, axis=1))
    tl.store(residual_norm_ptr + offs_m, residual_norm, mask=mask_m)

    # qjl_t[i, j] = qjl_matrix[j, i]  (i.e. qjl_matrix.T)
    qjl_t = tl.load(
        qjl_ptr + offs_d[:, None] + offs_d[None, :] * stride_qjl_row,
        mask=mask_d[:, None] & mask_d[None, :], other=0.0,
    )
    proj = tl.dot(residual, qjl_t, input_precision="ieee")  # (BLOCK_M, BLOCK_D) = residual @ qjl_matrix.T
    signs = tl.where(proj >= 0, 1.0, -1.0)

    tl.store(signs_ptr + offs_m[:, None] * D + offs_d[None, :], signs, mask=mask_2d)


@triton.jit
def _qjl_correct_kernel(
    x_hat_mse_ptr, signs_ptr, residual_norm_ptr, qjl_ptr, out_ptr,
    correction_scale,
    stride_xhat_row, stride_qjl_row, stride_out_row,
    M, D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < M
    mask_d = offs_d < D
    mask_2d = mask_m[:, None] & mask_d[None, :]

    signs = tl.load(signs_ptr + offs_m[:, None] * D + offs_d[None, :], mask=mask_2d, other=0.0)
    residual_norm = tl.load(residual_norm_ptr + offs_m, mask=mask_m, other=0.0)
    x_hat_mse = tl.load(
        x_hat_mse_ptr + offs_m[:, None] * stride_xhat_row + offs_d[None, :], mask=mask_2d, other=0.0
    )

    # qjl[j, k] = qjl_matrix[j, k], loaded as-is (no transpose)
    qjl = tl.load(
        qjl_ptr + offs_d[:, None] * stride_qjl_row + offs_d[None, :],
        mask=mask_d[:, None] & mask_d[None, :], other=0.0,
    )
    correction = tl.dot(signs, qjl, input_precision="ieee") * residual_norm[:, None] * correction_scale
    out = x_hat_mse + correction

    tl.store(out_ptr + offs_m[:, None] * stride_out_row + offs_d[None, :], out, mask=mask_2d)


@triton.jit
def _prod_inner_product_term1_kernel(
    y_ptr, indices_ptr, centroids_ptr, norm_ptr,
    rotation_ptr, out_ptr,
    stride_y_row, stride_rot_row,
    M, D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """term1 = norm * sum_j(y_hat_j * rotated_y_j), the pure-MSE-term half of
    the inner product. Kept as its own kernel (rather than fused with term2's
    QJL matmul) so only one D x D matrix tile (`rotation`) is ever resident --
    see the module docstring for why holding both D x D tiles at once
    overflows shared memory at D=128."""
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < M
    mask_d = offs_d < D
    mask_2d = mask_m[:, None] & mask_d[None, :]

    y = tl.load(y_ptr + offs_m[:, None] * stride_y_row + offs_d[None, :], mask=mask_2d, other=0.0)
    idx = tl.load(indices_ptr + offs_m[:, None] * D + offs_d[None, :], mask=mask_2d, other=0)
    y_hat = tl.load(centroids_ptr + idx, mask=mask_2d, other=0.0)
    norm = tl.load(norm_ptr + offs_m, mask=mask_m, other=0.0)

    rot_t = tl.load(
        rotation_ptr + offs_d[:, None] + offs_d[None, :] * stride_rot_row,
        mask=mask_d[:, None] & mask_d[None, :], other=0.0,
    )
    rotated_y = tl.dot(y, rot_t, input_precision="ieee")  # (BLOCK_M, BLOCK_D) = y @ rotation.T

    term1 = tl.sum(y_hat * rotated_y, axis=1) * norm
    tl.store(out_ptr + offs_m, term1, mask=mask_m)


@triton.jit
def _prod_inner_product_term2_kernel(
    y_ptr, qjl_ptr, signs_ptr, residual_norm_ptr, out_ptr,
    correction_scale,
    stride_y_row, stride_qjl_row,
    M, D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """term2 = residual_norm * correction_scale * sum_j(y_proj_j * signs_j),
    the QJL-correction half of the inner product. See
    `_prod_inner_product_term1_kernel`'s docstring for why this is a separate
    kernel rather than fused with term1's rotation matmul."""
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < M
    mask_d = offs_d < D
    mask_2d = mask_m[:, None] & mask_d[None, :]

    y = tl.load(y_ptr + offs_m[:, None] * stride_y_row + offs_d[None, :], mask=mask_2d, other=0.0)
    signs = tl.load(signs_ptr + offs_m[:, None] * D + offs_d[None, :], mask=mask_2d, other=0.0)
    residual_norm = tl.load(residual_norm_ptr + offs_m, mask=mask_m, other=0.0)

    qjl_t = tl.load(
        qjl_ptr + offs_d[:, None] + offs_d[None, :] * stride_qjl_row,
        mask=mask_d[:, None] & mask_d[None, :], other=0.0,
    )
    y_projected = tl.dot(y, qjl_t, input_precision="ieee")  # (BLOCK_M, BLOCK_D) = y @ qjl_matrix.T

    term2 = tl.sum(y_projected * signs, axis=1) * residual_norm * correction_scale
    tl.store(out_ptr + offs_m, term2, mask=mask_m)


def quantize(x: torch.Tensor, mse_stage, qjl_matrix: torch.Tensor) -> dict:
    """mse_stage: a TurboQuantMSE instance already constructed with backend='kernel'."""
    orig_shape = x.shape
    d = orig_shape[-1]
    x_flat = x.reshape(-1, d).contiguous()
    n = x_flat.shape[0]
    block_d = triton.next_power_of_2(d)
    block_m = _BLOCK_M
    qjl_matrix = qjl_matrix.to(x.device).contiguous()

    indices, norm = mse_stage.quantize(x)
    x_hat = mse_stage.dequantize(indices, norm)
    x_hat_flat = x_hat.reshape(-1, d).contiguous()

    signs = torch.empty((n, d), dtype=x.dtype, device=x.device)
    residual_norm = torch.empty((n,), dtype=x.dtype, device=x.device)

    grid = (triton.cdiv(n, block_m),)
    _qjl_project_sign_kernel[grid](
        x_flat, x_hat_flat, qjl_matrix, signs, residual_norm,
        x_flat.stride(0), x_hat_flat.stride(0), qjl_matrix.stride(0),
        n, D=d, BLOCK_M=block_m, BLOCK_D=block_d,
        num_warps=16,
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
    block_m = _BLOCK_M
    qjl_matrix = qjl_matrix.to(x_hat_mse.device).contiguous()

    out = torch.empty((n, d), dtype=x_hat_mse.dtype, device=x_hat_mse.device)

    grid = (triton.cdiv(n, block_m),)
    _qjl_correct_kernel[grid](
        x_hat_mse_flat, signs_flat, residual_norm_flat, qjl_matrix, out,
        correction_scale,
        x_hat_mse_flat.stride(0), qjl_matrix.stride(0), out.stride(0),
        n, D=d, BLOCK_M=block_m, BLOCK_D=block_d,
        num_warps=16,
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
    block_m = _BLOCK_M
    qjl_matrix = qjl_matrix.to(y.device).contiguous()
    rotation = mse_stage.rotation.to(y.device).contiguous()
    centroids = mse_stage.codebook.centroids.to(y.device).contiguous()

    term1 = torch.empty((n,), dtype=y.dtype, device=y.device)
    term2 = torch.empty((n,), dtype=y.dtype, device=y.device)

    grid = (triton.cdiv(n, block_m),)
    _prod_inner_product_term1_kernel[grid](
        y_flat, indices_flat, centroids, norm_flat,
        rotation, term1,
        y_flat.stride(0), rotation.stride(0),
        n, D=d, BLOCK_M=block_m, BLOCK_D=block_d,
        num_warps=16,
    )
    _prod_inner_product_term2_kernel[grid](
        y_flat, qjl_matrix, signs_flat, residual_norm_flat, term2,
        correction_scale,
        y_flat.stride(0), qjl_matrix.stride(0),
        n, D=d, BLOCK_M=block_m, BLOCK_D=block_d,
        num_warps=16,
    )

    return (term1 + term2).reshape(*orig_shape)
