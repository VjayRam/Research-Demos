"""
TurboQuant: Two-stage vector quantization with near-optimal distortion.

Stage 1 (MSE): Randomized Hadamard rotation + per-coordinate Lloyd-Max quantization
Stage 2 (QJL): 1-bit Quantized Johnson-Lindenstrauss on residuals for unbiased inner products

The rotation uses a Fast Walsh-Hadamard Transform (O(d log d)) with random sign
flips, matching the paper's production recipe and the SGLang/llama.cpp implementations.

Reference: "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate" (ICLR 2026)
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

from lloyd_max import LloydMaxCodebook


# ---------------------------------------------------------------------------
# Fast Walsh-Hadamard Transform
# ---------------------------------------------------------------------------

@torch.no_grad()
def fwht(x: torch.Tensor) -> torch.Tensor:
    """Normalized Fast Walsh-Hadamard Transform on the last dimension.

    Complexity: O(d log d) vs O(d^2) for a dense matmul.
    Self-inverse: fwht(fwht(x)) == x.
    Last dimension must be a power of 2.
    """
    d = x.shape[-1]
    batch_shape = x.shape[:-1]
    h = 1
    while h < d:
        x = x.view(*batch_shape, d // (2 * h), 2, h)
        a = x[..., 0, :] + x[..., 1, :]
        b = x[..., 0, :] - x[..., 1, :]
        x = torch.stack([a, b], dim=-2).contiguous()
        x = x.view(*batch_shape, d)
        h *= 2
    return x * (d ** -0.5)


def _build_hadamard_matrix(d: int, device: str = "cpu") -> torch.Tensor:
    """Build the normalized d x d Hadamard matrix. d must be a power of 2."""
    H = torch.ones(1, 1, device=device)
    while H.shape[0] < d:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0)
    return H * (d ** -0.5)


def generate_hadamard_signs(d: int, seed: Optional[int] = None, device: str = "cpu") -> torch.Tensor:
    """Generate random +/-1 sign vector for randomized Hadamard rotation."""
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    return (torch.randint(0, 2, (d,), generator=gen).float() * 2 - 1).to(device)


def build_inverse_rotation(d: int, signs: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    """Precompute the d x d inverse rotation matrix for fast GPU decompress.

    Forward rotation:  y = fwht(x * signs)           -- O(d log d), used with bucketize
    Inverse rotation:  x = y @ Pi_inv = signs * fwht(y)  -- O(d^2) matmul via cuBLAS
    """
    H = _build_hadamard_matrix(d, device=device)
    return H * signs.to(device)


# ---------------------------------------------------------------------------
# Legacy: dense random rotation (kept for backward compatibility with tests)
# ---------------------------------------------------------------------------

def generate_rotation_matrix(d: int, seed: Optional[int] = None, device: str = "cpu") -> torch.Tensor:
    """Generate a Haar-distributed random orthogonal rotation matrix via QR decomposition.

    Deprecated in favor of fwht + generate_hadamard_signs (O(d log d) vs O(d^2)).
    """
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    G = torch.randn(d, d, generator=gen)
    Q, R = torch.linalg.qr(G)
    diag_sign = torch.sign(torch.diag(R))
    diag_sign[diag_sign == 0] = 1.0
    Q = Q * diag_sign.unsqueeze(0)
    return Q.to(device)


def generate_qjl_matrix(d: int, m: Optional[int] = None, seed: Optional[int] = None, device: str = "cpu") -> torch.Tensor:
    """Generate the random projection matrix S for QJL. S has i.i.d. N(0,1) entries."""
    if m is None:
        m = d
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    S = torch.randn(m, d, generator=gen)
    return S.to(device)


# ---------------------------------------------------------------------------
# Stage 1: MSE-optimal quantizer
# ---------------------------------------------------------------------------

class TurboQuantMSE(nn.Module):
    """Stage 1: MSE-optimal quantizer.

    Applies randomized Hadamard rotation (FWHT), then per-coordinate Lloyd-Max
    quantization with binary search via torch.bucketize.
    """

    def __init__(self, d: int, bits: int, seed: int = 42, device: str = "cpu"):
        super().__init__()
        self.d = d
        self.bits = bits
        self.device = device

        self.register_buffer("signs", generate_hadamard_signs(d, seed=seed, device=device))
        self.codebook = LloydMaxCodebook(d, bits)
        self.register_buffer("centroids", self.codebook.centroids.to(device))
        self.register_buffer("boundaries", self.codebook.boundaries.to(device))
        self.register_buffer("Pi_inv", build_inverse_rotation(d, self.signs, device=device))

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        return fwht(x * self.signs)

    def unrotate(self, y: torch.Tensor) -> torch.Tensor:
        return y @ self.Pi_inv

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        y = self.rotate(x)
        return torch.bucketize(y, self.boundaries)

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        y_hat = self.centroids[indices]
        return self.unrotate(y_hat)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = self.quantize(x)
        x_hat = self.dequantize(indices)
        return x_hat, indices


# ---------------------------------------------------------------------------
# Stage 1 + Stage 2: Unbiased inner product quantizer
# ---------------------------------------------------------------------------

class TurboQuantProd(nn.Module):
    """
    Stage 1 + Stage 2: Unbiased inner product quantizer.
    Uses (b-1)-bit MSE quantizer + 1-bit QJL on residuals.
    """

    def __init__(self, d: int, bits: int, qjl_dim: Optional[int] = None, seed: int = 42, device: str = "cpu"):
        super().__init__()
        self.d = d
        self.bits = bits
        self.mse_bits = max(bits - 1, 1)
        self.qjl_dim = qjl_dim or d
        self.device = device

        self.mse = TurboQuantMSE(d, self.mse_bits, seed=seed, device=device)
        self.register_buffer("S", generate_qjl_matrix(d, m=self.qjl_dim, seed=seed + 1, device=device))

    def quantize(self, x: torch.Tensor) -> dict:
        x_hat, mse_indices = self.mse(x)
        residual = x - x_hat
        residual_norm = torch.norm(residual, dim=-1, keepdim=True)

        projected = residual @ self.S.T
        qjl_signs = torch.sign(projected)
        qjl_signs[qjl_signs == 0] = 1.0

        return {
            "mse_indices": mse_indices,
            "qjl_signs": qjl_signs,
            "residual_norm": residual_norm.squeeze(-1),
        }

    def dequantize(self, compressed: dict) -> torch.Tensor:
        return self.mse.dequantize(compressed["mse_indices"])

    def inner_product(self, y: torch.Tensor, compressed: dict) -> torch.Tensor:
        """Compute unbiased inner product estimate <x, y> using compressed x."""
        x_mse = self.mse.dequantize(compressed["mse_indices"])
        term1 = (y * x_mse).sum(dim=-1)

        y_projected = y @ self.S.T
        qjl_ip = (y_projected * compressed["qjl_signs"]).sum(dim=-1)
        m = self.qjl_dim
        correction_scale = math.sqrt(math.pi / 2) / m
        term2 = compressed["residual_norm"] * correction_scale * qjl_ip

        return term1 + term2

    def forward(self, x: torch.Tensor) -> dict:
        return self.quantize(x)
