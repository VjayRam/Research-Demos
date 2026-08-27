"""Algorithm 1 (TurboQuant_mse) and Algorithm 2 (TurboQuant_prod), verbatim."""

import math

from .codebook import Codebook
from .distributions import beta_coordinate_density
from .qjl import generate_qjl_matrix, sign_quantize
from .rotation import generate_rotation_matrix

import torch


class TurboQuantMSE:
    """Algorithm 1: rotate, per-coordinate Lloyd-Max quantize, unrotate."""

    def __init__(self, d: int, bits: int, seed: int = 0):
        if bits < 1:
            raise ValueError(f"bits must be >= 1, got {bits}")
        self.d = d
        self.bits = bits
        self.rotation = generate_rotation_matrix(d, seed)
        self.codebook = Codebook.for_density(beta_coordinate_density(d), bits)

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.rotation.T

    def unrotate(self, y: torch.Tensor) -> torch.Tensor:
        return y @ self.rotation

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: any nonzero vector(s), shape (..., d). Returns (indices, norm)."""
        norm = torch.norm(x, dim=-1, keepdim=True)
        unit = x / norm.clamp_min(1e-12)
        y = self.rotate(unit)
        indices = self.codebook.quantize(y)
        return indices, norm.squeeze(-1)

    def dequantize(self, indices: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        y_hat = self.codebook.dequantize(indices)
        x_hat = self.unrotate(y_hat)
        return x_hat * norm.unsqueeze(-1)


class TurboQuantProd:
    """Algorithm 2: (bits-1)-bit MSE stage + 1-bit QJL sign-quantized residual,
    for unbiased inner-product estimation."""

    def __init__(self, d: int, bits: int, seed: int = 0):
        if bits < 2:
            raise ValueError(f"bits must be >= 2 for TurboQuantProd, got {bits}")
        self.d = d
        self.bits = bits
        self.mse = TurboQuantMSE(d, bits - 1, seed=seed)
        self.qjl_matrix = generate_qjl_matrix(d, seed=seed + 1)
        self._correction_scale = math.sqrt(math.pi / 2) / self.d

    def quantize(self, x: torch.Tensor) -> dict:
        indices, norm = self.mse.quantize(x)
        x_hat = self.mse.dequantize(indices, norm)
        residual = x - x_hat
        residual_norm = torch.norm(residual, dim=-1, keepdim=True)

        projected = residual @ self.qjl_matrix.T
        qjl_signs = sign_quantize(projected)

        return {
            "indices": indices,
            "norm": norm,
            "qjl_signs": qjl_signs,
            "residual_norm": residual_norm.squeeze(-1),
        }

    def _qjl_correction(self, compressed: dict) -> torch.Tensor:
        return (
            compressed["residual_norm"].unsqueeze(-1)
            * self._correction_scale
            * (compressed["qjl_signs"] @ self.qjl_matrix)
        )

    def dequantize(self, compressed: dict) -> torch.Tensor:
        x_hat_mse = self.mse.dequantize(compressed["indices"], compressed["norm"])
        return x_hat_mse + self._qjl_correction(compressed)

    def inner_product(self, y: torch.Tensor, compressed: dict) -> torch.Tensor:
        """Unbiased estimate of <x, y> using compressed x (Algorithm 2's payoff)."""
        x_hat_mse = self.mse.dequantize(compressed["indices"], compressed["norm"])
        term1 = (y * x_hat_mse).sum(dim=-1)

        y_projected = y @ self.qjl_matrix.T
        qjl_ip = (y_projected * compressed["qjl_signs"]).sum(dim=-1)
        term2 = compressed["residual_norm"] * self._correction_scale * qjl_ip

        return term1 + term2
