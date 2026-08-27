"""Algorithm 1 (TurboQuant_mse) and Algorithm 2 (TurboQuant_prod), verbatim."""

from .codebook import Codebook
from .distributions import beta_coordinate_density
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
