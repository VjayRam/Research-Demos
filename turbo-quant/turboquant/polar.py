"""PolarQuant: recursive Cartesian -> polar decomposition with per-level
Lloyd-Max codebooks on the sin-power angle densities."""

import math

import torch

from .codebook import Codebook
from .distributions import polar_angle_density
from .kernel._require import require_kernel_backend
from .rotation import generate_rotation_matrix


class PolarQuant:
    def __init__(
        self, d: int, bits: int, seed: int = 0, device: str | None = None, backend: str = "native"
    ):
        if d < 2 or (d & (d - 1)) != 0:
            raise ValueError(f"d must be a power of 2 and >= 2, got {d}")
        if bits < 1:
            raise ValueError(f"bits must be >= 1, got {bits}")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if backend not in ("native", "kernel"):
            raise ValueError(f"backend must be 'native' or 'kernel', got {backend!r}")
        if backend == "kernel":
            require_kernel_backend(device)

        self.d = d
        self.bits = bits
        self.device = device
        self.backend = backend
        self.n_levels = int(math.log2(d))
        self.rotation = generate_rotation_matrix(d, seed, device=device)
        self.codebooks = [
            Codebook.for_density(polar_angle_density(level), bits)
            for level in range(1, self.n_levels + 1)
        ]

    @staticmethod
    def _decompose(v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pair up the last dimension of v: (..., 2k) -> radii (..., k), angles (..., k)."""
        evens = v[..., 0::2]
        odds = v[..., 1::2]
        radii = torch.sqrt(evens ** 2 + odds ** 2)
        angles = torch.atan2(odds, evens) % (2 * math.pi)
        return radii, angles

    def quantize(self, x: torch.Tensor) -> dict:
        v = x @ self.rotation.T
        angle_indices = []
        if self.backend == "kernel":
            from .kernel import polar as kernel_polar

            for level in range(self.n_levels):
                angle_idx, v = kernel_polar.quantize_level(v, self.codebooks[level].centroids)
                angle_indices.append(angle_idx)
            return {"angle_indices": angle_indices, "final_radius": v.squeeze(-1)}
        for level in range(self.n_levels):
            v, angles = self._decompose(v)
            angle_indices.append(self.codebooks[level].quantize(angles))
        return {"angle_indices": angle_indices, "final_radius": v.squeeze(-1)}

    def dequantize(self, compressed: dict) -> torch.Tensor:
        v = compressed["final_radius"].unsqueeze(-1)
        if self.backend == "kernel":
            from .kernel import polar as kernel_polar

            for level in reversed(range(self.n_levels)):
                v = kernel_polar.dequantize_level(
                    compressed["angle_indices"][level], v, self.codebooks[level].centroids
                )
            return v @ self.rotation
        for level in reversed(range(self.n_levels)):
            angles = self.codebooks[level].dequantize(compressed["angle_indices"][level])
            evens = v * torch.cos(angles)
            odds = v * torch.sin(angles)
            v = torch.stack([evens, odds], dim=-1).flatten(start_dim=-2)
        return v @ self.rotation
