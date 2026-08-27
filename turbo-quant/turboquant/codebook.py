"""A Lloyd-Max codebook: solved centroids/boundaries plus quantize/dequantize."""

from dataclasses import dataclass

import torch

from .distributions import Density
from .lloyd_max import expected_distortion, solve_lloyd_max

_codebook_cache: dict[tuple[str, int], "Codebook"] = {}


@dataclass
class Codebook:
    centroids: torch.Tensor
    boundaries: torch.Tensor
    distortion: float

    @classmethod
    def for_density(cls, density: Density, bits: int) -> "Codebook":
        key = (density.name, bits)
        if key in _codebook_cache:
            return _codebook_cache[key]

        centroids, boundaries = solve_lloyd_max(density.pdf, density.support, bits)
        distortion = expected_distortion(density.pdf, density.support, centroids, boundaries)
        codebook = cls(centroids=centroids, boundaries=boundaries, distortion=distortion)
        _codebook_cache[key] = codebook
        return codebook

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Literal per-element argmin over centroids (paper's Algorithm 1 line)."""
        diffs = x.unsqueeze(-1) - self.centroids.to(x.device)
        return diffs.abs().argmin(dim=-1)

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        return self.centroids.to(indices.device)[indices]
