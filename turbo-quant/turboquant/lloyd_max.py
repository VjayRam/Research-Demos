"""Generic continuous Lloyd-Max scalar quantizer solver.

Parameterized entirely by a (pdf, support) pair -- no knowledge of any
specific density baked in. Used by both cartesian.py (Beta density) and
polar.py (sin-power angle densities).
"""

from typing import Callable

import torch
from scipy import integrate


def solve_lloyd_max(
    pdf: Callable[[float], float],
    support: tuple[float, float],
    bits: int,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Continuous 1-D Lloyd-Max quantizer for an arbitrary density on a finite support.

    Returns (centroids, boundaries): centroids has 2**bits entries (sorted
    ascending), boundaries has 2**bits - 1 entries (midpoints between
    neighboring centroids).
    """
    n_levels = 2 ** bits
    lo, hi = support

    centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]

    for _ in range(max_iter):
        boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]
        edges = [lo] + boundaries + [hi]

        new_centroids = []
        for i in range(n_levels):
            a, b = edges[i], edges[i + 1]
            numerator, _ = integrate.quad(lambda x: x * pdf(x), a, b)
            denominator, _ = integrate.quad(pdf, a, b)
            if denominator > 1e-15:
                new_centroids.append(numerator / denominator)
            else:
                new_centroids.append(centroids[i])

        max_shift = max(abs(new_centroids[i] - centroids[i]) for i in range(n_levels))
        centroids = new_centroids
        if max_shift < tol:
            break

    boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]
    return (
        torch.tensor(centroids, dtype=torch.float32),
        torch.tensor(boundaries, dtype=torch.float32),
    )


def expected_distortion(
    pdf: Callable[[float], float],
    support: tuple[float, float],
    centroids: torch.Tensor,
    boundaries: torch.Tensor,
) -> float:
    """Expected squared quantization error under pdf: C(f, b) from the paper."""
    lo, hi = support
    edges = [lo] + boundaries.tolist() + [hi]
    total = 0.0
    for i in range(len(centroids)):
        a, b = edges[i], edges[i + 1]
        c = centroids[i].item()
        dist, _ = integrate.quad(lambda x: (x - c) ** 2 * pdf(x), a, b)
        total += dist
    return total
