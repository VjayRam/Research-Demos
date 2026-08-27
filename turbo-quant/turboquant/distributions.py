"""Exact coordinate densities used by TurboQuant (Beta) and PolarQuant (sin-power angles)."""

import math
from dataclasses import dataclass
from typing import Callable

from scipy import integrate


@dataclass(frozen=True)
class Density:
    """A probability density with a known, finite support interval."""

    pdf: Callable[[float], float]
    support: tuple[float, float]
    name: str


def beta_coordinate_density(d: int) -> Density:
    """Exact coordinate density of a uniform point on S^(d-1) (paper Eq. 4).

    f_X(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2),
    x in [-1, 1].
    """
    log_coeff = math.lgamma(d / 2) - 0.5 * math.log(math.pi) - math.lgamma((d - 1) / 2)
    coeff = math.exp(log_coeff)
    power = (d - 3) / 2

    def pdf(x: float) -> float:
        if abs(x) >= 1.0:
            return 0.0
        return coeff * (1 - x * x) ** power

    return Density(pdf=pdf, support=(-1.0, 1.0), name=f"beta_d{d}")


def polar_angle_density(level: int) -> Density:
    """PolarQuant angle density at a given recursion level.

    Level 1 pairs raw (signed) coordinates: uniform on [0, 2*pi).
    Level >= 2 pairs nonnegative radii from the previous level: density
    proportional to sin(2*theta)^(2^(level-1) - 1) on [0, pi/2].
    """
    if level < 1:
        raise ValueError(f"level must be >= 1, got {level}")

    if level == 1:
        lo, hi = 0.0, 2 * math.pi
        c = 1.0 / (hi - lo)
        return Density(pdf=lambda theta: c, support=(lo, hi), name="polar_uniform")

    power = 2 ** (level - 1) - 1
    lo, hi = 0.0, math.pi / 2

    def unnormalized(theta: float) -> float:
        return math.sin(2 * theta) ** power

    z, _ = integrate.quad(unnormalized, lo, hi)

    def pdf(theta: float) -> float:
        if theta < lo or theta > hi:
            return 0.0
        return unnormalized(theta) / z

    return Density(pdf=pdf, support=(lo, hi), name=f"polar_level{level}")
