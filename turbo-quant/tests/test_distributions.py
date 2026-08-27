import math

import numpy as np
import torch
from scipy import integrate, stats

from turboquant.distributions import beta_coordinate_density, polar_angle_density
from turboquant.rotation import generate_rotation_matrix


def test_beta_density_integrates_to_one():
    density = beta_coordinate_density(d=8)
    total, _ = integrate.quad(density.pdf, *density.support)
    assert abs(total - 1.0) < 1e-6


def test_beta_density_matches_closed_form_at_d4():
    # f_X(x; d=4) = (2/pi) * sqrt(1 - x^2); coeff at x=0 is exactly 2/pi.
    density = beta_coordinate_density(d=4)
    assert math.isclose(density.pdf(0.0), 2 / math.pi, rel_tol=1e-6)
    assert density.pdf(1.0) == 0.0
    assert density.pdf(-1.0) == 0.0


def test_beta_density_is_symmetric():
    density = beta_coordinate_density(d=32)
    for x in (0.1, 0.3, 0.7):
        assert math.isclose(density.pdf(x), density.pdf(-x), rel_tol=1e-9)


def test_polar_level1_is_uniform_on_full_circle():
    density = polar_angle_density(level=1)
    assert density.support == (0.0, 2 * math.pi)
    assert math.isclose(density.pdf(1.0), density.pdf(4.0))


def test_polar_level_ge2_integrates_to_one_and_peaks_near_quarter_pi():
    for level in (2, 3, 4):
        density = polar_angle_density(level=level)
        total, _ = integrate.quad(density.pdf, *density.support)
        assert abs(total - 1.0) < 1e-6
        # deeper levels concentrate more sharply around pi/4
        assert density.pdf(math.pi / 4) > density.pdf(math.pi / 4 - 0.5)


def test_polar_density_rejects_invalid_level():
    import pytest

    with pytest.raises(ValueError):
        polar_angle_density(level=0)


def test_rotation_coordinate_matches_beta_density_via_ks_test():
    # Pi @ e1 should be distributed as beta_coordinate_density(d) across seeds.
    d = 8
    e1 = torch.zeros(d)
    e1[0] = 1.0
    samples = []
    for seed in range(400):
        q = generate_rotation_matrix(d=d, seed=seed, device="cpu")
        samples.append((q @ e1)[0].item())

    density = beta_coordinate_density(d)

    def cdf(x):
        if np.ndim(x) == 0:  # scalar
            val, _ = integrate.quad(density.pdf, -1.0, x)
            return val
        else:  # array-like
            return np.array([integrate.quad(density.pdf, -1.0, xi)[0] for xi in x])

    ks_stat, p_value = stats.kstest(samples, cdf)
    assert p_value > 0.01
