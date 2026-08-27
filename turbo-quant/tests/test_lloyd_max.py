import math

from turboquant.distributions import beta_coordinate_density
from turboquant.lloyd_max import expected_distortion, solve_lloyd_max


def test_centroid_and_boundary_counts():
    density = beta_coordinate_density(d=64)
    centroids, boundaries = solve_lloyd_max(density.pdf, density.support, bits=2)
    assert centroids.shape == (4,)
    assert boundaries.shape == (3,)


def test_centroids_are_sorted():
    density = beta_coordinate_density(d=64)
    centroids, _ = solve_lloyd_max(density.pdf, density.support, bits=3)
    assert list(centroids) == sorted(centroids.tolist())


def test_b1_centroid_matches_exact_half_normal_formula():
    # For b=1, the optimal centroid is exactly E[X | X > 0] = sqrt(2/pi)/sqrt(d).
    d = 128
    density = beta_coordinate_density(d)
    centroids, _ = solve_lloyd_max(density.pdf, density.support, bits=1)
    expected = math.sqrt(2 / math.pi) / math.sqrt(d)
    assert math.isclose(centroids[1].item(), expected, rel_tol=1e-3)
    assert math.isclose(centroids[0].item(), -expected, rel_tol=1e-3)


def test_reproduces_paper_theorem1_table_at_d128():
    # Table: b -> d * C(f_X, b) approx 0.360, 0.117, 0.030, 0.009
    expected_by_bits = {1: 0.360, 2: 0.117, 3: 0.030, 4: 0.009}
    d = 128
    density = beta_coordinate_density(d)
    for bits, expected in expected_by_bits.items():
        centroids, boundaries = solve_lloyd_max(density.pdf, density.support, bits=bits)
        distortion = expected_distortion(density.pdf, density.support, centroids, boundaries)
        d_mse = d * distortion
        assert abs(d_mse - expected) < 0.02, f"bits={bits}: got {d_mse}, want {expected}"


def test_distortion_decreases_with_more_bits():
    density = beta_coordinate_density(d=64)
    distortions = []
    for bits in (1, 2, 3, 4):
        centroids, boundaries = solve_lloyd_max(density.pdf, density.support, bits=bits)
        distortions.append(expected_distortion(density.pdf, density.support, centroids, boundaries))
    assert distortions == sorted(distortions, reverse=True)
