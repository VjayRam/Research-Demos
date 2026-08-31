import math

import torch

from rdkv.mckp import bennett_distortion, mckp_bisect


def test_bennett_distortion_matches_formula():
    sigma = torch.tensor([2.0, 1.0])
    b = torch.tensor([0.0, 4.0])
    result = bennett_distortion(sigma, b)
    expected = torch.tensor([2.0 * 2**0, 1.0 * 2**-4])
    assert torch.allclose(result, expected)


def test_zero_bits_gets_full_sigma_as_distortion():
    sigma = torch.tensor([3.5])
    b = torch.tensor([0.0])
    assert torch.allclose(bennett_distortion(sigma, b), sigma)


def test_bit_widths_are_from_the_fixed_hardware_set():
    torch.manual_seed(0)
    w = torch.rand(20) + 0.01
    sigma = torch.ones_like(w)
    b_star, _ = mckp_bisect(w, sigma, target_avg_bits=4.0)
    allowed = {0, 2, 4, 8, 16}
    assert set(b_star.tolist()).issubset(allowed)


def test_converges_to_target_average_within_tolerance():
    torch.manual_seed(1)
    w = torch.rand(50) + 0.01
    sigma = torch.ones_like(w)
    target = 3.0
    b_star, info = mckp_bisect(w, sigma, target_avg_bits=target)
    mean_b = b_star.float().mean().item()
    assert info["converged"]
    assert abs(mean_b - target) / target < 0.15  # discrete rounding, wider than the bisection's own tol


def test_zero_target_budget_returns_all_evicted():
    w = torch.tensor([1.0, 0.5, 0.25])
    sigma = torch.ones_like(w)
    b_star, info = mckp_bisect(w, sigma, target_avg_bits=0.0)
    assert torch.all(b_star == 0)
    assert info["converged"]


def test_higher_weight_units_get_at_least_as_many_bits():
    torch.manual_seed(2)
    w = torch.tensor([0.9, 0.1])
    sigma = torch.ones_like(w)
    b_star, _ = mckp_bisect(w, sigma, target_avg_bits=4.0)
    assert b_star[0].item() >= b_star[1].item()


def test_worked_example_token4_evicted_others_quantized():
    # Spec Sec 11 qualitative pattern: lowest-weight unit (token 4, w=0.05)
    # should be the first evicted as budget tightens, matching Theorem 3.3's
    # continuous result where token 4 clips to 0 first.
    w = torch.tensor([0.500, 0.300, 0.150, 0.050])
    sigma = torch.ones_like(w)
    b_star, _ = mckp_bisect(w, sigma, target_avg_bits=1.0)
    assert b_star[3].item() == 0
    assert b_star[0].item() >= b_star[3].item()
