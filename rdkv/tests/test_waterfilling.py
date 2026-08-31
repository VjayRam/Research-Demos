import math

import torch

from rdkv.waterfilling import continuous_waterfill


def test_matches_spec_worked_example_at_fixed_lambda():
    # Spec Sec 11: b_u* = [log2(w_u*sigma_u / 0.1)]_+ at lambda/ln2 = 0.1, sigma_u = 1.
    # continuous_waterfill solves for lambda given a budget, so we instead
    # check the underlying formula directly at the paper's lambda value by
    # picking a target_budget that yields lambda/ln2 == 0.1, then comparing
    # to the spec's table of continuous b_u* values.
    w = torch.tensor([0.500, 0.300, 0.150, 0.050, 1.1619, 0.6893])
    sigma = torch.ones_like(w)
    # At lambda/ln2 = 0.1 (i.e. lambda = 0.1*ln2), sum of clipped b_u* is:
    target_lambda_over_ln2 = 0.1
    lambda_fixed = target_lambda_over_ln2 * math.log(2)
    b_expected = torch.clamp(torch.log2(math.log(2) * w * sigma / lambda_fixed), min=0.0)
    target_budget = b_expected.sum().item()

    b_star, lambda_ = continuous_waterfill(w, sigma, target_budget)

    assert torch.allclose(b_star, b_expected, atol=1e-2)
    assert math.isclose(lambda_, lambda_fixed, rel_tol=1e-2)


def test_token_4_is_evicted_in_worked_example():
    # Spec Sec 11: token 4 (w=0.050) falls below the water level and clips to 0.
    w = torch.tensor([0.500, 0.300, 0.150, 0.050])
    sigma = torch.ones_like(w)
    lambda_fixed = 0.1 * math.log(2)
    b_expected = torch.clamp(torch.log2(math.log(2) * w * sigma / lambda_fixed), min=0.0)
    target_budget = b_expected.sum().item()

    b_star, _ = continuous_waterfill(w, sigma, target_budget)

    assert b_star[3].item() == 0.0
    assert b_star[0].item() > 0.0


def test_budget_binds():
    w = torch.tensor([1.0, 0.5, 0.25, 0.1, 0.05])
    sigma = torch.ones_like(w)
    target_budget = 6.0

    b_star, _ = continuous_waterfill(w, sigma, target_budget)

    assert math.isclose(b_star.sum().item(), target_budget, abs_tol=1e-2)


def test_tightening_budget_increases_eviction_count():
    w = torch.tensor([1.0, 0.5, 0.25, 0.1, 0.05])
    sigma = torch.ones_like(w)

    b_loose, _ = continuous_waterfill(w, sigma, target_budget=15.0)
    b_tight, _ = continuous_waterfill(w, sigma, target_budget=1.0)

    evicted_loose = (b_loose <= 0).sum().item()
    evicted_tight = (b_tight <= 0).sum().item()
    assert evicted_tight >= evicted_loose
