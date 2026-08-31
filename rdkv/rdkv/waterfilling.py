"""Theorem 3.3 (spec Sec 5): continuous reverse water-filling closed-form
bit allocation.

    b_u* = [ log2( ln2 * w_u * sigma_u / lambda ) ]_+

with lambda > 0 chosen so sum(b_u*) matches a target budget. This module
does not discretize to hardware bit-widths {0,2,4,8,16} -- see
rdkv.mckp for the discrete version (Algorithm 2).
"""

import math

import torch


def continuous_waterfill(
    w: torch.Tensor,
    sigma: torch.Tensor,
    target_budget: float,
    lambda_lo: float = 1e-6,
    lambda_hi: float = 1e6,
    iters: int = 60,
) -> tuple[torch.Tensor, float]:
    """Solve Theorem 3.3 for the lambda that makes sum(b_u*) == target_budget.

    w, sigma: 1-D tensors of equal length (per-unit weight and Bennett
    sigma_u). target_budget: desired sum of continuous bit-widths.
    Bisection is geometric in lambda since lambda can span many orders
    of magnitude.

    Returns (b_star, lambda_): b_star is a 1-D float tensor, lambda_ is
    the converged Lagrange multiplier.
    """
    if w.shape != sigma.shape:
        raise ValueError(f"w and sigma must have the same shape, got {w.shape} vs {sigma.shape}")

    def b_at(lambda_: float) -> torch.Tensor:
        value = torch.log2(math.log(2) * w * sigma / lambda_)
        return torch.clamp(value, min=0.0)

    lo, hi = lambda_lo, lambda_hi
    lambda_ = math.sqrt(lo * hi)
    for _ in range(iters):
        lambda_ = math.sqrt(lo * hi)
        current_sum = b_at(lambda_).sum().item()
        if abs(current_sum - target_budget) < 1e-4:
            break
        if current_sum > target_budget:
            lo = lambda_
        else:
            hi = lambda_
    lambda_ = math.sqrt(lo * hi)
    return b_at(lambda_), lambda_
