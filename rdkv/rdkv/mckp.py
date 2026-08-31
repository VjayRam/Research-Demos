"""Algorithm 2 (spec Sec 10, Appendix C): MCKP via Lagrangian bisection.

Solves the discrete multiple-choice knapsack problem from Eq. (6):

    {b_u*} = argmin_{b_u in B} sum_u w_u * eps_u(b_u)   s.t. sum_u b_u <= B

by bisecting on the Lagrange multiplier lambda. Each unit independently
picks the bit-width minimizing w_u*eps(b) + lambda*b; bisection adjusts
lambda until the mean chosen bit-width matches the target average.

DISCLOSED APPROXIMATION (spec Sec 14, decision 2): eps_u(b) here is the
analytic Bennett curve sigma_u * 2**-b (spec Sec 4), NOT the paper's real
empirically-calibrated per-coordinate distortion table (Appendix B, fit
offline on 32 LongBench prefill sequences). Callers who later have a real
calibrated table can pass their own eps_fn to mckp_bisect.
"""

import torch

DEFAULT_BIT_WIDTHS = (0, 2, 4, 8, 16)


def bennett_distortion(sigma: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """eps_u(b) = sigma_u * 2**-b -- the Phase 1 stand-in for the paper's
    empirically-calibrated distortion table (see module docstring)."""
    return sigma * torch.pow(2.0, -b)


def mckp_bisect(
    w: torch.Tensor,
    sigma: torch.Tensor,
    target_avg_bits: float,
    bit_widths: tuple[int, ...] = DEFAULT_BIT_WIDTHS,
    tol: float = 1e-2,
    max_iter: int = 64,
    abs_tol_zero: float = 1e-9,
    eps_fn=bennett_distortion,
) -> tuple[torch.Tensor, dict]:
    """Algorithm 2, verbatim bisection structure.

    w, sigma: 1-D tensors of equal length. target_avg_bits: b-bar, the
    desired mean bit-width across all units. eps_fn(sigma, b) -> distortion,
    defaulting to the Bennett-curve stand-in (see module docstring).

    Implementation note (not in the paper, spec Sec 10): when
    target_avg_bits == 0, the paper's relative-tolerance check
    |mean_b - target| / target is undefined (division by zero). This is
    special-cased to converge as soon as mean_b <= abs_tol_zero, returning
    all-zero bit-widths.

    Returns (b_star, info) where info has keys: lambda_, iters, converged,
    trace (list of dicts with keys it, lambda_lo, lambda_hi, lambda_, mean_b).
    """
    if w.shape != sigma.shape:
        raise ValueError(f"w and sigma must have the same shape, got {w.shape} vs {sigma.shape}")

    bit_options = torch.tensor(bit_widths, dtype=torch.float32, device=w.device)
    n_units = w.shape[0]
    n_options = bit_options.shape[0]

    lambda_lo = 0.0
    lambda_hi = max(w.max().item(), 1.0)

    trace = []
    b_star = torch.zeros(n_units, dtype=torch.float32)
    lambda_ = 0.0
    converged = False
    iters_used = 0

    for it in range(1, max_iter + 1):
        lambda_ = (lambda_lo + lambda_hi) / 2.0
        # cost[u, k] = w_u * eps(sigma_u, bit_options[k]) + lambda * bit_options[k]
        distortion = eps_fn(sigma.unsqueeze(1), bit_options.unsqueeze(0))  # (n_units, n_options)
        cost = w.unsqueeze(1) * distortion + lambda_ * bit_options.unsqueeze(0)
        best_idx = torch.argmin(cost, dim=1)
        b_star = bit_options[best_idx]

        mean_b = b_star.mean().item()
        iters_used = it

        if target_avg_bits <= 0.0:
            converged = mean_b <= abs_tol_zero
        else:
            converged = abs(mean_b - target_avg_bits) / target_avg_bits < tol

        trace.append(
            {"it": it, "lambda_lo": lambda_lo, "lambda_hi": lambda_hi, "lambda_": lambda_, "mean_b": mean_b}
        )

        if converged:
            break
        if mean_b > target_avg_bits:
            lambda_lo = lambda_
        else:
            lambda_hi = lambda_

    return b_star.long(), {
        "lambda_": lambda_,
        "iters": iters_used,
        "converged": converged,
        "trace": trace,
    }
