"""Propositions 3.1, 3.2, and Bennett's approximation (spec Sec 2-4):
per-unit weight computation for the V-cache token weight w_t, K-cache
channel weight w_c, and quantization hardness sigma_u.
"""

import math

import torch


def token_weight_v(attn: torch.Tensor) -> torch.Tensor:
    """Eq. (2): w_t := sum_tau a_{tau,t}.

    attn: (n_queries, T) attention weights (post-softmax), one row per
    query tau, one column per V-cache token t. Returns a (T,) tensor.
    With a single query (n_queries == 1), this collapses to w_t = a_{tau,t}
    directly (spec Sec 11).
    """
    return attn.sum(dim=0)


def total_variation_after_eviction(attn_row: torch.Tensor, evict_idx: int) -> float:
    """Verifies Proposition 3.1: TV(a_tau, a_hat_tau) == a_{tau,t} exactly,
    for a single query's attention row, when token evict_idx is evicted
    and the rest renormalize per Eq. (1).

    This is a verification helper (used to reproduce the spec Sec 11 hand
    check), not part of the production weight-computation path -- w_t
    (token_weight_v) already IS the TV distance per unit, by the theorem.
    """
    evicted_mass = attn_row[evict_idx].item()
    renorm = attn_row.clone()
    renorm[evict_idx] = 0.0
    renorm = renorm / (1.0 - evicted_mass)
    tv = 0.5 * torch.abs(attn_row - renorm).sum().item()
    return tv


def channel_weight_k(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Eq. (4): w_c := (1/sqrt(d)) * ||Q[:,c]||_2 * ||K[:,c]||_2.

    q, k: (T, d) query and key matrices for one head. Returns a (d,)
    tensor, one weight per channel.
    """
    if q.shape != k.shape:
        raise ValueError(f"q and k must have the same shape, got {q.shape} vs {k.shape}")
    d = q.shape[1]
    q_norms = torch.norm(q, dim=0)  # (d,)
    k_norms = torch.norm(k, dim=0)  # (d,)
    return (q_norms * k_norms) / math.sqrt(d)


def bennett_sigma(dynamic_range: torch.Tensor) -> torch.Tensor:
    """sigma_u := R_u / (2*sqrt(3)) (spec Sec 4, Bennett's high-rate
    approximation). dynamic_range is R_u, any shape."""
    return dynamic_range / (2.0 * math.sqrt(3.0))
