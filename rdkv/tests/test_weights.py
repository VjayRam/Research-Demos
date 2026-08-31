import math

import torch

from rdkv.weights import bennett_sigma, channel_weight_k, token_weight_v, total_variation_after_eviction


def test_token_weight_is_sum_over_queries():
    # Eq. (2): w_t := sum_tau a_{tau,t}. attn shape (n_queries, T).
    attn = torch.tensor([[0.5, 0.3, 0.15, 0.05], [0.2, 0.2, 0.3, 0.3]])
    w_t = token_weight_v(attn)
    expected = torch.tensor([0.7, 0.5, 0.45, 0.35])
    assert torch.allclose(w_t, expected)


def test_token_weight_single_query_collapses_to_attn_row():
    # Spec Sec 11: with only one query, w_t = a_{tau,t} directly.
    attn = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    w_t = token_weight_v(attn)
    assert torch.allclose(w_t, attn[0])


def test_proposition_3_1_total_variation_equals_evicted_attention_mass():
    # Spec Sec 11 worked example: evicting token 1 (a=0.5) from
    # a_tau = [0.5, 0.3, 0.15, 0.05] gives TV distance exactly 0.5.
    a_tau = torch.tensor([0.5, 0.3, 0.15, 0.05])
    tv = total_variation_after_eviction(a_tau, evict_idx=0)
    assert math.isclose(tv, 0.5, rel_tol=1e-6)
    assert math.isclose(tv, a_tau[0].item(), rel_tol=1e-6)


def test_channel_weight_matches_worked_example():
    # Spec Sec 11: d=2, T=4.
    q = torch.tensor([[1.0, 0.3], [0.0, 0.9], [1.0, 0.2], [0.5, 0.1]])  # (T, d)
    k = torch.tensor([[0.8, 0.5], [0.6, 0.5], [0.4, 0.5], [0.2, 0.5]])  # (T, d)
    w_c = channel_weight_k(q, k)
    expected = torch.tensor([1.1619, 0.6893])
    assert torch.allclose(w_c, expected, atol=1e-3)


def test_bennett_sigma_formula():
    # sigma_u := R_u / (2*sqrt(3))
    dynamic_range = torch.tensor([2.0 * 2 * math.sqrt(3), 6.0])
    sigma = bennett_sigma(dynamic_range)
    expected = torch.tensor([2.0, 6.0 / (2 * math.sqrt(3))])
    assert torch.allclose(sigma, expected)
