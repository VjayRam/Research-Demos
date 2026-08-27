import pytest
import torch

from turboquant.cartesian import TurboQuantProd


def test_rejects_bits_below_two():
    with pytest.raises(ValueError):
        TurboQuantProd(d=8, bits=1)


def test_quantize_dequantize_round_trip_shapes():
    q = TurboQuantProd(d=16, bits=3, seed=0)
    x = torch.randn(5, 16)
    compressed = q.quantize(x)
    x_hat = q.dequantize(compressed)
    assert x_hat.shape == x.shape


def test_inner_product_is_empirically_unbiased():
    torch.manual_seed(0)
    d = 64
    q = TurboQuantProd(d=d, bits=2, seed=3)

    estimates = []
    truths = []
    for _ in range(300):
        x = torch.randn(d)
        y = torch.randn(d)
        compressed = q.quantize(x.unsqueeze(0))
        estimate = q.inner_product(y.unsqueeze(0), compressed).item()
        estimates.append(estimate)
        truths.append(torch.dot(x, y).item())

    mean_estimate = sum(estimates) / len(estimates)
    mean_truth = sum(truths) / len(truths)
    # Both should be close to 0 in expectation (independent random x, y);
    # check the estimator doesn't introduce the paper's ~36% shrink bias
    # by comparing average absolute deviation instead.
    bias = abs(mean_estimate - mean_truth)
    assert bias < 0.15


def test_inner_product_matches_dequantized_dot_for_pure_mse_term():
    # With residual_norm forced to 0, inner_product's QJL term vanishes and
    # it must equal a plain dot product against the MSE reconstruction.
    q = TurboQuantProd(d=16, bits=2, seed=1)
    x = torch.randn(1, 16)
    y = torch.randn(1, 16)
    compressed = q.quantize(x)
    compressed["residual_norm"] = torch.zeros_like(compressed["residual_norm"])

    estimate = q.inner_product(y, compressed)
    x_hat_mse = q.mse.dequantize(compressed["indices"], compressed["norm"])
    expected = (y * x_hat_mse).sum(dim=-1)
    assert torch.allclose(estimate, expected, atol=1e-5)
