import math

import pytest
import torch

from turboquant.cartesian import TurboQuantMSE


def test_rejects_invalid_bits():
    with pytest.raises(ValueError):
        TurboQuantMSE(d=4, bits=0)


def test_round_trip_reduces_error_with_more_bits():
    torch.manual_seed(0)
    x = torch.randn(4, 32)
    errors = []
    for bits in (1, 2, 3, 4):
        q = TurboQuantMSE(d=32, bits=bits, seed=1)
        indices, norm = q.quantize(x)
        x_hat = q.dequantize(indices, norm)
        errors.append(((x - x_hat) ** 2).sum().item())
    assert errors == sorted(errors, reverse=True)


def test_rotation_is_orthogonal_round_trip():
    q = TurboQuantMSE(d=16, bits=3, seed=2)
    x = torch.randn(3, 16)
    assert torch.allclose(q.unrotate(q.rotate(x)), x, atol=1e-4)


def test_worked_example_d4_b1_matches_primer():
    # Primer's hand-worked example (#worked-simple): input x=(1,0,0,0), a
    # concrete (normalized) Hadamard matrix as the orthogonal transform,
    # b=1. We inject that exact Hadamard rotation in place of the package's
    # random QR rotation so the deterministic worked numbers are
    # reproducible: after rotation every coordinate is 0.5, centroids are
    # +/- sqrt(2/pi)/sqrt(4) = +/-0.39894, reconstruction is
    # (0.79788, 0, 0, 0), and the squared error is ~0.20212.
    q = TurboQuantMSE(d=4, bits=1, seed=0)
    hadamard = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0, 1.0],
        ]
    ) * 0.5
    q.rotation = hadamard

    x = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    rotated = q.rotate(x / x.norm())
    assert torch.allclose(rotated, torch.full((1, 4), 0.5), atol=1e-6)

    expected_centroid = math.sqrt(2 / math.pi) / math.sqrt(4)
    assert math.isclose(q.codebook.centroids[1].item(), expected_centroid, rel_tol=1e-3)

    indices, norm = q.quantize(x)
    x_hat = q.dequantize(indices, norm)
    assert torch.allclose(x_hat, torch.tensor([[0.79788, 0.0, 0.0, 0.0]]), atol=1e-3)

    squared_error = ((x - x_hat) ** 2).sum().item()
    assert math.isclose(squared_error, 0.20212, abs_tol=2e-3)
