import pytest
import torch

from turboquant.polar import PolarQuant


def test_rejects_non_power_of_two_dimension():
    with pytest.raises(ValueError):
        PolarQuant(d=6, bits=2)


def test_rejects_invalid_bits():
    with pytest.raises(ValueError):
        PolarQuant(d=8, bits=0)


def test_number_of_levels_and_angle_indices():
    q = PolarQuant(d=8, bits=2, seed=0, device="cpu")
    assert q.n_levels == 3  # log2(8)
    x = torch.randn(2, 8)
    compressed = q.quantize(x)
    assert len(compressed["angle_indices"]) == 3
    assert compressed["angle_indices"][0].shape == (2, 4)  # d/2 angles at level 1
    assert compressed["angle_indices"][1].shape == (2, 2)  # d/4 at level 2
    assert compressed["angle_indices"][2].shape == (2, 1)  # d/8 at level 3
    assert compressed["final_radius"].shape == (2,)


def test_round_trip_shape():
    q = PolarQuant(d=16, bits=3, seed=1, device="cpu")
    x = torch.randn(4, 16)
    compressed = q.quantize(x)
    x_hat = q.dequantize(compressed)
    assert x_hat.shape == x.shape


def test_round_trip_error_decreases_with_more_bits():
    torch.manual_seed(0)
    x = torch.randn(4, 16)
    errors = []
    for bits in (1, 2, 3, 4):
        q = PolarQuant(d=16, bits=bits, seed=2, device="cpu")
        compressed = q.quantize(x)
        x_hat = q.dequantize(compressed)
        errors.append(((x - x_hat) ** 2).sum().item())
    assert errors == sorted(errors, reverse=True)


def test_norm_is_approximately_preserved():
    # Rotation preserves norm exactly; quantization error should keep the
    # reconstructed norm reasonably close to the original at moderate bits.
    q = PolarQuant(d=16, bits=4, seed=3, device="cpu")
    x = torch.randn(1, 16)
    compressed = q.quantize(x)
    x_hat = q.dequantize(compressed)
    assert torch.allclose(x.norm(), x_hat.norm(), rtol=0.1)
