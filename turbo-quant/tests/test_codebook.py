import torch

from turboquant.codebook import Codebook
from turboquant.distributions import beta_coordinate_density


def test_quantize_picks_nearest_centroid_exactly():
    codebook = Codebook.for_density(beta_coordinate_density(d=32), bits=2)
    # Feeding a centroid's exact value back in must return its own index.
    for i, c in enumerate(codebook.centroids):
        idx = codebook.quantize(c.unsqueeze(0))
        assert idx.item() == i


def test_dequantize_looks_up_centroid_values():
    codebook = Codebook.for_density(beta_coordinate_density(d=32), bits=2)
    indices = torch.tensor([0, 1, 2, 3])
    values = codebook.dequantize(indices)
    assert torch.equal(values, codebook.centroids)


def test_quantize_dequantize_shapes_for_batched_vectors():
    codebook = Codebook.for_density(beta_coordinate_density(d=16), bits=3)
    x = torch.randn(5, 16) * 0.1
    indices = codebook.quantize(x)
    assert indices.shape == (5, 16)
    reconstructed = codebook.dequantize(indices)
    assert reconstructed.shape == (5, 16)


def test_for_density_caches_by_name_and_bits():
    c1 = Codebook.for_density(beta_coordinate_density(d=64), bits=2)
    c2 = Codebook.for_density(beta_coordinate_density(d=64), bits=2)
    assert c1 is c2


def test_for_density_distinguishes_different_bits():
    c1 = Codebook.for_density(beta_coordinate_density(d=64), bits=1)
    c2 = Codebook.for_density(beta_coordinate_density(d=64), bits=2)
    assert c1 is not c2
    assert c1.centroids.numel() == 2
    assert c2.centroids.numel() == 4
