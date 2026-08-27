import torch

from turboquant.qjl import generate_qjl_matrix, sign_quantize


def test_qjl_matrix_shape_and_determinism():
    s1 = generate_qjl_matrix(d=16, seed=5)
    s2 = generate_qjl_matrix(d=16, seed=5)
    assert s1.shape == (16, 16)
    assert torch.equal(s1, s2)


def test_qjl_matrix_different_seeds_differ():
    s1 = generate_qjl_matrix(d=16, seed=1)
    s2 = generate_qjl_matrix(d=16, seed=2)
    assert not torch.equal(s1, s2)


def test_sign_quantize_only_returns_plus_minus_one():
    x = torch.tensor([-2.0, -0.001, 0.0, 0.001, 3.0])
    signs = sign_quantize(x)
    assert torch.equal(signs.abs(), torch.ones_like(signs))


def test_sign_quantize_zero_maps_to_plus_one():
    x = torch.tensor([0.0])
    signs = sign_quantize(x)
    assert signs.item() == 1.0
