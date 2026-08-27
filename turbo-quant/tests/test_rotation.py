import torch

from turboquant.rotation import generate_rotation_matrix


def test_output_is_orthogonal():
    q = generate_rotation_matrix(d=16, seed=0, device="cpu")
    identity = torch.eye(16)
    assert torch.allclose(q.T @ q, identity, atol=1e-5)


def test_output_shape_and_dtype():
    q = generate_rotation_matrix(d=8, seed=0)
    assert q.shape == (8, 8)
    assert q.dtype == torch.float32


def test_deterministic_given_same_seed():
    q1 = generate_rotation_matrix(d=16, seed=42)
    q2 = generate_rotation_matrix(d=16, seed=42)
    assert torch.equal(q1, q2)


def test_different_seeds_differ():
    q1 = generate_rotation_matrix(d=16, seed=1)
    q2 = generate_rotation_matrix(d=16, seed=2)
    assert not torch.equal(q1, q2)


def test_cache_returns_same_tensor_object():
    q1 = generate_rotation_matrix(d=16, seed=7)
    q2 = generate_rotation_matrix(d=16, seed=7)
    assert q1 is q2
