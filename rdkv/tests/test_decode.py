import math

import torch

from rdkv.pipeline import AllocationResult
from rdkv.trizone import pack_trizone
from rdkv.decode import packed_decode


def _full_precision_reference_output(k_all, v_all, q_tau, sqrt_d):
    """Unquantized ground truth: standard attention over every original
    (non-evicted) token plus the new decode token, no packing at all."""
    scores = (q_tau @ k_all.T) / sqrt_d
    weights = torch.softmax(scores, dim=-1)
    return weights @ v_all


def test_packed_decode_matches_full_precision_within_quantization_noise():
    torch.manual_seed(0)
    T, d = 12, 8
    k = torch.randn(T, d)
    v = torch.randn(T, d)
    # Give every token a generous bit-width so quantization noise is small;
    # this test checks the *decomposition* is correct, not aggressive compression.
    b_v = torch.tensor([16, 8, 16, 8, 16, 8, 16, 8, 16, 8, 16, 8])
    b_k = torch.tensor([16, 8, 16, 8, 16, 8, 16, 8])
    kept = torch.nonzero(b_v > 0, as_tuple=True)[0]
    allocation = AllocationResult(
        b_v=b_v, b_k=b_k, kept_tokens=kept,
        w_t=torch.ones_like(b_v, dtype=torch.float32), w_c=torch.ones_like(b_k, dtype=torch.float32),
    )
    packed = pack_trizone(k, v, allocation)

    q_tau = torch.randn(d)
    k_new = torch.randn(1, d)
    v_new = torch.randn(1, d)
    sqrt_d = math.sqrt(d)

    output = packed_decode(packed, q_tau, k_new, v_new, sqrt_d)

    k_all = torch.cat([k[kept], k_new], dim=0)
    v_all = torch.cat([v[kept], v_new], dim=0)
    reference = _full_precision_reference_output(k_all, v_all, q_tau, sqrt_d)

    assert output.shape == (d,)
    # Loose tolerance: Zone A's K/V rows went through real quantization noise.
    assert torch.allclose(output, reference, atol=0.5)


def test_output_shape_is_head_dim():
    torch.manual_seed(1)
    T, d = 6, 4
    k = torch.randn(T, d)
    v = torch.randn(T, d)
    b_v = torch.tensor([16, 0, 8, 4, 16, 2])
    b_k = torch.tensor([8, 4, 16, 2])
    kept = torch.nonzero(b_v > 0, as_tuple=True)[0]
    allocation = AllocationResult(
        b_v=b_v, b_k=b_k, kept_tokens=kept,
        w_t=torch.ones_like(b_v, dtype=torch.float32), w_c=torch.ones_like(b_k, dtype=torch.float32),
    )
    packed = pack_trizone(k, v, allocation)

    q_tau = torch.randn(d)
    k_new = torch.randn(2, d)
    v_new = torch.randn(2, d)
    output = packed_decode(packed, q_tau, k_new, v_new, math.sqrt(d))
    assert output.shape == (d,)


def test_all_evicted_falls_back_to_new_tokens_only():
    torch.manual_seed(2)
    T, d = 4, 4
    k = torch.randn(T, d)
    v = torch.randn(T, d)
    b_v = torch.zeros(T, dtype=torch.long)
    b_k = torch.tensor([8, 4, 16, 2])
    kept = torch.nonzero(b_v > 0, as_tuple=True)[0]
    allocation = AllocationResult(
        b_v=b_v, b_k=b_k, kept_tokens=kept,
        w_t=torch.ones_like(b_v, dtype=torch.float32), w_c=torch.ones_like(b_k, dtype=torch.float32),
    )
    packed = pack_trizone(k, v, allocation)

    q_tau = torch.randn(d)
    k_new = torch.randn(3, d)
    v_new = torch.randn(3, d)
    output = packed_decode(packed, q_tau, k_new, v_new, math.sqrt(d))

    reference = _full_precision_reference_output(k_new, v_new, q_tau, math.sqrt(d))
    assert torch.allclose(output, reference, atol=1e-4)
