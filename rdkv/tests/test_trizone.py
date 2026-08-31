import torch

from rdkv.pipeline import AllocationResult
from rdkv.trizone import pack_trizone


def _make_allocation(b_v, b_k):
    b_v = torch.tensor(b_v, dtype=torch.long)
    b_k = torch.tensor(b_k, dtype=torch.long)
    kept = torch.nonzero(b_v > 0, as_tuple=True)[0]
    return AllocationResult(
        b_v=b_v, b_k=b_k, kept_tokens=kept,
        w_t=torch.ones_like(b_v, dtype=torch.float32),
        w_c=torch.ones_like(b_k, dtype=torch.float32),
    )


def test_zone_b_holds_exactly_the_16bit_tokens():
    T, d = 6, 4
    k = torch.randn(T, d)
    v = torch.randn(T, d)
    allocation = _make_allocation(b_v=[16, 8, 0, 4, 16, 2], b_k=[8, 4, 2, 16])

    packed = pack_trizone(k, v, allocation)

    expected_zone_b_idx = torch.tensor([0, 4])
    assert torch.equal(packed.zone_b_token_idx, expected_zone_b_idx)
    assert packed.zone_b_v.shape == (2, d)
    assert torch.allclose(packed.zone_b_v, v[expected_zone_b_idx])


def test_zone_a_v_subsegments_partition_the_non16bit_kept_tokens():
    T, d = 6, 4
    k = torch.randn(T, d)
    v = torch.randn(T, d)
    allocation = _make_allocation(b_v=[16, 8, 0, 4, 16, 2], b_k=[8, 4, 2, 16])

    packed = pack_trizone(k, v, allocation)

    # tokens 1 (b=8), 3 (b=4), 5 (b=2) are kept and not 16-bit -> Zone A(V)
    all_zone_a_tokens = set()
    for bit_width, segment in packed.zone_a_v.items():
        assert bit_width in (2, 4, 8)
        all_zone_a_tokens.update(range(segment.shape[0]))  # just shape sanity below
    assert set(packed.zone_a_v.keys()) == {2, 4, 8}
    assert packed.zone_a_v[8].shape[0] == 1
    assert packed.zone_a_v[4].shape[0] == 1
    assert packed.zone_a_v[2].shape[0] == 1


def test_zone_a_k_has_one_row_per_kept_token():
    T, d = 6, 4
    k = torch.randn(T, d)
    v = torch.randn(T, d)
    allocation = _make_allocation(b_v=[16, 8, 0, 4, 16, 2], b_k=[8, 4, 2, 16])

    packed = pack_trizone(k, v, allocation)

    n_kept = allocation.kept_tokens.shape[0]
    assert packed.zone_a_k.shape == (n_kept, d)


def test_channel_permutation_sorts_by_bit_width():
    T, d = 4, 4
    k = torch.randn(T, d)
    v = torch.randn(T, d)
    allocation = _make_allocation(b_v=[16, 8, 4, 2], b_k=[2, 16, 4, 8])

    packed = pack_trizone(k, v, allocation)

    sorted_b_k = allocation.b_k[packed.k_channel_perm]
    assert torch.equal(sorted_b_k, torch.sort(allocation.b_k).values)


def test_all_evicted_v_yields_empty_zones():
    T, d = 4, 4
    k = torch.randn(T, d)
    v = torch.randn(T, d)
    allocation = _make_allocation(b_v=[0, 0, 0, 0], b_k=[8, 4, 2, 16])

    packed = pack_trizone(k, v, allocation)

    assert packed.zone_b_v.shape[0] == 0
    assert packed.zone_a_k.shape[0] == 0
    for segment in packed.zone_a_v.values():
        assert segment.shape[0] == 0


def test_zone_a_k_dequantizes_back_close_to_original_within_bit_budget():
    # Sanity check: k_scale/k_zero_point should let us approximately
    # reconstruct the original (permuted) K rows for the kept tokens.
    torch.manual_seed(0)
    T, d = 10, 8
    k = torch.randn(T, d)
    v = torch.randn(T, d)
    allocation = _make_allocation(b_v=[16, 8, 4, 2, 0, 8, 4, 2, 16, 8], b_k=[8, 4, 16, 2, 8, 4, 16, 2])

    packed = pack_trizone(k, v, allocation)

    dequant = packed.zone_a_k * packed.k_scale + packed.k_zero_point
    original_permuted = k[allocation.kept_tokens][:, packed.k_channel_perm]
    # Loose tolerance -- this only checks the affine params are self-consistent,
    # not tight quantization error (that's covered once decode.py exists in Task 11).
    # Widened from atol=0.5 to 1.0: the 2-bit columns in this synthetic
    # allocation have a dynamic range wide enough that the observed max
    # quantization error (~0.63) exceeds 0.5 -- per the brief's Step 4 note,
    # widen the tolerance rather than change the seed to hide it.
    assert torch.allclose(dequant, original_permuted, atol=1.0)
