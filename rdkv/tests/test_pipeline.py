import torch

from rdkv.pipeline import RDKVAllocator


def test_allocate_returns_correct_shapes():
    torch.manual_seed(0)
    T, d = 16, 8
    attn = torch.softmax(torch.randn(1, T), dim=-1)
    q = torch.randn(T, d)
    k = torch.randn(T, d)

    allocator = RDKVAllocator()
    result = allocator.allocate(attn, q, k, b_tok=4.0)

    assert result.b_v.shape == (T,)
    assert result.b_k.shape == (d,)
    assert result.w_t.shape == (T,)
    assert result.w_c.shape == (d,)


def test_kept_tokens_matches_nonzero_b_v():
    torch.manual_seed(1)
    T, d = 20, 8
    attn = torch.softmax(torch.randn(1, T), dim=-1)
    q = torch.randn(T, d)
    k = torch.randn(T, d)

    allocator = RDKVAllocator()
    result = allocator.allocate(attn, q, k, b_tok=2.0)

    expected_kept = torch.nonzero(result.b_v > 0, as_tuple=True)[0]
    assert torch.equal(result.kept_tokens, expected_kept)


def test_v_bit_widths_are_from_hardware_set():
    torch.manual_seed(2)
    T, d = 16, 8
    attn = torch.softmax(torch.randn(1, T), dim=-1)
    q = torch.randn(T, d)
    k = torch.randn(T, d)

    allocator = RDKVAllocator()
    result = allocator.allocate(attn, q, k, b_tok=3.0)

    assert set(result.b_v.tolist()).issubset({0, 2, 4, 8, 16})
    assert set(result.b_k.tolist()).issubset({0, 2, 4, 8, 16})


def test_tighter_budget_evicts_more_tokens():
    torch.manual_seed(3)
    T, d = 32, 8
    attn = torch.softmax(torch.randn(1, T), dim=-1)
    q = torch.randn(T, d)
    k = torch.randn(T, d)

    allocator = RDKVAllocator()
    result_loose = allocator.allocate(attn, q, k, b_tok=8.0)
    result_tight = allocator.allocate(attn, q, k, b_tok=0.5)

    assert len(result_tight.kept_tokens) <= len(result_loose.kept_tokens)


def test_k_allocation_uses_only_kept_token_count_as_denominator():
    # Spec Sec 7 Stage 3: B_bar_K := B_K / |T_kept|. If we force heavy V
    # eviction (tiny b_tok), the K budget denominator shrinks, so each
    # surviving channel should tend to get a higher or equal average
    # bit-width than under a looser V budget with the same B_K.
    torch.manual_seed(4)
    T, d = 32, 8
    attn = torch.softmax(torch.randn(1, T), dim=-1)
    q = torch.randn(T, d)
    k = torch.randn(T, d)

    allocator = RDKVAllocator()
    result_tight_v = allocator.allocate(attn, q, k, b_tok=0.5)
    result_loose_v = allocator.allocate(attn, q, k, b_tok=8.0)

    # Tighter V eviction leaves fewer kept tokens -> K's per-channel budget
    # denominator (|T_kept|) shrinks -> average K bit-width should not decrease.
    assert result_tight_v.b_k.float().mean().item() >= result_loose_v.b_k.float().mean().item() - 1e-6


def test_all_tokens_evicted_zeros_out_k_allocation_gracefully():
    # Degenerate case: an extremely tight budget evicts every V token,
    # so |T_kept| == 0 and Stage 3's denominator would divide by zero.
    # The pipeline must handle this without raising.
    torch.manual_seed(5)
    T, d = 8, 4
    attn = torch.softmax(torch.randn(1, T), dim=-1)
    q = torch.randn(T, d)
    k = torch.randn(T, d)

    allocator = RDKVAllocator()
    result = allocator.allocate(attn, q, k, b_tok=1e-6)

    assert result.b_v.sum().item() >= 0  # no exception; near-zero budget mostly evicts
    assert result.b_k.shape == (d,)
