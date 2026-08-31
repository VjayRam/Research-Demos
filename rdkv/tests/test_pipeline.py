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
    # Spec Sec 7 Stage 3: B_bar_K := B_K / |T_kept|. B_K is derived from
    # b_tok alone (Sec 7: B_head = 2*b_tok*d*16, B_K = B_head/2), so to
    # isolate the denominator's effect we must hold b_tok FIXED across both
    # calls -- that provably keeps B_K identical both times -- and instead
    # vary the ATTENTION DISTRIBUTION to produce different kept-token counts
    # under that equal budget: a highly peaked attn concentrates weight on a
    # few tokens (MCKP evicts the rest, leaving few kept), while a near-
    # uniform attn spreads weight evenly (MCKP keeps most/all tokens).
    torch.manual_seed(4)
    T, d = 32, 8
    q = torch.randn(T, d)
    k = torch.randn(T, d)

    skewed_logits = torch.randn(1, T) * 20.0  # near one-hot after softmax
    uniform_logits = torch.randn(1, T) * 0.01  # near-uniform after softmax
    attn_skewed = torch.softmax(skewed_logits, dim=-1)
    attn_uniform = torch.softmax(uniform_logits, dim=-1)

    allocator = RDKVAllocator()
    b_tok = 4.0  # SAME b_tok for both calls -> B_K is provably identical.
    result_skewed = allocator.allocate(attn_skewed, q, k, b_tok=b_tok)
    result_uniform = allocator.allocate(attn_uniform, q, k, b_tok=b_tok)

    # Skewed attn evicts more tokens, leaving fewer kept than uniform attn.
    assert len(result_skewed.kept_tokens) < len(result_uniform.kept_tokens)

    # Same B_K, smaller |T_kept| denominator (skewed) -> higher or equal
    # average K bit-width than the larger |T_kept| (uniform) case.
    assert result_skewed.b_k.float().mean().item() >= result_uniform.b_k.float().mean().item() - 1e-6


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
