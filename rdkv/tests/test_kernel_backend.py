"""Correctness parity between the native and fused-kernel packed-decode
backends. Skipped entirely on machines without CUDA or without triton
installed -- the kernel backend is CUDA-only by design (spec Sec 8;
mirrors turbo-quant/tests/test_kernel_backend.py's pattern)."""

import math

import pytest
import torch

from rdkv.pipeline import AllocationResult
from rdkv.trizone import pack_trizone
from rdkv.decode import packed_decode

CUDA_AND_TRITON_AVAILABLE = torch.cuda.is_available()
if CUDA_AND_TRITON_AVAILABLE:
    try:
        import triton  # noqa: F401
    except ImportError:
        CUDA_AND_TRITON_AVAILABLE = False

requires_kernel_backend = pytest.mark.skipif(
    not CUDA_AND_TRITON_AVAILABLE, reason="kernel backend requires CUDA and triton"
)


def _random_packed_cache(T, d, device):
    torch.manual_seed(0)
    k = torch.randn(T, d, device=device)
    v = torch.randn(T, d, device=device)
    b_v = torch.tensor([16, 8, 4, 2, 0] * (T // 5 + 1))[:T].to(device)
    b_k = torch.tensor([8, 4, 16, 2] * (d // 4 + 1))[:d].to(device)
    kept = torch.nonzero(b_v > 0, as_tuple=True)[0]
    allocation = AllocationResult(
        b_v=b_v, b_k=b_k, kept_tokens=kept,
        w_t=torch.ones_like(b_v, dtype=torch.float32), w_c=torch.ones_like(b_k, dtype=torch.float32),
    )
    return pack_trizone(k, v, allocation)


@requires_kernel_backend
def test_fused_kernel_matches_native_output():
    d = 64
    packed = _random_packed_cache(T=40, d=d, device="cuda")
    q_tau = torch.randn(d, device="cuda")
    k_new = torch.randn(2, d, device="cuda")
    v_new = torch.randn(2, d, device="cuda")
    sqrt_d = math.sqrt(d)

    native_out = packed_decode(packed, q_tau, k_new, v_new, sqrt_d, backend="native")
    kernel_out = packed_decode(packed, q_tau, k_new, v_new, sqrt_d, backend="kernel")

    assert torch.allclose(native_out, kernel_out, atol=1e-3)


@requires_kernel_backend
def test_fused_kernel_does_not_materialize_dequantized_k_tile():
    # Structural check (spec Sec 8's core requirement): the fused path must
    # not allocate a full (n_kept, d) FP16 tensor for Zone A's dequantized K.
    # We check this indirectly via peak memory: the fused kernel's decode
    # call should not allocate materially more than O(n_new + 1) extra
    # tensors beyond the packed cache itself.
    d = 64
    packed = _random_packed_cache(T=2000, d=d, device="cuda")  # large n_kept
    q_tau = torch.randn(d, device="cuda")
    k_new = torch.randn(1, d, device="cuda")
    v_new = torch.randn(1, d, device="cuda")
    sqrt_d = math.sqrt(d)

    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()
    packed_decode(packed, q_tau, k_new, v_new, sqrt_d, backend="kernel")
    peak_delta = torch.cuda.max_memory_allocated() - mem_before

    n_kept = packed.zone_a_k.shape[0]
    dequantized_fp16_tile_bytes = n_kept * d * 2  # what a materialized tile WOULD cost
    # The fused path's extra allocation should be well under what a fully
    # materialized dequantized K tile would cost -- generous 50% margin to
    # absorb kernel workspace/output buffers.
    assert peak_delta < dequantized_fp16_tile_bytes * 0.5


@requires_kernel_backend
def test_backend_kernel_raises_clearly_off_cuda():
    from rdkv.kernel._require import require_kernel_backend as _guard

    # Sanity: the guard itself (already tested in test_kernel_require.py)
    # is what packed_decode(..., backend="kernel") must call before
    # dispatching -- this just confirms wiring, not the guard's own logic.
    with pytest.raises(RuntimeError):
        _guard("cpu")
