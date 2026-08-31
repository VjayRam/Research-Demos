"""Scratch profiling script (not part of the package) -- breaks down
fused_packed_decode's sub-call costs at the d=64, n_kept=20000 config that
regressed in run_perf_benchmark.py, using torch.profiler with CUDA events.
"""

import math
import sys

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, ".")

from rdkv.pipeline import AllocationResult
from rdkv.trizone import pack_trizone
from rdkv.decode import packed_decode
from rdkv.kernel.fused_decode import _fused_zone_a_scores, fused_packed_decode


def _random_packed_cache(n_kept_target, d, device):
    torch.manual_seed(0)
    T = max(5, round(n_kept_target * 5 / 4))
    k = torch.randn(T, d, device=device)
    v = torch.randn(T, d, device=device)
    b_v = torch.tensor([16, 8, 4, 2, 0] * (T // 5 + 1))[:T].to(device)
    b_k = torch.tensor([8, 4, 16, 2] * (d // 4 + 1))[:d].to(device)
    kept = torch.nonzero(b_v > 0, as_tuple=True)[0]
    allocation = AllocationResult(
        b_v=b_v, b_k=b_k, kept_tokens=kept,
        w_t=torch.ones_like(b_v, dtype=torch.float32), w_c=torch.ones_like(b_k, dtype=torch.float32),
    )
    return pack_trizone(k, v, allocation), kept.shape[0]


def profile_config(d, n_kept_target, label):
    packed, n_kept = _random_packed_cache(n_kept_target, d, "cuda")
    q_tau = torch.randn(d, device="cuda")
    k_new = torch.randn(1, d, device="cuda")
    v_new = torch.randn(1, d, device="cuda")
    sqrt_d = math.sqrt(d)

    # warm up (JIT compile etc)
    for _ in range(3):
        packed_decode(packed, q_tau, k_new, v_new, sqrt_d, backend="kernel")
    torch.cuda.synchronize()

    print(f"\n=== {label}: d={d} n_kept={n_kept} ===")

    # Time just the fused K-score kernel in isolation
    torch.cuda.synchronize()
    import time
    reps = 100
    start = time.perf_counter()
    for _ in range(reps):
        _fused_zone_a_scores(q_tau, packed)
    torch.cuda.synchronize()
    kscore_ms = (time.perf_counter() - start) / reps * 1000
    print(f"_fused_zone_a_scores alone: {kscore_ms:.4f} ms/call")

    # Time the full fused_packed_decode
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(reps):
        fused_packed_decode(packed, q_tau, k_new, v_new, sqrt_d)
    torch.cuda.synchronize()
    full_ms = (time.perf_counter() - start) / reps * 1000
    print(f"fused_packed_decode (full): {full_ms:.4f} ms/call")
    print(f"glue overhead (full - kscore): {full_ms - kscore_ms:.4f} ms/call")

    # Native for comparison
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(reps):
        packed_decode(packed, q_tau, k_new, v_new, sqrt_d, backend="native")
    torch.cuda.synchronize()
    native_ms = (time.perf_counter() - start) / reps * 1000
    print(f"native packed_decode: {native_ms:.4f} ms/call")

    # Full profiler trace, sorted by CUDA time
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(20):
            fused_packed_decode(packed, q_tau, k_new, v_new, sqrt_d)
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))


if __name__ == "__main__":
    profile_config(64, 20000, "REGRESSED config")
    profile_config(128, 20000, "WORKING config")
    profile_config(64, 1000, "small-n_kept baseline")
