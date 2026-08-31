"""Benchmark rdkv's packed_decode latency: native vs fused-kernel backend,
across cache sizes and head dimensions.

This benchmarks the packed-decode primitive directly (synthetic K/V, no
HuggingFace model needed) -- the actual performance claim of RDKV's Phase 2
(spec Sec 8: the fused kernel should get faster, relative to native, as
n_kept grows, since native pays an explicit (n_kept, d) dequantization
before every decode step while the fused kernel never materializes it).

GPU-only: the kernel backend requires CUDA + triton. On a CPU-only machine
this script benchmarks the native backend only and reports that the kernel
comparison was skipped.

Usage:
    python run_perf_benchmark.py --smoke-test
    python run_perf_benchmark.py --n-kept-values 100 1000 10000 --dims 64 128
"""

import argparse
import math
import time

import torch

from results_logger import default_output_path, write_csv

from rdkv.pipeline import AllocationResult
from rdkv.trizone import pack_trizone
from rdkv.decode import packed_decode

CUDA_AND_TRITON_AVAILABLE = torch.cuda.is_available()
if CUDA_AND_TRITON_AVAILABLE:
    try:
        import triton  # noqa: F401
    except ImportError:
        CUDA_AND_TRITON_AVAILABLE = False


def _random_packed_cache(n_kept_target: int, d: int, device: str):
    """Builds a PackedCache with roughly `n_kept_target` kept tokens, cycling
    through every non-evicted bit-width (16/8/4/2) so Zone A and Zone B are
    both non-trivially populated -- mirrors tests/test_kernel_backend.py's
    helper, generalized to a target kept-count rather than a fixed T."""
    torch.manual_seed(0)
    # b_v pattern [16,8,4,2,0] keeps 4 of every 5 tokens -- scale T up so the
    # kept count lands near n_kept_target.
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


@torch.no_grad()
def _timed_call(fn, device: str, repeats: int):
    """Run fn() `repeats` times (after a caller-supplied warm-up), returning
    (mean_ms, min_ms) wall-clock latency, with proper CUDA synchronization."""
    latencies_s = []
    for _ in range(repeats):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if device == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        latencies_s.append(end - start)
    mean_s = sum(latencies_s) / len(latencies_s)
    min_s = min(latencies_s)
    return mean_s, min_s


@torch.no_grad()
def benchmark_config(n_kept_target: int, d: int, device: str, backend: str, repeats: int) -> dict:
    packed, n_kept = _random_packed_cache(n_kept_target, d, device)
    q_tau = torch.randn(d, device=device)
    k_new = torch.randn(1, d, device=device)
    v_new = torch.randn(1, d, device=device)
    sqrt_d = math.sqrt(d)

    def do_decode():
        return packed_decode(packed, q_tau, k_new, v_new, sqrt_d, backend=backend)

    # Warm-up (discarded) -- for backend="kernel" this also triggers Triton
    # JIT compilation, which must not be counted as steady-state latency.
    do_decode()

    mean_s, min_s = _timed_call(do_decode, device, repeats)

    return {
        "n_kept": n_kept,
        "decode_latency_ms_mean": mean_s * 1000.0,
        "decode_latency_ms_min": min_s * 1000.0,
    }


def run(n_kept_values: list[int], dims: list[int], repeats: int, output: str | None) -> None:
    if output is None:
        output = default_output_path("run_perf_benchmark")

    if not CUDA_AND_TRITON_AVAILABLE:
        print("note: CUDA+triton not available -- benchmarking backend=native only, kernel comparison skipped")

    rows = []
    for d in dims:
        for n_kept_target in n_kept_values:
            native_stats = benchmark_config(n_kept_target, d, "cpu", "native", repeats)
            print(
                f"d={d:4d} n_kept~{native_stats['n_kept']:6d} backend=native (cpu): "
                f"decode={native_stats['decode_latency_ms_mean']:7.3f}ms "
                f"(min {native_stats['decode_latency_ms_min']:7.3f}ms)"
            )
            rows.append(
                {"device": "cpu", "backend": "native", "head_dim": d, **native_stats, "speedup_vs_native_cuda": ""}
            )

            if not CUDA_AND_TRITON_AVAILABLE:
                continue

            native_cuda_stats = benchmark_config(n_kept_target, d, "cuda", "native", repeats)
            print(
                f"d={d:4d} n_kept~{native_cuda_stats['n_kept']:6d} backend=native (cuda): "
                f"decode={native_cuda_stats['decode_latency_ms_mean']:7.3f}ms "
                f"(min {native_cuda_stats['decode_latency_ms_min']:7.3f}ms)"
            )
            rows.append(
                {
                    "device": "cuda",
                    "backend": "native",
                    "head_dim": d,
                    **native_cuda_stats,
                    "speedup_vs_native_cuda": "",
                }
            )

            kernel_stats = benchmark_config(n_kept_target, d, "cuda", "kernel", repeats)
            speedup = native_cuda_stats["decode_latency_ms_mean"] / kernel_stats["decode_latency_ms_mean"]
            print(
                f"d={d:4d} n_kept~{kernel_stats['n_kept']:6d} backend=kernel (cuda): "
                f"decode={kernel_stats['decode_latency_ms_mean']:7.3f}ms "
                f"(min {kernel_stats['decode_latency_ms_min']:7.3f}ms) "
                f"-- {speedup:.2f}x vs native (cuda)"
            )
            rows.append(
                {
                    "device": "cuda",
                    "backend": "kernel",
                    "head_dim": d,
                    **kernel_stats,
                    "speedup_vs_native_cuda": speedup,
                }
            )

    write_csv(rows, output)
    print(f"\nResults written to: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--n-kept-values",
        nargs="+",
        type=int,
        default=[50, 200, 1000, 5000, 20000],
        help="approximate kept-token counts to benchmark (actual T is scaled to hit these)",
    )
    parser.add_argument("--dims", nargs="+", type=int, default=[64, 128])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--output",
        default=None,
        help="path to write CSV results (default: timestamped file under examples/results/)",
    )
    parser.add_argument("--smoke-test", action="store_true", help="tiny config, for quick CI-free verification")
    args = parser.parse_args()

    if args.smoke_test:
        run(n_kept_values=[20], dims=[64], repeats=3, output=args.output)
    else:
        run(args.n_kept_values, args.dims, args.repeats, args.output)


if __name__ == "__main__":
    main()
