"""Benchmark turboquant quantize/dequantize latency & throughput: CPU vs GPU.

This benchmarks the turboquant primitives directly (no HuggingFace model
needed) across device, algorithm, bit-width, and head dimension.

Usage:
    python run_perf_benchmark.py --smoke-test
    python run_perf_benchmark.py --algorithms mse prod polar --bits 1 2 3 4 --dims 64 128
"""

import argparse
import time

import torch

from results_logger import default_output_path, write_csv
from turboquant import PolarQuant, TurboQuantMSE, TurboQuantProd

ALGORITHMS = {
    "mse": TurboQuantMSE,
    "prod": TurboQuantProd,
    "polar": PolarQuant,
}


def is_power_of_2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


@torch.no_grad()
def _timed_call(fn, device: str, repeats: int):
    """Run fn() `repeats` times (after a caller-supplied warm-up), returning
    (mean_ms, min_ms) wall-clock latency, with proper CUDA synchronization."""
    latencies_s = []
    for _ in range(repeats):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = fn()
        if device == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        latencies_s.append(end - start)
    mean_s = sum(latencies_s) / len(latencies_s)
    min_s = min(latencies_s)
    return mean_s, min_s, result


@torch.no_grad()
def benchmark_config(cls, d: int, bits: int, device: str, batch_size: int, repeats: int) -> dict:
    quantizer = cls(d, bits, seed=0, device=device)
    x = torch.randn(batch_size, d, device=device)

    # Warm-up pass (discarded) to avoid measuring one-time cache-population
    # costs, e.g. the Lloyd-Max codebook solve which is cached globally.
    warm_compressed = quantizer.quantize(x)
    if isinstance(warm_compressed, tuple):
        quantizer.dequantize(*warm_compressed)
    else:
        quantizer.dequantize(warm_compressed)

    def do_quantize():
        return quantizer.quantize(x)

    q_mean_s, q_min_s, compressed = _timed_call(do_quantize, device, repeats)

    def do_dequantize():
        if isinstance(compressed, tuple):
            return quantizer.dequantize(*compressed)
        return quantizer.dequantize(compressed)

    dq_mean_s, dq_min_s, _ = _timed_call(do_dequantize, device, repeats)

    return {
        "quantize_latency_ms_mean": q_mean_s * 1000.0,
        "quantize_latency_ms_min": q_min_s * 1000.0,
        "quantize_throughput_vecs_per_sec": batch_size / q_mean_s,
        "dequantize_latency_ms_mean": dq_mean_s * 1000.0,
        "dequantize_latency_ms_min": dq_min_s * 1000.0,
        "dequantize_throughput_vecs_per_sec": batch_size / dq_mean_s,
    }


def run(
    devices: list[str],
    algorithms: list[str],
    bits_list: list[int],
    dims: list[int],
    batch_size: int,
    repeats: int,
    output: str | None,
):
    if output is None:
        output = default_output_path("run_perf_benchmark")

    rows = []
    for device in devices:
        if device == "cuda" and not torch.cuda.is_available():
            print(f"note: skipping device=cuda (CUDA not available)")
            continue
        for algorithm in algorithms:
            cls = ALGORITHMS[algorithm]
            for bits in bits_list:
                if algorithm == "prod" and bits < 2:
                    print(f"{device} {algorithm} b={bits}: skipped (prod requires bits >= 2)")
                    continue
                for d in dims:
                    if algorithm == "polar" and not is_power_of_2(d):
                        print(f"{device} {algorithm} b={bits} d={d}: skipped (polar requires power-of-2 d)")
                        continue

                    stats = benchmark_config(cls, d, bits, device, batch_size, repeats)

                    print(
                        f"{device} {algorithm} b={bits} d={d}: "
                        f"quantize={stats['quantize_latency_ms_mean']:.2f}ms "
                        f"(min {stats['quantize_latency_ms_min']:.2f}ms, "
                        f"{stats['quantize_throughput_vecs_per_sec']:.0f} vecs/s), "
                        f"dequantize={stats['dequantize_latency_ms_mean']:.2f}ms "
                        f"(min {stats['dequantize_latency_ms_min']:.2f}ms, "
                        f"{stats['dequantize_throughput_vecs_per_sec']:.0f} vecs/s)"
                    )

                    rows.append(
                        {
                            "device": device,
                            "algorithm": algorithm,
                            "bits": bits,
                            "head_dim": d,
                            "batch_size": batch_size,
                            **stats,
                        }
                    )

    if not rows:
        print("No configs were run; nothing to write.")
        return

    write_csv(rows, output)
    print(f"Results written to: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda"])
    parser.add_argument("--algorithms", nargs="+", default=["mse", "prod", "polar"], choices=list(ALGORITHMS))
    parser.add_argument("--bits", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--dims", nargs="+", type=int, default=[64, 128])
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--output",
        default=None,
        help="path to write CSV results (default: timestamped file under examples/results/)",
    )
    parser.add_argument("--smoke-test", action="store_true", help="tiny config, for quick CI-free verification")
    args = parser.parse_args()

    if args.smoke_test:
        devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        run(devices, ["mse"], [2], [64], batch_size=args.batch_size, repeats=2, output=args.output)
    else:
        run(
            args.devices,
            args.algorithms,
            args.bits,
            args.dims,
            batch_size=args.batch_size,
            repeats=args.repeats,
            output=args.output,
        )


if __name__ == "__main__":
    main()
