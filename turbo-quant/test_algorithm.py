"""
Synthetic verification of TurboQuant algorithm.
No model download required -- validates the math against the paper's theoretical bounds.

Run: python test_algorithm.py
"""

import torch
import math
import time

from lloyd_max import LloydMaxCodebook
from turboquant import TurboQuantMSE, TurboQuantProd


def test_lloyd_max_codebook():
    print("=" * 60)
    print("TEST 1: Lloyd-Max Codebook Properties")
    print("=" * 60)

    for d in [64, 128, 256]:
        for bits in [1, 2, 3, 4]:
            cb = LloydMaxCodebook(d, bits)
            print(f"  d={d:>4d}, bits={bits}: {cb.n_levels} levels, "
                  f"distortion/coord={cb.distortion:.6f}, "
                  f"centroids range=[{cb.centroids.min():.4f}, {cb.centroids.max():.4f}]")

    cb = LloydMaxCodebook(128, 3)
    centroid_sum = cb.centroids.sum().abs().item()
    print(f"\n  Symmetry check (d=128, b=3): sum of centroids = {centroid_sum:.6f} (should be ~0)")
    assert centroid_sum < 0.01, "Centroids should be symmetric!"
    print("  PASSED\n")


def test_mse_quantizer():
    print("=" * 60)
    print("TEST 2: MSE Quantizer Distortion vs Paper Bounds")
    print("=" * 60)

    d = 128
    n_vectors = 1000
    device = "cpu"

    for bits in [1, 2, 3, 4]:
        quantizer = TurboQuantMSE(d, bits, seed=42, device=device)
        x = torch.randn(n_vectors, d, device=device)
        x = x / torch.norm(x, dim=-1, keepdim=True)

        x_hat, indices = quantizer(x)
        mse = ((x - x_hat) ** 2).sum(dim=-1).mean().item()
        theoretical_bound = math.sqrt(3) * math.pi / 2 * (1 / (4 ** bits))
        ratio = mse / theoretical_bound
        status = "OK" if ratio <= 1.5 else "WARN"

        print(f"  bits={bits}: MSE={mse:.6f}, paper_bound={theoretical_bound:.6f}, "
              f"ratio={ratio:.3f} [{status}]")
    print()


def test_inner_product_unbiasedness():
    print("=" * 60)
    print("TEST 3: Inner Product Unbiasedness (QJL Correction)")
    print("=" * 60)

    d = 128
    n_trials = 2000
    device = "cpu"

    for bits in [2, 3, 4]:
        quantizer = TurboQuantProd(d, bits, seed=42, device=device)
        x = torch.randn(n_trials, d, device=device)
        x = x / torch.norm(x, dim=-1, keepdim=True)
        y = torch.randn(n_trials, d, device=device)
        y = y / torch.norm(y, dim=-1, keepdim=True)

        true_ip = (x * y).sum(dim=-1)
        compressed = quantizer.quantize(x)
        estimated_ip = quantizer.inner_product(y, compressed)

        bias = (estimated_ip - true_ip).mean().item()
        rmse = ((estimated_ip - true_ip) ** 2).mean().sqrt().item()
        correlation = torch.corrcoef(torch.stack([true_ip, estimated_ip]))[0, 1].item()

        print(f"  bits={bits}: bias={bias:+.6f}, RMSE={rmse:.6f}, corr={correlation:.4f}")
    print()


def test_needle_in_haystack():
    print("=" * 60)
    print("TEST 4: Needle-in-Haystack Retrieval (Synthetic)")
    print("=" * 60)

    d = 128
    device = "cpu"

    for bits in [2, 3, 4]:
        for seq_len in [512, 2048, 8192]:
            keys = torch.randn(seq_len, d, device=device)
            keys = keys / torch.norm(keys, dim=-1, keepdim=True)

            needle_pos = seq_len // 3
            query = keys[needle_pos].clone().unsqueeze(0)

            quantizer = TurboQuantProd(d, bits, seed=42, device=device)
            compressed = quantizer.quantize(keys)
            estimated_ips = quantizer.inner_product(query.expand(seq_len, -1), compressed)

            top_idx = estimated_ips.argmax().item()
            found = top_idx == needle_pos
            top5 = estimated_ips.topk(5).indices.tolist()
            in_top5 = needle_pos in top5
            status = "EXACT" if found else ("TOP-5" if in_top5 else "MISS")

            print(f"  bits={bits}, seq={seq_len:>5d}: top1={top_idx:>5d} "
                  f"(needle={needle_pos:>5d}) [{status}]")
    print()


def test_compression_ratios():
    print("=" * 60)
    print("TEST 5: Theoretical Compression Ratios")
    print("=" * 60)

    from compressors import MSECompressor

    head_dim = 128
    seq_len = 4096
    B, H = 1, 2

    for bits in [2, 3, 4, 6, 8]:
        comp = MSECompressor(head_dim, bits, seed=42, device="cpu")
        mem = comp.memory_bytes(B, H, seq_len)
        print(f"  {bits}-bit: {mem['compressed_bytes']/1024:.1f} KB compressed vs "
              f"{mem['fp16_bytes']/1024:.1f} KB fp16 = {mem['compression_ratio']:.1f}x")
    print()


def test_v3_compress_decompress():
    print("=" * 60)
    print("TEST 6: V3 Compress/Decompress Fidelity")
    print("=" * 60)

    from compressors import TurboQuantV3

    head_dim = 128
    B, H, S, D = 1, 2, 1024, head_dim
    device = "cpu"

    configs = [
        {"key_bits": 4, "value_bits": 2, "residual_window": 128, "label": "K4/V2 rw=128"},
        {"key_bits": 6, "value_bits": 4, "residual_window": 128, "label": "K6/V4 rw=128"},
        {"key_bits": 8, "value_bits": 4, "residual_window": 128, "label": "K8/V4 rw=128"},
        {"key_bits": 4, "value_bits": 2, "residual_window": 0, "label": "K4/V2 rw=0"},
    ]

    keys = torch.randn(B, H, S, D, device=device)
    values = torch.randn(B, H, S, D, device=device)

    for cfg in configs:
        comp = TurboQuantV3(
            head_dim=head_dim,
            key_bits=cfg["key_bits"],
            value_bits=cfg["value_bits"],
            residual_window=cfg["residual_window"],
            layer_idx=5, n_layers=36, protected_layers=0,
            seed=42, device=device,
        )
        ck, cv = comp.compress_kv(keys, values)
        keys_r, values_r = comp.decompress_kv(ck, cv)

        k_mse = ((keys.float() - keys_r.float()) ** 2).mean().item()
        v_mse = ((values.float() - values_r.float()) ** 2).mean().item()
        mem = comp.memory_bytes(B, H, S)

        print(f"  {cfg['label']:<16s}: K_MSE={k_mse:.6f}, V_MSE={v_mse:.6f}, "
              f"compression={mem['compression_ratio']:.1f}x")
    print()


def test_gpu_benchmark():
    print("=" * 60)
    print("TEST 7: GPU Performance (if CUDA available)")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  CUDA not available, skipping\n")
        return

    from compressors import MSECompressor

    device = "cuda"
    head_dim = 128
    B, H, S = 1, 2, 8192

    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  Shape: B={B}, H={H}, S={S}, D={head_dim}")

    states = torch.randn(B, H, S, head_dim, device=device)

    for bits in [2, 4]:
        comp = MSECompressor(head_dim, bits, seed=42, device=device)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            compressed = comp.compress(states)
        torch.cuda.synchronize()
        compress_ms = (time.perf_counter() - t0) / 10 * 1000

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            decompressed = comp.decompress(compressed)
        torch.cuda.synchronize()
        decompress_ms = (time.perf_counter() - t0) / 10 * 1000

        mem = comp.memory_bytes(B, H, S)
        print(f"  {bits}-bit: compress={compress_ms:.1f}ms, decompress={decompress_ms:.1f}ms, "
              f"ratio={mem['compression_ratio']:.1f}x")
    print()


if __name__ == "__main__":
    print()
    print("TurboQuant Algorithm Verification")
    print("Based on: 'TurboQuant: Online Vector Quantization' (ICLR 2026)")
    print()

    test_lloyd_max_codebook()
    test_mse_quantizer()
    test_inner_product_unbiasedness()
    test_needle_in_haystack()
    test_compression_ratios()
    test_v3_compress_decompress()
    test_gpu_benchmark()

    print("=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)
