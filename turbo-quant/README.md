# TurboQuant

A paper-accurate PyTorch implementation of Google's **TurboQuant** (Algorithms 1
and 2) and **PolarQuant**, based on *"TurboQuant: Online Vector Quantization
with Near-optimal Distortion Rate"* (Zandieh et al., ICLR 2026,
[arXiv:2504.19874](https://arxiv.org/abs/2504.19874)).

This package implements the paper's math with no approximation shortcuts: a
true Haar-random rotation (QR of a Gaussian matrix, not a Hadamard transform)
and Lloyd-Max solved against the exact Beta / sin-power densities (not a
Gaussian approximation).

An earlier exploratory V3 variant (Hadamard rotation, Gaussian Lloyd-Max,
asymmetric K/V bits, residual windowing) lived as flat files in this directory
and has been removed. The `turboquant/` package below is the replacement.

Interactive walkthrough of the math: [`turboquant-primer.html`](turboquant-primer.html).

---

## Table of Contents

1. [What is implemented](#what-is-implemented)
2. [Algorithm overview](#algorithm-overview)
3. [Mathematical foundation](#mathematical-foundation)
4. [Results](#results)
5. [File structure](#file-structure)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Experiments](#experiments)
9. [Limitations](#limitations)

---

## What is implemented

| Class | Paper | Role |
|-------|-------|------|
| `TurboQuantMSE` | Algorithm 1 | Rotate, per-coordinate Lloyd-Max, unrotate. MSE-optimal reconstruction. |
| `TurboQuantProd` | Algorithm 2 | `(bits-1)`-bit MSE stage plus 1-bit QJL on the residual. Unbiased inner-product estimator. |
| `PolarQuant` | Related method (blog / PolarQuant) | Recursive Cartesian→polar decomposition with per-level Lloyd-Max on sin-power angle densities. Requires `d` to be a power of 2. |

Shared machinery:

- Haar-distributed rotation via QR (`rotation.py`), cached per `(d, seed, device)`
- Exact Beta coordinate density and PolarQuant angle densities (`distributions.py`)
- Generic continuous Lloyd-Max solver (`lloyd_max.py`) and cached codebooks (`codebook.py`)
- Square QJL matrix and `sign` quantization (`qjl.py`)

CPU and CUDA are both supported. Passing `device=None` auto-detects CUDA.

---

## Algorithm overview

### TurboQuantMSE (Algorithm 1)

```
x  →  store ||x||  →  unit vector  →  y = Π x
                                      ↓
                           per-coordinate Lloyd-Max (b bits)
                                      ↓
                           look up centroids  →  Πᵀ ŷ  →  rescale by ||x||
```

`Π` is a Haar-random orthogonal matrix, generated once per `(d, seed)` by QR
on i.i.d. Gaussians with the standard sign fix on `diag(R)`. Quantization is
literal nearest-centroid (`argmin` over centroids), matching Algorithm 1.

### TurboQuantProd (Algorithm 2)

The MSE stage uses `b-1` bits. The residual `r = x - x̂_mse` is sign-quantized
after a Gaussian QJL projection `S`. Dequantization adds the paper's correction
`(√(π/2)/d) · ‖r‖ · Sᵀ sign(Sr)`. `inner_product(y, compressed)` estimates
`⟨x, y⟩` without forming the full reconstruction of the residual.

### PolarQuant

Rotate, then recursively pair coordinates into `(radius, angle)` for
`log₂(d)` levels. Level 1 angles are uniform on `[0, 2π)`; later levels use
the sin-power density on `[0, π/2]`. The final radius is stored in floating
point; every angle is Lloyd-Max quantized at `b` bits.

---

## Mathematical foundation

### Why rotation works

A unit vector `x ∈ S^{d-1}` after a Haar-random rotation has coordinates
distributed as (paper Eq. 4):

```
f(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1 − x²)^((d−3)/2),    x ∈ [−1, 1]
```

Every coordinate shares this marginal, independent of the structure of `x`,
so one scalar codebook is valid for all dimensions and all inputs.

### Lloyd-Max

Centroids and Voronoi boundaries minimize MSE for a known density:

```
boundary[i] = (centroid[i] + centroid[i+1]) / 2
centroid[i] = E[X | boundary[i−1] < X ≤ boundary[i]]
```

Solved offline with `scipy.integrate.quad` on the exact density, then cached
by `(density name, bits)`.

### Distortion (Theorem 1)

The paper reports `d · C(f_X, b)` ≈ **0.360, 0.117, 0.030, 0.009** at
`d = 128` for `b = 1..4`. The solver here reproduces that table (see
`tests/test_lloyd_max.py`).

On **real** Qwen2.5-0.5B key vectors the empirical mean-squared relative
error tracks the *solved* codebook distortion `d · C(f, b)`, not the looser
closed-form `1.5 · 4^{-b}`:

| Bits | Empirical (4988 keys, d=64) | Solved codebook `d·C` | General `1.5/4^b` |
|------|-----------------------------|-----------------------|-------------------|
| 1    | 0.3645                      | 0.3584                | 0.3750            |
| 2    | 0.1159                      | 0.1145                | 0.0938            |
| 3    | 0.0336                      | 0.0334                | 0.0234            |
| 4    | 0.00913                     | 0.00913               | 0.00586           |

The 1-bit case lands inside the general bound. At 2–4 bits the general bound
is optimistic; the vectors still match the codebook they were actually
quantized with.

Source: `examples/results/run_experiments_distortion_20260827_163418.csv`.

---

## Results

Hardware: CUDA (RTX 4070-class). Model runs use **Qwen2.5-0.5B** (`head_dim=64`),
fp32 weights. Compression ratios below are **analytical** (index bits + one
fp16 norm versus fp16 storage), not measured packed-byte size.

### Perplexity vs compression (WikiText-2)

Full KV cache round-tripped through the quantizer on every step via
`QuantizingCache` (reconstructed tensors are stored, not packed indices).
Text is a ~2000-word WikiText-2 test sample.

| Algorithm | Bits | Perplexity | Δ vs baseline | Compression |
|-----------|------|------------|---------------|-------------|
| baseline  | —    | 10.13      | 0             | 1.00×       |
| mse       | 1    | 4135       | +4125         | 12.8×       |
| mse       | 2    | 1497       | +1486         | 7.11×       |
| mse       | 3    | 200.3      | +190.2        | 4.92×       |
| mse       | 4    | 156.1      | +145.9        | 3.76×       |
| prod      | 2    | 20196      | +20186        | 7.11×       |
| prod      | 3    | 3231       | +3221         | 4.92×       |
| prod      | 4    | 799.6      | +789.4        | 3.76×       |
| polar     | 1    | 17570      | +17560        | 12.8×       |
| polar     | 2    | 2728       | +2718         | 7.11×       |
| polar     | 3    | 728.6      | +718.5        | 4.92×       |
| **polar** | **4**| **70.44**  | **+60.3**     | **3.76×**   |

Takeaways:

- Quantizing **every** token at 1–4 bits, with no residual fp16 window, is
  harsh on this 0.5B model. Perplexity stays far above the 10.13 baseline.
- **PolarQuant at 4 bits is the best compressed setting** (70 vs 156 MSE vs
  800 Prod).
- **Prod is worse than MSE at every bit-width** on this attention-through-
  softmax workload. QJL is unbiased for inner products, not for softmax.

Source: `examples/results/run_experiments_perplexity_20260827_163254.csv`.

### Quantize / dequantize throughput

Random Gaussian batches of 4096 vectors. Mean of 5 timed runs after warmup.

**CUDA, `d=64`** (vectors / s):

| Algorithm | Bits | Quantize | Dequantize |
|-----------|------|----------|------------|
| mse       | 1    | 10.3M    | 25.9M      |
| mse       | 2    | 10.1M    | 18.2M      |
| mse       | 4    | 7.4M     | 20.2M      |
| prod      | 4    | 5.9M     | 16.1M      |
| polar     | 4    | 2.8M     | 3.3M       |

**CPU vs CUDA, MSE 4-bit `d=64`:** quantize 0.57M → 7.4M vec/s (~13×).
PolarQuant is slower than MSE because of the recursive polar loop.

Source: `examples/results/run_perf_benchmark_20260827_160800.csv`.

---

## File structure

```
turbo-quant/
    turboquant/
        rotation.py          Haar-random orthogonal matrix (QR)
        distributions.py     Exact Beta / sin-power densities
        lloyd_max.py         Continuous Lloyd-Max solver
        codebook.py          Cached centroids, argmin quantize
        qjl.py               Algorithm 2 projection + sign quantize
        cartesian.py         TurboQuantMSE, TurboQuantProd
        polar.py             PolarQuant
        __init__.py          Public API
    tests/                   55 tests (CPU + CUDA)
    examples/
        kv_cache_hook.py     QuantizingCache (quality harness)
        run_benchmark.py     Perplexity sweep on repeated sample text
        run_experiments.py   WikiText-2 perplexity + real-key distortion
        run_perf_benchmark.py  CPU vs CUDA latency / throughput
        results_logger.py    Timestamped CSV writer
    turboquant-primer.html
    README.md
```

---

## Installation

From the repository root (this package is a uv workspace member):

```
uv sync --all-packages
uv run pytest turbo-quant/tests -v
```

Standalone:

```
cd turbo-quant
pip install -e ".[test]"
pip install -e ".[examples]"    # transformers, datasets, for the scripts
```

---

## Usage

```python
from turboquant import TurboQuantMSE, TurboQuantProd, PolarQuant

# Algorithm 1
q = TurboQuantMSE(d=128, bits=4, seed=0)
indices, norm = q.quantize(x)          # x: (..., 128)
x_hat = q.dequantize(indices, norm)

# Algorithm 2
q = TurboQuantProd(d=128, bits=4, seed=0)
compressed = q.quantize(x)
x_hat = q.dequantize(compressed)
estimate = q.inner_product(y, compressed)   # unbiased ⟨x, y⟩

# PolarQuant (d must be a power of 2)
q = PolarQuant(d=128, bits=4, seed=0)
compressed = q.quantize(x)
x_hat = q.dequantize(compressed)
```

Pass `device="cpu"` or `device="cuda"` to pin tensors; omit it to auto-detect.

---

## Experiments

```
# Tiny smoke tests (no large download)
uv run python turbo-quant/examples/run_benchmark.py --smoke-test
uv run python turbo-quant/examples/run_experiments.py --smoke-test
uv run python turbo-quant/examples/run_perf_benchmark.py --smoke-test

# Paper-scale local sweeps (writes timestamped CSVs under examples/results/)
uv run python turbo-quant/examples/run_experiments.py --model Qwen/Qwen2.5-0.5B --bits 1 2 3 4
uv run python turbo-quant/examples/run_perf_benchmark.py --algorithms mse prod polar --bits 1 2 3 4 --dims 64 128
uv run python turbo-quant/examples/run_benchmark.py --model Qwen/Qwen2.5-0.5B --algorithm mse prod polar --bits 1 2 3 4
```

`QuantizingCache` is a **correctness/quality harness**, not a production
compressed cache. Every new key/value is quantized and immediately
dequantized; HuggingFace still stores fp16 tensors. Peak generation memory
does not drop.

CSV results are gitignored (`turbo-quant/.gitignore`).

---

## Limitations

1. **No packed KV storage in the serving path.** Indices and norms exist only
   inside `quantize` / `dequantize`. Integration into vLLM or SGLang is
   required for real memory savings.

2. **No residual window or layer-adaptive bits.** The paper's streaming
   experiments still quantize online; community V3 work kept recent tokens in
   fp16. Those heuristics are not in this package. That is why 0.5B WikiText
   perplexity is far above baseline at 1–4 bits.

3. **No mixed-precision 2.5 / 3.5-bit split** (paper §4.3 outlier channels).
   Bit-width is uniform per quantizer instance.

4. **No entropy coding of indices.** The paper skipped this for speed (~5% at
   4-bit); we match that choice.

5. **PolarQuant requires power-of-two `d`.** Covered for 64/128 head dims;
   other sizes need padding.

6. **Prod is the wrong default for attention.** Use `TurboQuantMSE` or
   `PolarQuant` when the consumer is softmax. Keep Prod for inner-product /
   retrieval-style scores.

7. **No LongBench / 100K-context numbers.** Evaluation is WikiText-2
   perplexity on Qwen2.5-0.5B plus distortion on one middle layer's keys.

8. **Python / PyTorch ops only.** No Triton fused kernels. Throughput above
   is still millions of vectors per second on CUDA, but generation with
   `QuantizingCache` pays a full round-trip per layer per step.
