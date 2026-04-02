# TurboQuant KV Cache Compression

A from-scratch PyTorch implementation of Google's **TurboQuant** algorithm for LLM
KV cache compression, based on the ICLR 2026 paper *"TurboQuant: Online Vector
Quantization with Near-optimal Distortion Rate"* by Zandieh et al.

This implementation incorporates community-informed improvements (V3) drawn from
6+ independent implementations across SGLang, llama.cpp, and research forks.

---

## Table of Contents

1. [Deviations from the Official Paper and Reference Implementation](#deviations-from-the-official-paper-and-reference-implementation)
2. [Algorithm Overview](#algorithm-overview)
3. [Mathematical Foundation](#mathematical-foundation)
4. [Implementation Architecture](#implementation-architecture)
5. [Pseudocode](#pseudocode)
6. [Compression Profiles](#compression-profiles)
7. [Current Metrics](#current-metrics)
8. [Advantages](#advantages)
9. [Limitations](#limitations)
10. [File Structure](#file-structure)
11. [Usage](#usage)
12. [Path to Production](#path-to-production)

---

## Deviations from the Official Paper and Reference Implementation

This section exhaustively documents every difference between our implementation
and (a) the published paper, (b) the Google Research blog post, and (c) the
reference PyTorch implementation at `tonbistudio/turboquant-pytorch`.

Sources compared:

- **Paper**: *"TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"*,
  Zandieh et al., ICLR 2026, [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
- **Blog**: [Google Research blog post](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) (March 24, 2026)
- **Reference repo**: [tonbistudio/turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch)

### D1. Rotation matrix: Hadamard vs Haar-distributed orthogonal

| | Paper (Alg 1, line 2) | Reference repo | Our implementation |
|---|---|---|---|
| Method | "Generate a random rotation matrix Π ∈ R^{d×d}" via QR decomposition of i.i.d. Gaussian matrix | `generate_rotation_matrix()`: QR of Gaussian matrix, matches paper | Randomized Hadamard: `fwht(x * signs)` where `signs` is a random ±1 vector |
| Randomness | d² random Gaussian entries → O(d²) to generate | Same | d random sign bits → O(d) to generate |
| Complexity | O(d²) matmul per rotate/unrotate | O(d²) | O(d log d) for compress (FWHT); O(d²) for decompress (precomputed dense matmul) |
| Distribution | Haar-uniform on O(d) — exact uniform rotation | Same | Hadamard × diagonal signs — structured random, not Haar-uniform |

**Impact**: The paper's theoretical guarantees (Theorem 1) are proven for
Haar-distributed rotations. The Hadamard rotation achieves the same
distributional concentration on coordinates for practical dimensions (d ≥ 64)
but is not strictly Haar-uniform. All production implementations (SGLang,
llama.cpp, QuaRot) use Hadamard. The paper's `generate_rotation_matrix()` is
preserved as a legacy function in our `turboquant.py` but is not used by the
active code paths.

### D2. Quantization: `torch.bucketize` vs brute-force `argmin`

| | Paper (Alg 1, line 6) | Reference repo | Our implementation |
|---|---|---|---|
| Method | `idx_j ← argmin_{k} \|y_j - c_k\|` | `diffs.abs().argmin(dim=-1)` | `torch.bucketize(y, boundaries)` |
| Complexity | O(d × 2^b) per vector | Same | O(d × b) per vector (binary search) |

**Impact**: Mathematically identical — both find the nearest centroid for each
coordinate. Since centroids are sorted, the Voronoi partition boundaries are
midpoints of consecutive centroids, and `bucketize` on those boundaries produces
the same index mapping. Our approach is asymptotically faster, especially at
higher bit-widths (4-bit: 16 comparisons → 4).

### D3. Inverse rotation: precomputed matrix vs transpose

| | Paper (Alg 1, line 10) | Reference repo | Our implementation |
|---|---|---|---|
| Method | `x̃ ← Π^T · ỹ` | `y @ self.Pi` (equivalent to Π^T since Pi is stored as Π) | `y @ self.Pi_inv` where `Pi_inv = H_d × signs` |
| Storage | Store d×d rotation matrix | Same | Store d×d inverse rotation matrix |

**Impact**: Mathematically equivalent. For the Hadamard case, the forward
rotation is `y = FWHT(x * signs) = (H_d * diag(signs)) @ x / √d · √d = ...`.
The inverse is `H_d × diag(signs)` (Hadamard is symmetric and self-inverse
after normalization). We precompute this as a dense matrix so decompress can use
a single cuBLAS `matmul`, which is faster than 7 sequential FWHT butterfly
passes at d=128 on GPU.

### D4. Norm storage precision: fp16 vs "floating-point"

| | Paper | Reference repo | Our implementation |
|---|---|---|---|
| Norm precision | "floating-point precision" (unspecified, implies fp32) | `torch.float16` | `torch.float16` |

**Impact**: The paper says "compute and store the L2 norms in floating-point
precision." Both the reference and our implementation use fp16 to halve norm
storage. This introduces a small rounding error on the norm (relative error up
to ~0.1% for typical magnitudes) but saves 2 bytes per vector. For KV cache
vectors with norms in the range 1-800, fp16 is adequate.

### D5. Coordinate distribution: Gaussian approximation vs exact Beta

| | Paper | Reference repo | Our implementation |
|---|---|---|---|
| PDF | Exact Beta: `f(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1-x²)^((d-3)/2)` | Default: Gaussian N(0,1/d); exact Beta available via `use_exact=True` | Same as reference |

**Impact**: The paper defines the exact Beta distribution and notes it "converges
to the normal distribution N(0,1/d)" in high dimensions. Both the reference and
our implementation default to the Gaussian approximation for the Lloyd-Max
solver. This is accurate for d ≥ 64 (our target is d=128). The `beta_pdf`
function is available but unused by default. The centroids and boundaries
produced are identical to within ~10^-6 for d=128.

### D6. Lloyd-Max outer integration bounds

| | Paper | Reference repo | Our implementation |
|---|---|---|---|
| Outer edges | Implicitly ±∞ (integral over full support) | `[lo * 3] + boundaries + [hi * 3]` = ±10.5σ | Same as reference |

**Impact**: The outermost integration bounds for the first and last partition
should extend to ±∞ (or ±1 for exact Beta). Both implementations use ±10.5σ
(≈ ±0.93 for d=128) as a practical finite approximation. The probability mass
beyond 10.5σ is negligible (~10^-25 for Gaussian), so this has no measurable
effect.

### D7. QJL (Algorithm 2) — implemented but not used in V3

| | Paper (Alg 2) | Reference repo | Our implementation |
|---|---|---|---|
| Presence | Core algorithm: (b-1)-bit MSE + 1-bit QJL on residual | `TurboQuantProd` in `turboquant.py`; dropped in V3 `compressors_v3.py` | `TurboQuantProd` in `turboquant.py`; dropped in V3 `compressors.py` |
| QJL matrix S | i.i.d. N(0,1) entries, S ∈ R^{d×d} | Same | Same |
| Dequant formula | `x̃_qjl = √(π/2)/d · γ · S^T · qjl` | Same | Same |
| Inner product | `⟨y, x̃_mse⟩ + ‖r‖ · √(π/2)/m · ⟨Sy, sign(Sr)⟩` | Same | Same |

**Impact**: Our `TurboQuantProd` faithfully implements Algorithm 2, including
the QJL projection matrix, sign quantization, residual norm, and unbiased inner
product estimator. However, the V3 production path (`MSECompressor`,
`TurboQuantV3`) deliberately drops QJL and allocates all bits to Lloyd-Max MSE
quantization. This is a deliberate departure based on community findings:
softmax exponentially amplifies QJL's random variance, degrading attention
quality. The paper itself only uses QJL for inner-product estimation and vector
search, not for attention through softmax. Six independent implementations
confirmed MSE-only outperforms MSE+QJL for KV cache compression.

### D8. QJL zero handling

| | Paper | Reference repo | Our implementation |
|---|---|---|---|
| sign(0) | Not specified (probability 0 for continuous distributions) | `qjl_signs[qjl_signs == 0] = 1.0` | Same |

**Impact**: The paper defines `sign(S · x)` without specifying the case when
a projection is exactly zero (probability zero for continuous Gaussian S and
continuous x). Both implementations map zero to +1. This has no practical effect.

### D9. Mixed-precision bit allocation (2.5-bit, 3.5-bit)

| | Paper (Section 4.3) | Reference repo | Our implementation |
|---|---|---|---|
| Method | Split channels into outlier/non-outlier groups; apply different TurboQuant instances with different bit-widths to each group | Not implemented | Not implemented |
| Example | 2.5-bit: 32 outlier channels at 3-bit + 96 regular channels at 2-bit → effective (32×3+96×2)/128 = 2.5 | — | — |

**Impact**: The paper achieves non-integer bit-widths (2.5, 3.5) by identifying
outlier channels and allocating them more bits. This is referenced in Table 1
where TurboQuant achieves strong LongBench scores at 2.5 and 3.5 bits. Our
implementation uses uniform bit-width per K/V component (e.g., K=4, V=2) and
does not split channels. Adding outlier-aware mixed precision would improve
compression quality at the same average bit rate.

### D10. PolarQuant (blog-described related algorithm)

| | Blog | Paper | Our implementation |
|---|---|---|---|
| Status | Described as a key component: "PolarQuant converts vectors into polar coordinates" | Referenced as a related method (PolarQuant [28]) | Not implemented |

**Impact**: PolarQuant is a separate algorithm that converts Cartesian
coordinates to polar form, achieving zero-overhead quantization through a
different mathematical approach. The blog post describes it as part of the
TurboQuant family, but the paper treats it as a related method. Our
implementation does not include PolarQuant.

### D11. Asymmetric K/V bit allocation (V3 only)

| | Paper | Reference repo | Our implementation |
|---|---|---|---|
| K/V bits | Same bit-width for both K and V | V3 supports separate K/V bits; V2 uses same bits with QJL for K, MSE for V | Separate K/V bits (e.g., K=8/V=4, K=4/V=2) |

**Impact**: The paper uses the same quantization scheme for all vectors. Our V3
exploits the empirical finding (confirmed by community) that keys require more
precision than values because errors in keys are amplified by softmax. The
reference repo's V3 implements the same asymmetry. This is an enhancement not
present in the paper.

### D12. Residual windowing (V3 only)

| | Paper | Reference repo | Our implementation |
|---|---|---|---|
| Recent tokens | All tokens quantized during streaming generation (Section 4.3) | V3: configurable `residual_window` tokens kept in fp16 | Same as reference V3 |

**Impact**: The paper explicitly states "our method applies quantization even
during the streaming generation process." Our V3 keeps the most recent
`residual_window` tokens (128-256) in full fp16 to preserve generation quality.
This reduces the effective compression ratio for short contexts but significantly
improves generation correctness. The reference repo's V3 implements identical
windowing. This is a departure from the paper's approach.

### D13. Layer-adaptive precision (V3 only)

| | Paper | Reference repo | Our implementation |
|---|---|---|---|
| Per-layer bits | Uniform across all layers | V3: first/last N layers get `protected_bits` (8-bit) | Same as reference V3 |

**Impact**: The paper applies the same quantization to all transformer layers.
Our V3 protects early and late layers (which carry positional encodings and
final predictions) with higher bit-width while compressing middle layers more
aggressively. The reference repo's V3 implements identical layer-adaptive
precision. This is a community-informed enhancement.

### D14. Bit-packed storage format

| | Paper | Reference repo | Our implementation |
|---|---|---|---|
| Storage | Stores b-bit integer indices (format unspecified) | V3: bit-packs into `uint8` with shift/mask operations | Same as reference V3 |

**Impact**: The paper describes storing indices as b-bit integers but does not
detail the packing format. Both our implementation and the reference V3 pack
multiple indices per byte (e.g., 4-bit: 2 per byte, 2-bit: 4 per byte) using
bit-shifting. The packing/unpacking logic is identical between the reference V3
and our implementation.

### D15. Codebook caching

| | Paper | Reference repo | Our implementation |
|---|---|---|---|
| Caching | "Precompute and store these optimal codebooks for a range of practically useful bit-widths" | No caching — new `LloydMaxCodebook` created each time | `_codebook_cache` dict memoizes `LloydMaxCodebook` instances by `(d, bits, use_exact)` |

**Impact**: The paper says codebooks should be precomputed once and reused. The
reference repo creates a fresh codebook each time (rerunning the Lloyd-Max
solver), which is slow when creating many compressors (e.g., one per layer). Our
implementation adds a module-level cache that avoids redundant computation. This
is a performance optimization, not an algorithmic change.

### D16. `scipy.special` import

| | Reference repo (`lloyd_max.py`) | Our implementation |
|---|---|---|
| Import | `from scipy import integrate, special` | `from scipy import integrate` |

**Impact**: The reference repo imports `scipy.special` but never uses it. Our
implementation removes the unused import. No functional difference.

### D17. Entropy encoding of codebook indices

| | Paper (Section 3.1) | Reference repo | Our implementation |
|---|---|---|---|
| Status | Described but deliberately not implemented: "We have chosen not to incorporate this technique to maintain simplicity and speed." Saves ~5% at b=4. | Not implemented | Not implemented |

**Impact**: The paper derives that entropy coding can reduce the average
bit-width by ~5% for 4-bit quantization but chose simplicity over the marginal
gain. We match this decision.

### D18. KV cache class architecture

| | Paper | Reference repo | Our implementation |
|---|---|---|---|
| V2 cache | — | `TurboQuantKVCache` in `turboquant.py`: QJL for keys, MSE for values, with `attention_scores()` for asymmetric inner product | Not present (no V2 cache class) |
| V2 compressor | — | `TurboQuantCompressorV2` in `compressors.py`: stores `k_mse` + QJL signs + residual norms; has `asymmetric_attention_scores()` | Not present |
| V3 cache | — | `TurboQuantV3` in `compressors_v3.py` | `TurboQuantV3` in `compressors.py` |
| Live generation | — | — | `V3Cache` in `evaluate.py` (subclasses `transformers.DynamicCache`) |

**Impact**: The reference repo includes V2 compressors with asymmetric attention
score computation (computing `⟨Q, K⟩` directly from compressed K using the QJL
estimator, without decompressing). We do not implement V2 compressors. Our
`V3Cache` (for live generation evaluation) is unique to our implementation and
not present in the reference repo.

### D19. Validation and evaluation approach

| | Paper | Reference repo | Our implementation |
|---|---|---|---|
| Benchmarks | LongBench, Needle-in-Haystack (4K-104K), ZeroSCROLLS, RULER, L-Eval, NNS (GloVe, DBpedia) | Needle-in-Haystack (custom), attention cosine similarity, generation test | Memory measurement, attention fidelity (cosine + top-k), compress/decompress throughput, generation speed (tok/s), needle-in-haystack |
| Models | Llama-3.1-8B-Instruct, Ministral-7B-Instruct (fp16/bf16) | Qwen2.5-3B-Instruct (4-bit via bitsandbytes) | Same as reference |
| GPU | NVIDIA A100 | RTX 3060 (12GB) | RTX 4070 Laptop (8GB) |
| Context | Up to 104K tokens | Up to 8K tokens | Up to 2K tokens (default) |

**Impact**: Our evaluation is substantially smaller in scope. The paper's
results at 3.5-bit on Llama-3.1-8B with 104K context cannot be reproduced on
our hardware. Our metrics focus on demonstrating the algorithm works correctly
rather than matching the paper's large-scale numbers.

### D20. Attention speed benchmarks

| | Paper (Figure, Section 4.3) | Reference repo | Our implementation |
|---|---|---|---|
| Reported | "4-bit TurboQuant achieves up to 8x performance increase over 32-bit unquantized keys on H100 GPU" | Not measured | Python-level throughput only; no Triton/CUDA kernels |

**Impact**: The paper's 8x attention speedup is measured on H100 with fused
JAX/XLA kernels. Our implementation runs compression/decompression as PyTorch
Python operations without fused kernels. Generation speed is ~0.62x of FP16
baseline due to Python-level overhead in the V3Cache's per-step
decompression loop. A Triton kernel would close this gap.

### D21. Reference repo V2 `asymmetric_attention_scores()` (direct compressed attention)

| | Paper (implied by Algorithm 2) | Reference repo (`compressors.py`) | Our implementation |
|---|---|---|---|
| Feature | Inner product estimated from compressed representation without decompression | `TurboQuantCompressorV2.asymmetric_attention_scores()`: computes `Q @ K_mse^T + correction` | Not implemented |

**Impact**: The reference repo implements a direct attention computation from
compressed keys that avoids full decompression. It stores `k_mse` (the MSE
reconstruction in original space) as fp16 + QJL signs + residual norms, then
computes attention as `term1 + term2` where `term2` is the QJL correction. Our
implementation always decompresses before computing attention. Since we drop QJL
in V3, the asymmetric estimator is less relevant, but the idea of computing
`Q @ K_mse^T` directly from compressed storage (without decompressing K) could
be valuable for a Triton kernel.

### D22. Reference repo stores MSE reconstruction as fp16

| | Reference repo V2 (`compressors.py`) | Our implementation |
|---|---|---|
| Storage | Stores `k_mse` as fp16 tensor (the full MSE reconstruction) alongside QJL signs and residual norms | Stores only packed indices + norms; reconstructs on demand |

**Impact**: The reference V2 compressor stores the pre-decompressed MSE
reconstruction as a full fp16 tensor, which provides zero memory savings
(actually 38% larger than uncompressed, as noted in the reference README). Our
V3 stores only bit-packed indices + fp16 norms, achieving actual compression.
The reference V3 also fixes this with bit-packing identical to ours.

### D23. Sign-flip determinism and seeding

| | Paper | Reference repo | Our implementation |
|---|---|---|---|
| Seed for rotation | Not specified | `seed` parameter to `generate_rotation_matrix()` via `torch.Generator` | `seed` parameter to `generate_hadamard_signs()` via `torch.Generator` |
| Seed for QJL S | Not specified | `seed + 1` for QJL in `TurboQuantProd` | Same: `seed + 1` |
| Per-layer seeds | Not specified | `seed + layer_idx * 1000` for key, `+500` for value | Same: `seed + layer_idx * 1000` for key, `+500` for value |

**Impact**: Seeding is an implementation detail not specified by the paper. Both
the reference and our implementation use identical per-layer seed derivation
(`seed_base = seed + layer_idx * 1000`, key compressor uses `seed_base`, value
compressor uses `seed_base + 500`). This ensures different layers and K/V get
different rotation matrices.

### D24. Near-neighbor search experiments

| | Paper (Section 4.4) | Reference repo | Our implementation |
|---|---|---|---|
| Status | Evaluated on GloVe (d=200), OpenAI3 (d=1536, d=3072); comparison with PQ and RabitQ | Not implemented | Not implemented |

**Impact**: The paper demonstrates TurboQuant's applicability to vector search
tasks beyond KV cache. Neither the reference nor our implementation includes
nearest-neighbor search experiments.

### D25. Compression during streaming generation

| | Paper (Section 4.3) | Reference repo | Our implementation |
|---|---|---|---|
| Streaming | "our method applies quantization even during the streaming generation process" — all tokens quantized including generated ones | No live generation cache implementation | `V3Cache` subclasses `DynamicCache` and compresses overflow tokens during generation |

**Impact**: Our `V3Cache` in `evaluate.py` is the only implementation across
the three codebases that actually compresses tokens on-the-fly during
`model.generate()`. However, the `transformers` library's `DynamicCache`
stores full fp16 tensors internally, so peak GPU memory is not reduced.
True streaming compression requires integration into a serving engine (vLLM,
SGLang) where the compressed format can be stored directly in paged cache.

### D26. Model quantization (4-bit model weights via bitsandbytes)

| | Paper | Reference repo | Our implementation |
|---|---|---|---|
| Model precision | fp16/bf16 (full precision or bf16) | 4-bit via bitsandbytes (default) | 4-bit via bitsandbytes (default) |

**Impact**: The paper uses full-precision models. Both the reference and our
implementation default to 4-bit model weight quantization to fit larger models
on consumer GPUs. This means the KV cache values themselves may differ slightly
from the paper's setup (since the model computes different activations at 4-bit
vs fp16). KV cache compression quality metrics are thus not directly comparable
to the paper's numbers.

---

## Algorithm Overview

TurboQuant compresses the Key-Value (KV) cache of transformer models during
inference. The KV cache grows linearly with sequence length and is the primary
memory bottleneck for long-context LLM serving. TurboQuant reduces this
footprint by 2-7x while preserving attention fidelity.

The core insight is that a **random rotation** makes any vector's coordinates
approximately i.i.d. Gaussian, regardless of the original distribution. This
allows a single, precomputed **scalar quantizer** (Lloyd-Max) to be applied
per-coordinate with provably near-optimal distortion.

### Pipeline at a Glance

```
Input KV vector (fp16, d=128)
        |
        v
  [1] Normalize to unit vector, store ||x|| as fp16 scalar
        |
        v
  [2] Randomized Hadamard rotation (FWHT + sign flips)
        |   Spreads information across all coordinates
        |   Each coordinate becomes ~ N(0, 1/d)
        v
  [3] Per-coordinate Lloyd-Max quantization (b bits)
        |   Binary search via torch.bucketize on precomputed boundaries
        |   Maps each float to one of 2^b centroids
        v
  [4] Bit-pack indices into uint8 storage
        |   e.g., 4-bit: 2 indices per byte, 2-bit: 4 indices per byte
        v
  Compressed output: packed indices (uint8) + norm (fp16)
```

### Decompression (inverse path)

```
  Compressed: packed indices + norm
        |
        v
  [1] Unpack indices from uint8
        |
        v
  [2] Look up centroid values: centroids[indices]
        |
        v
  [3] Inverse rotation via precomputed matrix (cuBLAS matmul)
        |
        v
  [4] Rescale by stored norm
        |
        v
  Reconstructed KV vector (fp16, d=128)
```

---

## Mathematical Foundation

### Why rotation works

Given a d-dimensional vector `x` with `||x|| = 1`, applying a random orthogonal
rotation `R` (Haar-distributed per the paper; Hadamard-based in our
implementation — see [D1](#d1-rotation-matrix-hadamard-vs-haar-distributed-orthogonal))
produces `y = Rx` whose coordinates are marginally distributed as:

```
f(y_i) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - y_i^2)^((d-3)/2)
```

supported on `[-1, 1]`. For practical dimensions (d >= 64), this is
well-approximated by `N(0, 1/d)`. Crucially, all coordinates share the same
marginal distribution regardless of the structure of `x`, making a single
codebook universally applicable.

### Lloyd-Max quantizer

The Lloyd-Max quantizer minimizes mean squared error (MSE) for a known
distribution. It solves the fixed-point equations:

```
boundary[i]  = (centroid[i] + centroid[i+1]) / 2
centroid[i]  = E[X | boundary[i-1] < X <= boundary[i]]
             = integral(x * f(x), boundary[i-1], boundary[i])
               / integral(f(x), boundary[i-1], boundary[i])
```

These are solved iteratively at initialization via numerical integration
(scipy.integrate.quad) and cached. At runtime, quantization is a single
`torch.bucketize` call (binary search) per coordinate.

### Distortion bound (Theorem 1 from paper)

For b-bit quantization of a d-dimensional unit vector:

```
D_mse <= sqrt(3) * pi/2 * (1/4^b)
```

Our measured distortion matches this bound:

| Bits | Paper Bound | Measured MSE | Ratio |
|------|------------|-------------|-------|
| 1    | 0.680      | 0.359       | 0.53  |
| 2    | 0.170      | 0.115       | 0.68  |
| 3    | 0.043      | 0.034       | 0.80  |
| 4    | 0.011      | 0.009       | 0.87  |

### Hadamard rotation (replacing dense orthogonal matrix)

The paper uses a Haar-distributed random orthogonal matrix via QR decomposition.
All production implementations (SGLang, llama.cpp, QuaRot) instead use a
**randomized Hadamard transform**, which we adopt (see [D1](#d1-rotation-matrix-hadamard-vs-haar-distributed-orthogonal)):

```
Rotate(x)   = FWHT(x * signs)         -- O(d log d)
Unrotate(y) = signs * FWHT(y)         -- O(d log d)
```

where `signs` is a random vector of +/-1 and FWHT is the Fast Walsh-Hadamard
Transform. This achieves the same distributional properties as a full random
rotation but in `O(d log d)` instead of `O(d^2)`.

For the decompression path in our PyTorch implementation, we precompute the
d x d inverse rotation matrix and use a cuBLAS matmul, since GPU tensor cores
make the dense matmul faster than 7 sequential FWHT butterfly operations at
d=128. In a fused Triton kernel, the O(d log d) FWHT would be preferred.

---

## Implementation Architecture

### V3 Improvements over the base paper

This implementation goes beyond the paper's Algorithm 1 with five
community-validated improvements (each linked to its deviation entry above):

1. **MSE-only (no QJL)** ([D7](#d7-qjl-algorithm-2--implemented-but-not-used-in-v3)):
   The paper's Algorithm 2 adds a 1-bit QJL correction for unbiased inner
   products. In practice, the QJL variance is amplified by softmax, hurting
   attention quality. All production implementations (SGLang, llama.cpp) drop
   QJL and allocate all bits to Lloyd-Max centroids.

2. **Asymmetric K/V bit allocation** ([D11](#d11-asymmetric-kv-bit-allocation-v3-only)):
   Keys require more precision than values because attention scores are
   `softmax(QK^T)` -- errors in K are amplified by softmax. The value cache
   tolerates aggressive compression since it's only a weighted sum.
   Typical: K=8bit / V=4bit or K=4bit / V=2bit.

3. **Residual windowing** ([D12](#d12-residual-windowing-v3-only)):
   The most recent `rw` tokens are kept in full fp16. This preserves
   generation quality since the model attends most strongly to recent context.
   Older tokens are compressed. The paper explicitly states all tokens are
   quantized during streaming; our departure improves generation correctness.

4. **Layer-adaptive precision** ([D13](#d13-layer-adaptive-precision-v3-only)):
   Early and late transformer layers are more sensitive to quantization (they
   carry position encodings and final predictions). These "protected" layers
   use higher bit-width (8-bit) regardless of the profile setting. Middle
   layers tolerate aggressive compression.

5. **Bit-packed storage** ([D14](#d14-bit-packed-storage-format)):
   Indices are packed into uint8 for actual memory reduction. For example,
   4-bit quantization packs 2 indices per byte; 2-bit packs 4 per byte.

Additionally, we replace the paper's rotation and quantization methods with
faster alternatives:

6. **Hadamard rotation** ([D1](#d1-rotation-matrix-hadamard-vs-haar-distributed-orthogonal)):
   O(d log d) FWHT replaces O(d²) dense random orthogonal matmul.

7. **Binary search quantization** ([D2](#d2-quantization-torchbucketize-vs-brute-force-argmin)):
   O(d log k) `torch.bucketize` replaces O(d × k) brute-force argmin.

### Class hierarchy

```
LloydMaxCodebook          Precomputed optimal scalar quantizer (cached)
    |
TurboQuantMSE             Stage 1: FWHT rotation + Lloyd-Max + bucketize
    |
TurboQuantProd            Stage 1 + Stage 2: adds 1-bit QJL correction
                          (included for paper completeness; not used in V3)

MSECompressor             Production compressor: normalize, rotate, quantize, bit-pack
    |
TurboQuantV3              Full KV cache compressor: asymmetric K/V, residual window,
                          layer-adaptive precision, per-layer MSECompressor instances
```

---

## Pseudocode

### Initialization (once at model load)

```
function INIT_COMPRESSOR(head_dim, bits, seed):
    signs       <- random {+1, -1} vector of length head_dim, seeded
    centroids   <- solve_lloyd_max(head_dim, bits)    // 2^bits optimal centroids
    boundaries  <- midpoints(centroids)               // 2^bits - 1 decision thresholds
    Pi_inv      <- hadamard_matrix(head_dim) * signs  // d x d inverse rotation (precomputed)
    return {signs, centroids, boundaries, Pi_inv}
```

### Compress (encode path)

```
function COMPRESS(states, compressor):
    // states: tensor of shape (batch, heads, seq_len, head_dim), dtype=fp16
    // Returns: packed indices (uint8) + norms (fp16)

    flat <- reshape(states, [N, D])              // N = batch * heads * seq_len
    norms <- L2_norm(flat, dim=-1)               // shape (N,)
    flat_normalized <- flat / (norms + epsilon)   // unit vectors

    // Randomized Hadamard rotation: O(d log d)
    rotated <- FWHT(flat_normalized * signs)      // spread info across coords

    // Quantize via binary search: O(d log k) where k = 2^bits
    indices <- bucketize(rotated, boundaries)     // shape (N, D), values in [0, 2^bits)

    // Bit-pack into uint8
    indices_per_byte <- 8 / bits
    packed <- bit_pack(indices, bits)             // shape (N, D / indices_per_byte)

    return {packed_indices: packed, norms: fp16(norms)}
```

### Decompress (decode path)

```
function DECOMPRESS(compressed, compressor):
    // compressed: {packed_indices, norms}
    // Returns: reconstructed tensor of shape (batch, heads, seq_len, head_dim)

    indices <- bit_unpack(compressed.packed_indices, bits)  // (N, D)

    // Look up centroid values
    quantized <- centroids[indices]               // (N, D) float32

    // Inverse rotation via precomputed matrix: O(d^2) cuBLAS matmul
    reconstructed <- quantized @ Pi_inv           // (N, D)

    // Rescale by stored norms
    reconstructed <- reconstructed * norms        // (N, D)

    return reshape(reconstructed, original_shape)
```

### TurboQuantV3: Full KV cache compression

```
function COMPRESS_KV(keys, values, layer_idx, config):
    // Determine effective bit-widths (layer-adaptive)
    if layer_idx is in protected range:
        key_bits <- 8
        val_bits <- 8
    else:
        key_bits <- config.key_bits    // e.g., 4
        val_bits <- config.value_bits  // e.g., 2

    // Residual windowing: keep recent tokens in fp16
    rw <- config.residual_window
    if seq_len <= rw:
        return {keys, values}  // too short to compress, keep fp16

    split_at <- seq_len - rw

    // Compress older tokens
    compressed_K <- COMPRESS(keys[:, :, :split_at, :], key_compressor)
    compressed_V <- COMPRESS(values[:, :, :split_at, :], val_compressor)

    // Keep recent tokens in fp16
    recent_K <- keys[:, :, split_at:, :]
    recent_V <- values[:, :, split_at:, :]

    return {compressed_K, recent_K, compressed_V, recent_V}
```

```
function DECOMPRESS_KV(compressed_kv):
    old_K <- DECOMPRESS(compressed_kv.compressed_K, key_compressor)
    old_V <- DECOMPRESS(compressed_kv.compressed_V, val_compressor)

    keys   <- concatenate(old_K, compressed_kv.recent_K, dim=seq)
    values <- concatenate(old_V, compressed_kv.recent_V, dim=seq)

    return keys, values
```

### Lloyd-Max solver (offline, cached)

```
function SOLVE_LLOYD_MAX(d, bits, max_iter=200):
    k <- 2^bits
    pdf <- gaussian_approx(mean=0, var=1/d)
    sigma <- 1 / sqrt(d)

    // Initialize centroids uniformly in [-3.5*sigma, 3.5*sigma]
    centroids <- linspace(-3.5*sigma, 3.5*sigma, k)

    for iter in 1..max_iter:
        // Update boundaries to midpoints
        boundaries[i] <- (centroids[i] + centroids[i+1]) / 2

        // Update centroids to conditional expectations
        for i in 0..k-1:
            a, b <- boundaries around centroid i
            centroids[i] <- integral(x * pdf(x), a, b) / integral(pdf(x), a, b)

        if max_shift < tolerance:
            break

    return centroids, boundaries
```

### Fast Walsh-Hadamard Transform

```
function FWHT(x):
    // x: tensor with last dimension d (power of 2)
    // Returns: H @ x / sqrt(d) where H is the Hadamard matrix
    // Self-inverse: FWHT(FWHT(x)) = x

    d <- x.shape[-1]
    h <- 1
    while h < d:
        // Butterfly operation on pairs of blocks of size h
        reshape x to (..., d/(2h), 2, h)
        a <- x[..., 0, :] + x[..., 1, :]    // sum
        b <- x[..., 0, :] - x[..., 1, :]    // difference
        x <- stack(a, b, dim=-2)
        reshape x to (..., d)
        h <- h * 2

    return x / sqrt(d)
```

---

## Compression Profiles

Two named presets are provided:

### Moderate (safe for production)

| Parameter | Value |
|-----------|-------|
| Key bits | 8 |
| Value bits | 4 |
| Residual window | 128 tokens |
| Protected layers | 4 (first 4 + last 4) |
| Target compression | ~2.2x |

### Extreme (throughput-critical deployments)

| Parameter | Value |
|-----------|-------|
| Key bits | 4 |
| Value bits | 2 |
| Residual window | 256 tokens |
| Protected layers | 6 (first 6 + last 6) |
| Target compression | ~2.6x |

---

## Current Metrics

Measured on **Qwen2.5-3B-Instruct**, 2046-token context, NVIDIA RTX 4070
Laptop GPU (8 GB), 4-bit model quantization via bitsandbytes.

### Memory

| Profile | FP16 Cache | Compressed | Ratio |
|---------|-----------|-----------|-------|
| moderate | 71.9 MB | 32.2 MB | **2.2x** |
| extreme | 71.9 MB | 27.8 MB | **2.6x** |

### Attention Output Fidelity

Full `softmax(QK^T / sqrt(d)) @ V` computed with original vs. decompressed KV.

| Profile | Mean Cosine | Min Cosine | Worst Layer | Top-1 Match |
|---------|------------|-----------|-------------|-------------|
| moderate | 0.9959 | 0.9461 | layer 0 | 98.3% |
| extreme | 0.9669 | 0.8944 | layer 27 | 93.8% |

### Throughput (Python, no Triton)

| Profile | Compress | Decompress | Comp GB/s | Dec GB/s |
|---------|---------|-----------|----------|---------|
| moderate | 1.61 M tok/s | 6.67 M tok/s | 0.82 | 3.42 |
| extreme | 1.73 M tok/s | 7.03 M tok/s | 0.89 | 3.60 |

### Generation Speed

Actual `model.generate()` with needle-in-haystack quality check.

| Config | tok/s | vs FP16 | Quality |
|--------|-------|---------|---------|
| FP16 baseline | 3.1 | 1.00x | FOUND |
| moderate | 1.9 | 0.62x | FOUND |
| extreme | 1.9 | 0.62x | FOUND |

Both profiles find the needle (AURORA-7749) correctly, producing identical
responses to the FP16 baseline.

### KV Tensor Round-trip Fidelity

| Profile | Key Mean Cos | Key Min Cos | Value Mean Cos | Value Min Cos |
|---------|-------------|------------|---------------|--------------|
| moderate | 0.99997 | 0.99997 | 0.99661 | 0.99515 |
| extreme | 0.99736 | 0.99478 | 0.96557 | 0.94673 |

---

## Advantages

1. **Data-oblivious**: No calibration data, no training, no model-specific
   tuning. The codebook depends only on dimension `d` and bit-width `b`, both
   known at model load time.

2. **Provably near-optimal**: The distortion is bounded by Theorem 1 and
   matches the rate-distortion limit. You cannot do significantly better with
   any scalar quantizer on rotated coordinates.

3. **Online compression**: Each token's KV can be compressed independently as
   it arrives. No need to batch or see the full sequence.

4. **Drop-in compatible**: Works with any transformer model that uses standard
   multi-head attention with a KV cache. No architecture changes needed.

5. **Tunable quality-compression trade-off**: Bit-width, residual window, and
   protected layers give fine-grained control. The moderate profile achieves
   2.2x compression with 99.6% attention fidelity.

6. **Asymmetric K/V**: Exploits the empirical finding that values tolerate much
   more aggressive compression than keys, maximizing compression without
   sacrificing the softmax-sensitive key precision.

7. **Layer-adaptive**: Automatically protects sensitive early/late layers while
   compressing the more redundant middle layers.

8. **Fast rotation**: The Hadamard rotation is O(d log d) for compression and
   uses a precomputed cuBLAS matmul for decompression, making both paths
   efficient on GPU.

---

## Limitations

1. **Python-level overhead**: The current implementation runs compression and
   decompression as PyTorch operations with Python loop overhead. Generation
   speed is ~0.62x of FP16 baseline because every decoding step decompresses
   all cached layers through Python. Fused Triton/CUDA kernels would eliminate
   this overhead.

2. **No runtime memory savings in HuggingFace**: The `transformers` library's
   `DynamicCache` stores full fp16 tensors. Our V3Cache decompresses into these
   tensors on every step, so peak GPU memory during generation is not reduced.
   True runtime savings require integration into a serving engine (vLLM, SGLang)
   where the compressed format can be stored directly in the paged cache.

3. **Compression ratio at short contexts**: With a residual window of 128-256
   tokens, short sequences (< 512 tokens) see limited benefit because most
   tokens fall within the fp16 window. The ratio improves with longer contexts.

4. **Quality depends on K-norm amplification**: Models with large key vector
   norms relative to the random baseline (K-norm amplification > 2x) show
   reduced token-level match. Qwen models (~2.4x) are less favorable than
   Llama/Mistral models (~1.3x). This is a property of the model, not the
   algorithm.

5. **Power-of-2 head dimension required**: The FWHT requires the head dimension
   to be a power of 2 (64, 128, 256). This covers all mainstream models but
   would need padding for non-standard dimensions.

6. **No mixed-precision (2.5-bit, 3.5-bit)** ([D9](#d9-mixed-precision-bit-allocation-25-bit-35-bit)):
   The paper describes mixed-precision configurations that split channels into
   outlier/non-outlier groups with different bit-widths. This implementation
   uses uniform bit-width per K/V component. Adding mixed-precision is
   straightforward but not yet implemented.

7. **QJL mode not recommended** ([D7](#d7-qjl-algorithm-2--implemented-but-not-used-in-v3)):
   The Stage 2 QJL correction (`TurboQuantProd`) is implemented for paper
   completeness but degrades generation quality in practice. The MSE-only
   path (`TurboQuantMSE`, `MSECompressor`) is used by all production deployments.

8. **No PolarQuant** ([D10](#d10-polarquant-blog-described-related-algorithm)):
   The Google blog post describes PolarQuant as a companion algorithm. It is a
   separate method not part of the TurboQuant paper's Algorithms 1-2 and is
   not implemented here.

9. **No asymmetric attention** ([D21](#d21-reference-repo-v2-asymmetric_attention_scores-direct-compressed-attention)):
   The reference repo includes a V2 mode that computes `Q @ K^T` directly
   from compressed representations without decompressing K. We always
   decompress first. Asymmetric attention could be valuable in a fused kernel.

10. **No nearest-neighbor search** ([D24](#d24-near-neighbor-search-experiments)):
    The paper demonstrates TurboQuant for vector search tasks. Our
    implementation focuses exclusively on KV cache compression.

---

## File Structure

```
turbo-quant/
    lloyd_max.py       Lloyd-Max optimal scalar quantizer solver
    turboquant.py      Core: FWHT, Hadamard rotation, TurboQuantMSE, TurboQuantProd
    compressors.py     Production: MSECompressor (bit-packed), TurboQuantV3, profiles
    evaluate.py        Evaluation: memory, fidelity, throughput, generation speed
    test_algorithm.py  Synthetic algorithm correctness tests (no model needed)
    README.md          This file
```

---

## Usage

### Run algorithm tests (no GPU or model needed)

```bash
python test_algorithm.py
```

### Run full evaluation on a model

```bash
python evaluate.py --model Qwen/Qwen2.5-3B-Instruct --context 2048

# With specific profiles
python evaluate.py --profiles moderate extreme --context 4096

# Skip generation tests (faster, compression-only)
python evaluate.py --skip-generation

# Use fp16 model instead of 4-bit
python evaluate.py --no-4bit
```

### Use in your own code

```python
from compressors import TurboQuantV3, PROFILES

profile = PROFILES["moderate"]
compressor = TurboQuantV3(
    head_dim=128,
    key_bits=profile.key_bits,
    value_bits=profile.value_bits,
    residual_window=profile.residual_window,
    layer_idx=5,
    n_layers=36,
    protected_layers=profile.protected_layers,
    device="cuda",
)

# Compress
compressed_k, compressed_v = compressor.compress_kv(keys, values)

# Decompress
keys_recon, values_recon = compressor.decompress_kv(compressed_k, compressed_v)
```

---

## Path to Production

1. **Triton kernels**: Fuse the entire compress (sign-flip -> FWHT -> bucketize
   -> bit-pack) and decompress (unpack -> lookup -> matmul -> rescale) paths into
   single Triton kernels. The SGLang PR (#21419) has working reference code.
   Expected: 50-100x throughput improvement, generation speed matching FP16.

2. **vLLM integration**: Add `turboquant` as a `--kv-cache-dtype` option.
   Store compressed uint8 + fp16 norms directly in vLLM's paged cache blocks.
   Decompress inside the attention kernel. This delivers real runtime memory
   savings.

3. **SGLang integration**: Same pattern via their `RadixAttention` cache manager.
   An open PR exists at `sgl-project/sglang#21419`.

4. **Mixed-precision**: Implement the paper's 2.5-bit and 3.5-bit configurations
   by splitting channels into outlier/non-outlier groups with two independent
   TurboQuant instances.
