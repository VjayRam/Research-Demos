# TurboQuant KV Cache Compression

A from-scratch PyTorch implementation of Google's **TurboQuant** algorithm for LLM
KV cache compression, based on the ICLR 2026 paper *"TurboQuant: Online Vector
Quantization with Near-optimal Distortion Rate"* by Zandieh et al.

This implementation incorporates community-informed improvements (V3) drawn from
6+ independent implementations across SGLang, llama.cpp, and research forks.

---

## Table of Contents

1. [Algorithm Overview](#algorithm-overview)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Implementation Architecture](#implementation-architecture)
4. [Pseudocode](#pseudocode)
5. [Compression Profiles](#compression-profiles)
6. [Current Metrics](#current-metrics)
7. [Advantages](#advantages)
8. [Limitations](#limitations)
9. [File Structure](#file-structure)
10. [Usage](#usage)
11. [Path to Production](#path-to-production)

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

Given a d-dimensional vector `x` with `||x|| = 1`, applying a Haar-distributed
random orthogonal rotation `R` produces `y = Rx` whose coordinates are
marginally distributed as:

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

The paper and all production implementations use a **randomized Hadamard
transform** instead of a full random orthogonal matrix:

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
community-validated improvements:

1. **MSE-only (no QJL)**: The paper's Algorithm 2 adds a 1-bit QJL correction
   for unbiased inner products. In practice, the QJL variance is amplified by
   softmax, hurting attention quality. All production implementations (SGLang,
   llama.cpp) drop QJL and allocate all bits to Lloyd-Max centroids.

2. **Asymmetric K/V bit allocation**: Keys require more precision than values
   because attention scores are `softmax(QK^T)` -- errors in K are amplified
   by softmax. The value cache tolerates aggressive compression since it's
   only a weighted sum. Typical: K=8bit / V=4bit or K=4bit / V=2bit.

3. **Residual windowing**: The most recent `rw` tokens are kept in full fp16.
   This preserves generation quality since the model attends most strongly to
   recent context. Older tokens are compressed.

4. **Layer-adaptive precision**: Early and late transformer layers are more
   sensitive to quantization (they carry position encodings and final
   predictions). These "protected" layers use higher bit-width (8-bit)
   regardless of the profile setting. Middle layers tolerate aggressive
   compression.

5. **Bit-packed storage**: Indices are packed into uint8 for actual memory
   reduction. For example, 4-bit quantization packs 2 indices per byte; 2-bit
   packs 4 per byte.

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

6. **No mixed-precision (2.5-bit, 3.5-bit)**: The paper describes mixed-precision
   configurations that split channels into outlier/non-outlier groups with
   different bit-widths. This implementation uses uniform bit-width per
   K/V component. Adding mixed-precision is straightforward but not yet
   implemented.

7. **QJL mode not recommended**: The Stage 2 QJL correction (`TurboQuantProd`)
   is implemented for paper completeness but degrades generation quality in
   practice. The MSE-only path (`TurboQuantMSE`, `MSECompressor`) is used by
   all production deployments.

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
