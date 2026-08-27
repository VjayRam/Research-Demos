# I Implemented Google's TurboQuant Paper From Scratch — Here's What Actually Happened

*Implementing an ICLR 2026 paper on KV cache compression, discovering the gap between
theory and practice, and building something that actually works.*

---

## TL;DR

I took Google's TurboQuant paper (ICLR 2026) — a vector quantization algorithm that
promises 5x KV cache compression with near-optimal distortion — and implemented it
from scratch in PyTorch. The paper's core algorithm (random rotation + Lloyd-Max
quantization) works beautifully. Its flagship feature (QJL residual correction) is
mathematically elegant but **actively hurts** real generation quality. After hitting
a wall with the naive implementation, I collaborated with an AI coding agent (Claude
in Cursor) through ~25 iterative sessions to build a V3 variant informed by 6+
community implementations. Final result: **2.2–2.6x real compression** with
**99.6% attention fidelity** and correct needle-in-haystack retrieval. Not the 5x the
paper claims — but honest, reproducible, and a solid foundation for kernel-level
production deployment.

---

## Stack

| Layer | Tool | Why |
|-------|------|-----|
| Language | Python 3.12 | Fast prototyping, ecosystem |
| Deep Learning | PyTorch 2.11 (CUDA 12.8) | GPU tensors, autograd, `torch.bucketize` |
| Models | HuggingFace Transformers 5.3 | `AutoModelForCausalLM`, `DynamicCache` |
| Numerical | SciPy 1.17 | `integrate.quad` for Lloyd-Max codebook solving |
| Quantization | bitsandbytes 0.45 | 4-bit NF4 model weight quantization to fit on laptop GPU |
| Model loading | Accelerate 1.7 | `device_map="auto"`, mixed CPU/GPU offload |
| Package manager | uv | Fast dependency resolution |
| Hardware | NVIDIA RTX 4070 Laptop GPU (8 GB VRAM) | Consumer-grade, forces honest engineering |
| IDE | Cursor (with Claude agent) | AI-assisted pair programming |
| Version control | Git | Tracking the evolution of failed experiments |
| OS | Windows 11 | Yes, really |

No Docker, no Kubernetes, no distributed anything. One GPU, one script, `python evaluate.py`.

---

## The Architecture / System Design

### How the pieces talk to each other

```
┌─────────────────────────────────────────────────────────────────────┐
│                        evaluate.py (orchestrator)                   │
│  Loads model → Captures KV cache → Runs 5 measurement phases       │
└────────────┬──────────────────────────────┬─────────────────────────┘
             │                              │
             ▼                              ▼
┌────────────────────────┐    ┌──────────────────────────────────────┐
│   HuggingFace Model    │    │         TurboQuantV3                 │
│  (Qwen2.5-3B-Instruct) │    │  ┌──────────┐  ┌──────────────────┐ │
│                        │    │  │ Layer-    │  │ MSECompressor(K) │ │
│  4-bit NF4 weights     │    │  │ Adaptive  │  │ MSECompressor(V) │ │
│  via bitsandbytes      │    │  │ Precision │  │   (per layer)    │ │
│                        │    │  └──────────┘  └────────┬─────────┘ │
│  Outputs:              │    │                         │           │
│  past_key_values       │──▶│  Residual Window:       │           │
│  (DynamicCache)        │    │  Recent tokens → fp16   │           │
│                        │    │  Older tokens ──────────▼           │
└────────────────────────┘    │                                     │
                              │  ┌─────────────────────────────┐    │
                              │  │       MSECompressor         │    │
                              │  │                             │    │
                              │  │  Compress:                  │    │
                              │  │  x ──▶ normalize ──▶ FWHT   │    │
                              │  │    ──▶ bucketize ──▶ pack   │    │
                              │  │                             │    │
                              │  │  Decompress:                │    │
                              │  │  unpack ──▶ lookup ──▶      │    │
                              │  │  matmul(Pi_inv) ──▶ rescale │    │
                              │  └──────────────┬──────────────┘    │
                              │                 │                   │
                              └─────────────────┼───────────────────┘
                                                │
                              ┌─────────────────▼───────────────────┐
                              │         LloydMaxCodebook            │
                              │  (cached, solved once at startup)   │
                              │                                     │
                              │  scipy.integrate.quad solves the    │
                              │  continuous 1-D k-means problem     │
                              │  for N(0, 1/d) distribution         │
                              │                                     │
                              │  Outputs: centroids + boundaries    │
                              └─────────────────────────────────────┘
```

### The compress/decompress data flow

```
         COMPRESS (per vector, d=128)                    DECOMPRESS
    ┌─────────────────────────────┐              ┌─────────────────────────┐
    │ fp16 vector (256 bytes)     │              │ packed uint8 + fp16 norm│
    │         │                   │              │         │               │
    │         ▼                   │              │         ▼               │
    │  L2 norm → store as fp16   │              │  unpack indices         │
    │  x_norm = x / ||x||        │              │         │               │
    │         │                   │              │         ▼               │
    │         ▼                   │              │  centroids[indices]     │
    │  FWHT(x_norm * signs)      │              │         │               │
    │  O(d log d) butterfly ops  │              │         ▼               │
    │         │                   │              │  matmul @ Pi_inv        │
    │         ▼                   │              │  O(d²) via cuBLAS      │
    │  bucketize(rotated, bounds)│              │         │               │
    │  O(d log k) binary search  │              │         ▼               │
    │         │                   │              │  × stored norm          │
    │         ▼                   │              │         │               │
    │  bit-pack into uint8       │              │         ▼               │
    │  4-bit: 2 per byte         │              │  fp16 vector (256 bytes)│
    │  2-bit: 4 per byte         │              │                         │
    │         │                   │              └─────────────────────────┘
    │         ▼                   │
    │  packed (64 bytes @ 4-bit) │
    │  + norm (2 bytes)          │
    │  = 66 bytes total          │
    │  Compression: 3.9x         │
    └─────────────────────────────┘
```

---

## Idea & Research

### The paper's promise

In March 2026, Google published [TurboQuant](https://arxiv.org/abs/2504.19874) at
ICLR — a vector quantization algorithm that compresses the KV cache of transformer
models. The pitch:

> *"We achieve absolute quality neutrality with 3.5 bits per channel and marginal
> quality degradation with 2.5 bits per channel."*

The key insight is elegant: multiply any vector by a random orthogonal matrix. This
makes every coordinate independently follow a known distribution (Beta, which
approximates Gaussian for d ≥ 64). Since you know the distribution beforehand, you
can precompute the *optimal* scalar quantizer (Lloyd-Max) for each coordinate. No
calibration data. No training. No model-specific tuning. Just math.

The paper has two algorithms:
- **Algorithm 1 (TurboQuant_mse)**: Random rotation → per-coordinate Lloyd-Max
  quantization. Minimizes mean squared error. Provably within 2.7x of the
  information-theoretic lower bound.
- **Algorithm 2 (TurboQuant_prod)**: Algorithm 1 with (b-1) bits + a 1-bit QJL
  (Quantized Johnson-Lindenstrauss) correction on the residual. Makes inner product
  estimation mathematically *unbiased*.

The [Google blog post](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
adds PolarQuant (Cartesian → polar coordinate conversion) as a companion technique
and shows 8x speedup on H100 GPUs with fused JAX kernels.

I wanted to see this work on my laptop, on a real model, producing real text.

### The reference implementation

The community reference at [tonbistudio/turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch)
(725 stars) had already discovered something important: **QJL kills generation quality**.
Six independent teams across Python, C, and Rust implementations confirmed it. The
issue is that attention runs scores through softmax, which exponentially amplifies
variance. QJL's random noise, while unbiased in expectation, gets magnified into
garbage after exponentiation.

Their V3 variant drops QJL, adds asymmetric K/V bit allocation, and includes a
"residual window" of recent tokens kept in full fp16. This became my starting point.

---

## Implementation

I worked with Claude (Anthropic's AI model, running as an agent in Cursor IDE) through
approximately 25 iterative sessions. The workflow was genuinely collaborative: I would
describe what I wanted, Claude would implement and run code, I would examine outputs
and push back when something looked wrong, and we'd iterate. Some of the most
important decisions came from me saying "this doesn't look right" and forcing a
re-examination.

### What we built

**`lloyd_max.py`** — The Lloyd-Max optimal scalar quantizer solver. Uses
`scipy.integrate.quad` to solve the continuous 1-D k-means problem for the Gaussian
approximation of the Beta distribution. Precomputes centroids and boundaries once,
then caches them. This is the mathematical heart of the algorithm.

**`turboquant.py`** — Core rotation and quantization primitives:
- `fwht()`: Fast Walsh-Hadamard Transform — O(d log d) butterfly operations
- `generate_hadamard_signs()`: Random ±1 sign vector for randomized rotation
- `build_inverse_rotation()`: Precomputed dense d×d matrix for fast GPU decompress
- `TurboQuantMSE`: Stage 1 quantizer (what we actually use)
- `TurboQuantProd`: Stage 1 + QJL (implemented for completeness, not used in production)

**`compressors.py`** — Production compression layer:
- `MSECompressor`: Normalize → FWHT → bucketize → bit-pack (compress); unpack →
  lookup → cuBLAS matmul → rescale (decompress)
- `TurboQuantV3`: Orchestrates per-layer K/V compression with asymmetric bits,
  residual windowing, and layer-adaptive precision
- `CompressionProfile`: Named presets (moderate: K8/V4, extreme: K4/V2)

**`evaluate.py`** — Honest evaluation harness measuring 5 things: (1) actual GPU
memory of compressed tensors, (2) attention output fidelity via cosine similarity,
(3) compress/decompress throughput, (4) KV tensor round-trip cosine, and (5) actual
generation speed with needle-in-haystack quality checks.

**`test_algorithm.py`** — Synthetic tests validating distortion against the paper's
theoretical bounds (no model required).

### Key technical decisions

1. **Hadamard instead of random orthogonal**: The paper uses a dense random rotation
   via QR decomposition — O(d²). We use FWHT + random sign flips — O(d log d).
   Every production implementation (SGLang, llama.cpp, QuaRot) makes this same swap.
   The distributional properties are equivalent for d=128.

2. **`torch.bucketize` instead of `argmin`**: The paper finds the nearest centroid
   by computing distances to all centroids — O(d × 2^b). Since centroids are sorted,
   binary search on boundaries gives the same result in O(d × b). At 4-bit (16
   centroids), this is 4 comparisons instead of 16.

3. **Hybrid rotation strategy**: FWHT for compression (avoid large intermediates),
   precomputed dense matrix for decompression (cuBLAS is faster than 7 sequential
   butterfly passes at d=128). This was discovered after a painful performance
   regression — more on that below.

4. **fp16 norm storage**: The paper says "floating-point precision" for norms. We
   use fp16 (2 bytes) instead of fp32 (4 bytes). For KV cache vectors with norms
   in range 1–800, the ~0.1% relative error from fp16 is invisible.

---

## The Bottleneck / Challenges

This is where the real story is. Three things broke badly.

### Challenge 1: The V3Cache Illusion

My first request was straightforward: "I want to see the compressed model and its
performance in terms of resource consumption and generation." Claude built a
`V3Cache` that subclassed HuggingFace's `DynamicCache`, compressing tokens on the
fly during `model.generate()`.

It worked. The model generated correct text. The needle-in-haystack test passed.
I was excited.

Then I looked at the memory numbers. **Compressed memory was higher than fp16.**

The problem: HuggingFace's `DynamicCache` stores `layer.keys` and `layer.values` as
full fp16 tensors. Our V3Cache dutifully compressed overflow tokens into packed uint8 —
then immediately decompressed them back into fp16 tensors to hand back to the
attention computation. The compressed representation existed momentarily, but the
cache always held the full fp16 reconstruction.

I pushed back: *"Is this how it is actually implemented in the reference repository
and in the paper as well?"* and *"I want a legitimate and true implementation...
it should show tangible results which I can later work on to deploy."*

This was the critical inflection point. Claude acknowledged the approach was a hack
and we pivoted to an honest evaluation design: measure compression quality *separately*
from runtime integration, don't fake memory savings, and be transparent about what
Python-level code can and can't do.

**Lesson**: The `transformers` library was never designed for compressed KV storage.
True runtime memory savings require integration at the serving engine level (vLLM,
SGLang) where you control the paged cache format directly.

### Challenge 2: Lloyd-Max Solver Performance

After building the evaluation harness, the first run hung at "Phase 1: Measuring
memory..." for over 10 minutes.

The culprit: `LloydMaxCodebook` was being instantiated freshly for every layer ×
every compressor. With 36 layers × 2 compressors (K and V) × 2 profiles = 144
codebook solves, each running 200 iterations of `scipy.integrate.quad`... the math
was correct but absurdly slow.

The fix was embarrassingly simple: a module-level cache dictionary keyed by
`(d, bits, use_exact)`. Since all layers with the same bit-width use identical
codebooks (they depend only on dimension and bits, not on the data), we solve once
and reuse. Runtime dropped from 10+ minutes to under 30 seconds.

```python
_codebook_cache: dict[tuple, "LloydMaxCodebook"] = {}

class LloydMaxCodebook:
    def __new__(cls, d: int, bits: int, use_exact: bool = False):
        key = (d, bits, use_exact)
        if key in _codebook_cache:
            return _codebook_cache[key]
        instance = super().__new__(cls)
        instance._initialized = False
        return instance
```

**Lesson**: Profile before optimizing the algorithm. The bottleneck wasn't the
rotation or quantization — it was redundant codebook initialization.

### Challenge 3: The FWHT Speed Regression

This one hurt the most because the optimization made things *worse*.

The original implementation used a dense O(d²) random rotation matrix. We replaced
it with O(d log d) FWHT for both compress and decompress, expecting a speedup.
Compression got faster. But **generation speed cratered from 0.62x to 0.18x of
the FP16 baseline**.

The problem: FWHT is 7 sequential butterfly passes at d=128. Each pass reads and
writes the entire tensor. On a GPU, this means 7 kernel launches with memory
round-trips. A dense 128×128 matmul, by contrast, is a *single* cuBLAS call that
runs entirely in tensor cores with optimal memory access patterns.

For compression, FWHT still wins because it avoids materializing the d×d intermediate
rotation matrix (important when processing thousands of vectors). For decompression —
which happens on every single generation step — the dense matmul is 3-4x faster on
GPU because cuBLAS is absurdly well-optimized.

The fix was a **hybrid strategy**:
- **Compress path**: FWHT + `torch.bucketize` (O(d log d), avoids large intermediates)
- **Decompress path**: Precomputed dense matrix + cuBLAS matmul (O(d²), but faster on GPU)

```python
def build_inverse_rotation(d, signs, device="cpu"):
    H = _build_hadamard_matrix(d, device=device)
    return H * signs.to(device)  # d × d dense matrix

# Compress: O(d log d)
rotated = fwht(flat_norm * self.signs)

# Decompress: O(d²) but single cuBLAS call
reconstructed = self.centroids[indices] @ self.Pi_inv
```

Generation speed recovered from 0.18x to 0.62x.

**Lesson**: Asymptotic complexity doesn't determine GPU performance. Memory access
patterns and kernel launch overhead dominate. A "slower" O(d²) algorithm can beat a
"faster" O(d log d) one if it maps to a single optimized hardware primitive.

### Bonus Challenges

- **`DynamicCache` API changes**: `transformers` 5.x removed `_seen_tokens` and
  changed the cache from subscriptable (`cache[layer_idx]`) to attribute-based
  (`cache.layers[layer_idx].keys`). Broke our V3Cache twice.

- **Unicode on Windows**: Benchmark output used Unicode box-drawing characters.
  Windows PowerShell's default encoding (`charmap`) can't render them. Replaced with
  ASCII dashes.

- **`torch_dtype` deprecation**: `torch_dtype` parameter was renamed to `dtype` in
  newer transformers. A one-line fix after a confusing deprecation warning.

- **Top-5 fidelity bug**: A loop variable `_` that should have been `q` in the
  top-5 overlap calculation. Every query was being compared against query 0's top-5
  instead of its own. The metric looked good because it was wrong.

---

## Results / Fixes

### Final metrics

Measured on **Qwen2.5-3B-Instruct**, 2046-token context, NVIDIA RTX 4070 Laptop
GPU (8 GB), 4-bit model quantization.

| Profile | FP16 Cache | Compressed | Ratio | Attention Cosine | Top-1 Match |
|---------|-----------|-----------|-------|-----------------|-------------|
| moderate (K8/V4) | 71.9 MB | 32.2 MB | **2.2x** | 0.9959 | 98.3% |
| extreme (K4/V2) | 71.9 MB | 27.8 MB | **2.6x** | 0.9669 | 93.8% |

| Config | Generation tok/s | vs FP16 | Needle-in-Haystack |
|--------|-----------------|---------|-------------------|
| FP16 baseline | 3.1 | 1.00x | FOUND |
| moderate (K8/V4) | 1.9 | 0.62x | FOUND |
| extreme (K4/V2) | 1.9 | 0.62x | FOUND |

The algorithm's distortion matches the paper's theoretical bounds:

| Bits | Paper's Upper Bound | Measured MSE | Ratio to Bound |
|------|-------------------|-------------|----------------|
| 1 | 0.680 | 0.359 | 0.53x |
| 2 | 0.170 | 0.115 | 0.68x |
| 3 | 0.043 | 0.034 | 0.80x |
| 4 | 0.011 | 0.009 | 0.87x |

### What's honest about these numbers

- Memory ratio measures **actual tensor bytes** in the compressed representation,
  not theoretical bit counts
- Attention fidelity computes the full `softmax(QK^T/√d) @ V` with original vs
  decompressed KV — not just raw cosine between K vectors
- Generation speed is real `model.generate()` wall-clock time, not microbenchmarks
- The 0.62x speed ratio is because V3Cache decompresses all layers through Python
  on every decoding step — a Triton kernel would eliminate this

### What's not in the paper that we discovered

1. **QJL is poison for attention**: Mathematically unbiased inner products ≠ good
   attention scores. Softmax exponentially amplifies variance. MSE-only with biased
   inner products produces better downstream generation than unbiased QJL.

2. **Keys need more bits than values**: Attention scores are `softmax(QK^T)` — key
   errors are exponentiated. Values are just a weighted sum where errors average out.
   K=8/V=4 dramatically outperforms uniform 6-bit.

3. **You need an fp16 window**: The most recent 128-256 tokens must stay
   uncompressed. Without this, generation quality degrades even when attention metrics
   look perfect.

4. **Protect the edges**: First and last transformer layers are fragile (positional
   encodings, final predictions). Giving them 8-bit while compressing middle layers
   to 2-4 bit is free quality.

5. **Attention cosine ≠ generation quality**: 99.5% attention cosine similarity can
   coexist with completely broken text generation. The only reliable test is actual
   `model.generate()` with factual verification.

---

## Learnings

### On implementing papers

1. **The paper is the theory, not the product.** TurboQuant's Algorithm 1 is
   mathematically beautiful and provably near-optimal. But the paper's Algorithm 2
   (QJL) actively hurts the primary use case (KV cache for generation). The gap
   between "optimal in the metric we defined" and "works in practice" is where
   engineering lives.

2. **Read the community first.** The tonbistudio reference repo and its 8+ community
   forks had already discovered every pitfall I hit. The QJL problem, the K/V
   asymmetry, the residual window — all documented in GitHub issues before I wrote
   a single line. I should have read more carefully before implementing.

3. **Theoretical bounds are shockingly tight.** The measured MSE at every bit-width
   matches the paper's predictions within a factor of 0.5-0.9x of the upper bound.
   This isn't an accident — the Lloyd-Max quantizer really is optimal for the
   distribution induced by random rotation. The math works.

### On GPU engineering

4. **cuBLAS is a cheat code.** A single cuBLAS matrix multiply at d=128 is faster
   than 7 sequential O(d)-work kernel launches. When your "optimization" adds kernel
   launch overhead, you lose. Asymptotic complexity is a lie on GPUs.

5. **Python is the wrong layer for inference.** Every token generated runs through
   Python's V3Cache.update(), which decompresses 36 layers of packed uint8 into fp16.
   The 0.62x speed ratio is entirely Python overhead. A fused Triton kernel that
   reads packed indices and writes attention output directly would match or exceed
   FP16 speed.

6. **Profile the boring code.** The codebook solver (scipy's `integrate.quad`) was
   the bottleneck, not the rotation or quantization. Caching solved it instantly.
   Always profile before assuming you know what's slow.

### On AI-assisted development

7. **The agent is a fast, overconfident junior engineer.** Claude wrote correct
   TurboQuant math on the first try but built a V3Cache that faked memory savings.
   The most valuable thing I did was look at the output and say "these numbers don't
   make sense." AI agents are excellent at implementing known algorithms and terrible
   at knowing when their implementation is dishonest.

8. **Iterate in small, verifiable steps.** The most productive sessions had a tight
   loop: implement one thing → run → examine output → correct → repeat. The least
   productive ones tried to build everything at once. When Claude generated 400 lines
   of evaluation code, the bug was always in the 3 lines I didn't review.

9. **The human's job is taste and skepticism.** I didn't write most of the code, but
   I made the decisions: drop QJL, switch to honest evaluation, use hybrid
   rotation. The agent implemented each decision flawlessly. Knowing *what* to build
   is still harder than building it.

### On honest benchmarking

10. **If your compression shows no memory savings, it's not compression.** The
    original V3Cache stored compressed uint8 bytes *and* decompressed fp16 tensors.
    Memory went up, not down. This is embarrassingly common in KV cache papers that
    report "theoretical" compression ratios.

11. **The only metric that matters is generation quality.** Cosine similarity, MSE,
    attention score overlap — all useful diagnostics, none sufficient. If the model
    can't find "AURORA-7749" in a document, the compression is broken, no matter
    what the cosine says.

---

*Built with PyTorch on a single RTX 4070 laptop GPU, with Claude as a tireless
pair programmer who occasionally needs to be told "no, actually look at the numbers."*

*The full implementation, evaluation harness, and a 26-item deviation log comparing
our code against the paper are at
[github.com/VjayRam/Research-Demos/turbo-quant](https://github.com/VjayRam/Research-Demos/tree/main/turbo-quant).*
