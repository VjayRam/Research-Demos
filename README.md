## Research Implementations

This repository contains from-scratch implementations of research papers with evaluation and benchmarking.

## Papers

### TurboQuant -- KV Cache Compression ([`turbo-quant/`](turbo-quant/))

**Paper**: [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) (ICLR 2026, Zandieh et al.)

Paper-accurate PyTorch implementation of Algorithm 1 (`TurboQuantMSE`), Algorithm 2 (`TurboQuantProd`), and PolarQuant. Haar-random rotation via QR, Lloyd-Max against the exact Beta / sin-power densities, no Hadamard or Gaussian shortcuts.

**Results** (Qwen2.5-0.5B, WikiText-2, CUDA, `head_dim=64`). Compression is analytical (index bits vs fp16). Perplexity is a full-cache round-trip with no residual fp16 window.

| Setting | Compression | Perplexity | Notes |
|---------|-------------|------------|-------|
| fp32 baseline | 1.00× | 10.13 | — |
| PolarQuant 4-bit | **3.76×** | **70.4** | Best compressed setting |
| TurboQuantMSE 4-bit | 3.76× | 156.1 | Next-best reconstruction |
| TurboQuantProd 4-bit | 3.76× | 799.6 | QJL hurts softmax attention |
| TurboQuantMSE 1-bit | 12.8× | 4135 | Distortion on real keys still matches the solved codebook (0.365 vs 0.358) |

Real-key MSE distortion at 4-bit is **0.00913**, matching `d · C(f, b)` from Theorem 1. CUDA quantize throughput for MSE 4-bit (`d=64`) is **7.4M vec/s** (~13× CPU).

See [`turbo-quant/README.md`](turbo-quant/README.md) for the algorithm, distortion table, throughput, and how to reproduce.

### Sequential Attention -- Feature Selection ([`seq-attention/`](seq-attention/))

**Paper**: [Sequential Attention for Feature Selection](https://arxiv.org/abs/2209.14881) (ICLR 2023, Yasuda et al.)

Paper-accurate PyTorch implementation of Algorithm 1 (greedy sequential
selection, naive and one-pass variants), the softmax attention mask over
per-feature attention logits, and a numerical demonstration of the
paper's proven OMP/Sequential-LASSO equivalence. Benchmarked against the
paper's Table 2 on MNIST, Fashion-MNIST, and ISOLET.

| Dataset | Baseline (all features) | Sequential Attention (k=50) | Paper (Table 2) |
|---|---|---|---|
| MNIST | 0.9782 | 0.9409 | 0.944 → 0.956 |
| Fashion-MNIST | 0.8876 | 0.8602 | 0.843 → 0.854 |
| ISOLET | 0.9532 | 0.9089 | 0.866 → 0.920 |

Run on the project's RTX 4070. Selected-feature accuracy trails the
full-feature baseline on all three datasets here (unlike the paper),
attributed to this project's untuned single-run MLP rather than a
selection bug — see [`seq-attention/README.md`](seq-attention/README.md)
for the full diagnosis.

See [`seq-attention/README.md`](seq-attention/README.md) for the ISOLET
result's diagnosis and how to reproduce these numbers.

### RDKV -- Joint Eviction and Quantization of the KV Cache ([`rdkv/`](rdkv/))

**Paper**: [Rate-Distortion Bit Allocation for Joint Eviction and Quantization of the KV Cache](https://arxiv.org/abs/2605.08317) (arXiv:2605.08317)

RDKV treats KV cache eviction and quantization as the same operation — bit-width assignment — evaluated at different depths (0 bits = evicted). Implemented end to end: closed-form continuous water-filling (Theorem 3.3), discrete MCKP bit allocation via Lagrangian bisection (Algorithm 2), per-unit weight computation (Propositions 3.1/3.2), the three-stage allocation pipeline (Algorithm 1 Stages 1-3), TriZone packed storage (Algorithm 1 Stage 4), and a fused-dequantization decode kernel (Eq. 7) that never materializes a dequantized FP16 K tile.

**Disclosed approximation**: the empirically-calibrated per-coordinate distortion table from the paper's Appendix B is stood in for by the analytic Bennett curve `σ_u · 2^(−b)`; see [`rdkv/README.md`](rdkv/README.md).

See [`rdkv/rdkv-primer.html`](rdkv/rdkv-primer.html) for the math derivation, [`docs/superpowers/specs/2026-08-31-rdkv-design.md`](docs/superpowers/specs/2026-08-31-rdkv-design.md) for the full spec, and [`rdkv/README.md`](rdkv/README.md) for install/test instructions.
