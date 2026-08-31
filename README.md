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

See [`seq-attention/README.md`](seq-attention/README.md) for results and
how to reproduce them.
