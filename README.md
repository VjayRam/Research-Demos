## Research Implementations

This repository contains from-scratch implementations of research papers with evaluation and benchmarking.

## Papers

### TurboQuant -- KV Cache Compression ([`turbo-quant/`](turbo-quant/))

**Paper**: [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) (ICLR 2026, Zandieh et al.)

Compresses the KV cache of transformer models during inference using randomized Hadamard rotation + Lloyd-Max quantization with bit-packed storage. Implements the community-informed V3 variant with asymmetric K/V bits, residual windowing, and layer-adaptive precision.

**Results** (Qwen2.5-3B-Instruct, 2046 tokens, RTX 4070):

| Profile | Memory Ratio | Attention Cosine | Generation tok/s | Quality |
|---------|-------------|-----------------|-----------------|---------|
| FP16 baseline | 1.0x | -- | 3.1 | FOUND |
| moderate (K8/V4) | **2.2x** | 0.996 | 1.9 (0.62x) | FOUND |
| extreme (K4/V2) | **2.6x** | 0.967 | 1.9 (0.62x) | FOUND |

See [`turbo-quant/README.md`](turbo-quant/README.md) for algorithm details, pseudocode, and usage.

### Sequential Attention

**Paper**: [Sequential Attention: Making AI models leaner and faster without sacrificing accuracy](https://research.google/blog/sequential-attention-making-ai-models-leaner-and-faster-without-sacrificing-accuracy/)

*Implementation pending.*
