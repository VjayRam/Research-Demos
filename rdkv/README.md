# RDKV

Paper-accurate implementation of [RDKV: Rate-Distortion Bit Allocation for
Joint Eviction and Quantization of the KV Cache](https://arxiv.org/abs/2605.08317)
(arXiv:2605.08317).

See [`rdkv-primer.html`](rdkv-primer.html) for the interactive derivation
walkthrough, and
[`../docs/superpowers/specs/2026-08-31-rdkv-design.md`](../docs/superpowers/specs/2026-08-31-rdkv-design.md)
for the full spec this implementation follows.

**Phase 1:** continuous water-filling (Theorem 3.3), discrete MCKP bit
allocation (Algorithm 2), per-unit weight computation (Propositions
3.1/3.2), and the three-stage allocation pipeline (Algorithm 1 Stages 1-3).
Pure PyTorch, no custom GPU kernel.

**Phase 2:** TriZone packing (Algorithm 1 Stage 4) and packed decode
(Eq. 7), with a native PyTorch reference implementation and a GPU-only
fused Triton kernel (`backend="kernel"`) that never materializes a
dequantized FP16 K tile for the packed zone. Requires the `kernel` extra
and CUDA; falls back to a clear `RuntimeError` (not a silent CPU fallback)
otherwise.

**Disclosed approximation (both phases):** `ε_u(b)` is the analytic
Bennett curve `σ_u · 2^(−b)`, standing in for the paper's
empirically-calibrated per-coordinate distortion table (Appendix B). This
affects which bit-widths Phase 1 chooses; Phase 2 packs and decodes
whatever bit-widths it's given and does not depend on how they were chosen.

## Install

```bash
pip install -e ".[test]"          # core + tests
pip install -e ".[examples]"      # + real-model example scripts
pip install -e ".[kernel]"        # + fused Triton decode kernel (CUDA only)
```
