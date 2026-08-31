# RDKV

Paper-accurate implementation of [RDKV: Rate-Distortion Bit Allocation for
Joint Eviction and Quantization of the KV Cache](https://arxiv.org/abs/2605.08317)
(arXiv:2605.08317).

See [`rdkv-primer.html`](rdkv-primer.html) for the interactive derivation
walkthrough, and
[`../docs/superpowers/specs/2026-08-31-rdkv-design.md`](../docs/superpowers/specs/2026-08-31-rdkv-design.md)
for the full spec this implementation follows.

**Phase 1 (this code):** continuous water-filling (Theorem 3.3), discrete
MCKP bit allocation (Algorithm 2), per-unit weight computation
(Propositions 3.1/3.2), and the three-stage allocation pipeline (Algorithm 1
Stages 1-3). Pure PyTorch, no custom GPU kernel.

**Not yet implemented (Phase 2):** TriZone packing and the fused
dequantization attention kernel (Algorithm 1 Stage 4).

**Disclosed approximation:** the paper's empirically-calibrated
per-coordinate distortion table `ε_u(b)` (Appendix B) is stood in for by the
analytic Bennett curve `σ_u · 2^(−b)` throughout this phase — see
`rdkv/mckp.py`.

## Install

```bash
pip install -e ".[test]"
```

## Test

```bash
pytest tests/
```
