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

**Disclosed gap (Zone A(V)):** Zone A's K rows are really quantized
(per-channel affine, packed as integers). Zone A's V rows are only
*grouped* by their target bit-width (2/4/8) -- they are still stored at
full float32 precision, not actually quantized or byte-packed. The
compression this implies for V is not yet realized; real V
quantization/byte-packing is follow-up work. See `rdkv/trizone.py`'s
`PackedCache.zone_a_v` field comment for the same disclosure in code.

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

## Benchmarks

Two scripts under `examples/` (see their `--help` for full options):

```bash
python examples/run_allocation_sweep.py          # allocation tradeoff vs. b_tok
python examples/run_perf_benchmark.py            # native vs. kernel decode latency
```

**Allocation sweep** (`sshleifer/tiny-gpt2`, layer 0, `T=81`, `d=1` --
this model's embedding is only 2-dim split across 2 heads, so `mean_b_k`
doesn't move; try a larger model to see the K-channel axis vary too):

| b_tok | kept | kept % | mean b_v | mean b_k | compress (illustrative) |
|---|---|---|---|---|---|
| 0.25 | 2/81 | 2.5% | 0.05 | 2.00 | 324.00x |
| 0.5 | 4/81 | 4.9% | 0.10 | 2.00 | 162.00x |
| 1.0 | 8/81 | 9.9% | 0.20 | 2.00 | 81.00x |
| 2.0 | 16/81 | 19.8% | 0.40 | 2.00 | 40.50x |
| 4.0 | 30/81 | 37.0% | 0.79 | 2.00 | 20.90x |
| 8.0 | 51/81 | 63.0% | 1.58 | 2.00 | 11.27x |
| 16.0 | 72/81 | 88.9% | 3.14 | 2.00 | 6.51x |

"compress" is illustrative, not measured -- it assumes Zone A(V)/Zone A(K)
are bit-packed at their target widths, which isn't implemented yet (see
the disclosed gap above).

**Decode latency, native vs. fused-kernel backend** (RTX 4070 Laptop GPU,
`torch==2.11.0+cu130`, `triton==3.8.0`, synthetic packed cache,
`--repeats 50`, mean of `--repeats` timed calls after one warm-up call):

| d | n_kept | native (cpu) | native (cuda) | kernel (cuda) | kernel speedup |
|---|---|---|---|---|---|
| 64 | 50 | 0.23ms | 0.85ms | 0.68ms | 1.25x |
| 64 | 200 | 0.38ms | 1.19ms | 0.73ms | 1.64x |
| 64 | 1,000 | 0.47ms | 0.86ms | 0.65ms | 1.32x |
| 64 | 5,000 | 0.88ms | 0.91ms | 0.80ms | 1.13x |
| 64 | 20,000 | 2.81ms | 0.85ms | 1.24ms | **0.68x (regression)** |
| 128 | 50 | 0.25ms | 0.88ms | 0.71ms | 1.25x |
| 128 | 200 | 0.45ms | 0.84ms | 0.79ms | 1.06x |
| 128 | 1,000 | 0.60ms | 1.00ms | 0.72ms | 1.38x |
| 128 | 5,000 | 1.32ms | 0.77ms | 0.72ms | 1.08x |
| 128 | 20,000 | 5.49ms | 1.21ms | 0.81ms | 1.49x |

The kernel backend wins in every configuration except `d=64, n_kept=20000`,
which reproduces as a real regression (not measurement noise -- confirmed
at both `--repeats 20` and `--repeats 50`), not just a one-off. Only
`_fused_zone_a_scores` (the K-dequantization fusion itself) runs as a
single Triton kernel; the surrounding `fused_packed_decode` glue
(per-bit-width `searchsorted` gathers, `torch.cat` for Zone A(V)/B,
softmax+matmul) is still several small PyTorch/CUDA calls whose launch
overhead doesn't shrink with `d`, while the fusion's savings scale with
`d`. At `d=64` (little per-token compute to fuse away) and large `n_kept`
(overhead accumulates across the zone-splitting), that overhead can
outweigh the fusion win. This is a code-level explanation, not a profiled
one (no `torch.profiler`/CUDA-event breakdown per sub-call yet) -- treat it
as the leading hypothesis, not a confirmed root cause, and re-profile
before relying on the kernel backend at small `d` with very large caches.
