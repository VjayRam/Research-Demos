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
`torch==2.11.0+cu130`, `triton==3.8.0`, synthetic packed cache, **on AC
power** -- see the power-state note below for why that matters. 8
independent runs per config, `--repeats 50` each; native (cuda)/kernel
(cuda) columns are mean +/- stdev across those 8 runs):

| d | n_kept | native (cuda) | kernel (cuda) | speedup |
|---|---|---|---|---|
| 64 | 50 | 0.95 +/- 0.16ms | 0.76 +/- 0.08ms | 1.25x |
| 64 | 200 | 0.88 +/- 0.04ms | 0.71 +/- 0.04ms | 1.24x |
| 64 | 1,000 | 0.91 +/- 0.05ms | 0.74 +/- 0.04ms | 1.24x |
| 64 | 5,000 | 0.88 +/- 0.03ms | 0.74 +/- 0.02ms | 1.19x |
| 64 | 20,000 | 0.88 +/- 0.02ms | 0.76 +/- 0.03ms | 1.16x |
| 128 | 50 | 0.91 +/- 0.07ms | 0.71 +/- 0.03ms | 1.28x |
| 128 | 200 | 0.95 +/- 0.14ms | 0.76 +/- 0.12ms | 1.24x |
| 128 | 1,000 | 0.89 +/- 0.03ms | 0.71 +/- 0.03ms | 1.26x |
| 128 | 5,000 | 0.84 +/- 0.03ms | 0.71 +/- 0.03ms | 1.18x |
| 128 | 20,000 | 1.25 +/- 1.03ms | 0.80 +/- 0.03ms | 1.57x |

The kernel backend wins in all 10 configurations, consistently, by
1.16x-1.57x. Note `d=128, n_kept=20000`'s native stdev (1.03ms on a
0.87-0.91ms typical value across other runs) -- one of its 8 runs hit a
single high-latency outlier; the kernel backend's stdev stayed tight
(0.03ms) across the same 8 runs at that config, i.e. on AC power the
*native* backend was the noisier one, not the kernel.

**Power-state note (root-caused via `superpowers:systematic-debugging`).**
An earlier version of this table, gathered while the laptop was running on
**battery power**, reported a reproducible-looking 0.68x regression at
`d=64, n_kept=20000` -- but that claim was wrong. Root-causing it: calling
the benchmark harness's own `benchmark_config()` directly for that exact
config showed the kernel *winning*; three consecutive full-script battery
runs each disagreed on which config (if any) "regressed"; GPU clocks were
confirmed stable (ruling out thermal/boost-ramp effects) via
`nvidia-smi` sampling during a run; and switching to pure on-device
`torch.cuda.Event` timing (removing host-side Python/OS scheduling from
the measurement) *still* showed the same config's full latency
distribution shift in one battery trial and none in the next -- a max
outlier of 8.2ms against a ~1ms typical value. `nvidia-smi` showed the GPU
pinned at `pstate=P3` and never exceeding 1980MHz (its boost max is
3105MHz) throughout every battery run. Re-running the identical sweep
twice on AC power (`pstate=P0`) produced consistently tighter, lower
latencies and the kernel winning nearly everywhere, confirmed again with
`torch.cuda.Event` timing on the previously-worst config. Root cause:
these decode calls are sub-2ms, and on battery power this laptop's GPU
runs in a throttled power state (P3, capped well below its boost clock)
that appears to transition/stall more unpredictably under Windows' WDDM
driver scheduling than the stable P0 state AC power allows -- at this
timescale, that's enough occasional tail latency to flip which backend
looks faster in a mean-of-50 comparison. This was a measurement artifact,
not a kernel or architecture defect; the original "confirmed regression"
claim in this README was incorrect and has been corrected above.
**If benchmarking this on a laptop, plug into AC power first.**
