# RDKV Specification

**Paper:** RDKV: Rate-Distortion Bit Allocation for Joint Eviction and Quantization of the KV Cache
**arXiv:** 2605.08317
**Authors:** Junkai Zhang, Hang Guo, Luca Benini, Yawei Li

This spec transcribes the paper's math and algorithms exactly, as already
verified against the paper's LaTeX/HTML source in
[`rdkv-primer.html`](../../../rdkv/rdkv-primer.html). Every equation and algorithm below
is reproduced from the paper, not paraphrased. Where the primer explicitly
disclosed a substitution (the empirical distortion table), that is called out
here too — this spec does **not** repeat that substitution; it specifies the
real object the paper defines, and marks it as an open implementation
question (see §8).

---

## 1. Setup & notation

A decoder-only transformer has `L` layers, `H_q` query heads, `H_kv` KV heads
(group size `g = H_q / H_kv`), and per-head dimension `d`. For a prefilled
context of `T` tokens, the KV cache per layer is `K, V ∈ ℝ^(H_kv × T × d)`,
with:

```
a_{τ,t} = softmax_t( q_τᵀ k_t / √d )
o_τ     = Σ_t a_{τ,t} v_t
```

Every cache unit `u` — a token in the V cache, or a channel in the K cache —
gets assigned a bit-width `b_u ∈ 𝔹 = {0, 2, 4, 8, 16}`, from outright removal
to full FP16 retention, subject to a total bit budget `B`. The goal: find
`{b_u*}` minimizing the distortion this induces in the attention computation.

**Core reframing:** eviction and quantization are not two mechanisms.
`b_u = 0` is eviction; `b_u = 16` is full-precision retention; everything in
between is quantization. One optimization problem, one budget, one
allocation curve.

---

## 2. Token weight — V cache (§3.1, Proposition 3.1)

Each token enters the output as one weighted term `a_{τ,t} v_t`. Evicting
token `t` zeroes its logit; softmax renormalizes the rest:

```
Eq. (1):  â_{τ,t} = 0,   â_{τ,t'} = a_{τ,t'} / (1 − a_{τ,t})   for t' ≠ t
```

**Proposition 3.1.** Measuring the change with total-variation distance
between `a_τ` and `â_τ`:

```
‖a_τ − â_τ‖_TV = a_{τ,t}    for each query τ.

Eq. (2):  w_t := Σ_τ a_{τ,t}
```

`w_t` is cumulative attention (the same score H2O/SnapKV use for
keep/evict), but here it is a *multiplicative coefficient* in the objective,
not just a ranking score. Quantization noise `δv_t` propagates to the output
as `δo_τ = a_{τ,t} · δv_t`, scaling with the same `w_t` — this is what unifies
eviction and quantization under one weight.

---

## 3. Channel weight — K cache (§3.1, Proposition 3.2)

The K cache's compression unit is the *channel* (persistent per-channel
outliers make channels, not tokens, the natural thing to prune). Channel `c`
holds `{k_{t,c}}_{t=1}^T`. Since every logit is an inner product
`q_τᵀ k_t / √d`, zeroing channel `c` removes that coordinate's contribution
from every logit simultaneously:

```
Eq. (3):  δZ = −(1/√d) · Q[:,c] · K[:,c]ᵀ
```

This is a rank-one perturbation; its spectral norm gives the per-channel
weight.

**Proposition 3.2.**

```
Eq. (4):  w_c := (1/√d) · ‖Q[:,c]‖_2 · ‖K[:,c]‖_2
```

(This weight coincides with what ThinK derives independently via Frobenius
minimization — noted explicitly in the paper.) Quantizing rather than
zeroing a channel produces a logit perturbation scaling with the same `w_c`.

---

## 4. Quantization hardness — Bennett's approximation (§3.1)

`w_t` / `w_c` answer "how much does this unit matter." The other half is
"how hard is this unit to compress" — its quantization hardness. Under
Bennett's high-rate approximation, a uniform scalar quantizer with per-unit
dynamic range `R_u` achieves per-coordinate RMS error `σ_u · 2^(−b)` with:

```
σ_u := R_u / (2√3)
```

**Eq. (5) — the rate-distortion objective:**

```
ΔD({b_u}) = Σ_u w_u σ_u 2^(−b_u),    s.t. Σ_u b_u ≤ B
```

The coefficient actually ranked is `w_u σ_u`, not `w_u` alone: a
high-attention token that's easy to quantize (small dynamic range) can
rationally get fewer bits than a lower-attention token that's hard to
quantize.

---

## 5. Optimal bit allocation — reverse water-filling (§3.1, Theorem 3.3)

Minimizing Eq. (5) subject to `b_u ≥ 0` has an exact closed-form minimizer.

**Theorem 3.3.**

```
b_u* = [ log2( ln2 · w_u σ_u / λ ) ]_+
```

with `λ > 0` chosen so the budget binds. The water level `λ / ln2` induces a
hard phase transition: units with `w_u σ_u < λ / ln2` get `b_u* = 0`
(eviction) — eviction falls straight out of the same formula that assigns
quantization bit-widths to everything else. Tightening `B` raises `λ` and
pushes more units into eviction; loosening it promotes evicted units back to
low-bit retention.

There is no separate "eviction module" and "quantization module" — one
threshold, sliding with the budget, produces both.

---

## 6. Discrete allocation — MCKP (§3.1 Eq. 6, Appendix B/C)

Theorem 3.3 gives a real-valued `b_u*`, but GPU kernels need one of a fixed
set of bit-widths `𝔹 = {0, 2, 4, 8, 16}`. The paper replaces Bennett's smooth
curve with an **empirically calibrated per-coordinate distortion table**
`ε_u(b)`, and solves the resulting multiple-choice knapsack problem (MCKP).

**Eq. (6) — discrete MCKP:**

```
{b_u*} = argmin_{b_u ∈ 𝔹} Σ_u w_u ε_u(b_u)    s.t. Σ_u b_u ≤ B
```

Lagrangian relaxation decouples this into independent per-unit table
lookups. One-dimensional bisection on `λ` recovers a feasible allocation in
`O(U |𝔹|)` time.

### Empirical distortion table (Appendix B)

`ε_u(b)` is estimated on calibration sequences drawn from the prefill
prefixes of the LongBench tasks: **32 sequences, truncated to 4k tokens
each.** The calibration covers bit-widths `{0, 2, 4, 8, 16}` at both
per-token (V) and per-channel (K) granularity. This is the real object
Eq. (6) requires; how it is estimated in this implementation is an open
question tracked in §8, not resolved by this spec.

### Default hyperparameters (Appendix B)

| Hyperparameter | Value | Meaning |
|---|---|---|
| `S_w` (observation window) | 32 | Query window used to build the attention matrix `A` for weight computation |
| `w` (pooling kernel) | 5 | AvgPool1d kernel width applied to `w_t` |
| `r_K` (K/V budget split) | 1/2 | Fraction of `B_head` allocated to K; ablated and found best across budgets |
| `δ` (bisection tolerance) | 10⁻² | Relative tolerance on mean bit-width for Algorithm 2 convergence |
| `I` (max bisection iterations) | 64 | Iteration cap for Algorithm 2 |

---

## 7. Three-stage pipeline (§3.2, Fig. 2)

Run once per layer-head pair, immediately after prefill.

Under an FP16 reference, a per-head budget of `B_tok` tokens corresponds to
`B_head = 2 · B_tok · d · 16` total bits, split equally between V and K:
`B_V = B_K = ½ B_head`.

- **Stage 1 — weighting.** From the prefill forward pass, compute `w_t`
  (Prop. 3.1) from the attention matrix and `w_c` (Prop. 3.2) from Q/K
  column norms. Both come from the uncompressed cache, computed once — no
  value vectors or output residuals needed.
- **Stage 2 — V token allocation.** Each token holds `d` scalars, so `B_V`
  becomes `B̄_V := B_V / d` in summed-bit-width units. MCKP (Eq. 6) on `w_t`
  determines `𝒯_kept := {t : b_t^V > 0}`. Evicted tokens are gone entirely.
- **Stage 3 — K channel allocation, on kept tokens only.** `B_K` is spent
  only across `𝒯_kept` — each channel now holds `|𝒯_kept|` scalars, so
  `B̄_K := B_K / |𝒯_kept|`. MCKP runs again on `w_c` (reused unchanged from
  Stage 1 — no second forward pass).

  **Ordering constraint: V must be allocated before K.** K's shrunken
  denominator must reflect who actually survived eviction; reversing the
  order wastes K budget on soon-to-be-evicted tokens.

---

## 8. TriZone packed decode (§3.3, Fig. 3, Eq. 7)

A mixed-bit allocation only helps if the cache stays packed in HBM. If
quantized entries were unpacked to FP16 before the attention kernel reads
them, peak memory wouldn't move and dequantization would add pure latency.
Dequantization must fuse *into* the attention kernel while the cache stays
packed. GPU kernels want uniform-precision segments, so RDKV defines three
storage zones per `(ℓ, h)` pair:

- **Zone A — packed, quantized (old cache).** Retained K rows from
  `𝒯_kept`, plus V rows with `b_t^V ∈ {2, 4, 8}`, grouped into uniform-bit
  sub-segments.
- **Zone B — FP16, retained.** V rows with `b_t^V = 16`. Their K rows still
  live in Zone A, since K bit-widths follow the independent per-channel
  allocation, not `b^V`.
- **Zone C — FP16, new decode tokens.** Newly generated tokens, growing by
  one entry per decode step.

At each decode step the query attends over both zones at once (quantized
old K concatenated with FP16 new K), and the output splits into three value
sources matching the three zones.

**Eq. (7) — packed-decode output decomposition:**

```
o_τ = Σ_{t ∈ 𝒯_kept \ 𝒯_V16} a_{τ,t} · v̂_t     (Zone A, quantized V)
    + Σ_{t ∈ 𝒯_V16}          a_{τ,t} · v_t      (Zone B, FP16 retained)
    + Σ_{t ∈ 𝒯_new}          a_{τ,t} · v_t      (Zone C, FP16 new)
```

K-cache dequantization is fused algebraically rather than materialized:
per-channel quantization stores `k̂_{t,c} = s_c (k̃_{t,c} − z_c)`, so the
kernel rewrites `q_τᵀ k̂_t` as a sum over `(s_c q_{τ,c}) · k̃_{t,c}` minus one
per-query-head bias subtracted once — a dequantized FP16 tile is never
materialized and written back to memory.

**Open question tracked here (not part of the paper-verbatim math above):**
this project's `ε_u(b)` calibration methodology — the primer's interactive
demos substitute the analytic Bennett curve (`σ_u · 2^(−b)`) as a disclosed
stand-in, since the paper's real table requires offline calibration on
LongBench prefill activations (Appendix B: 32 sequences × 4k tokens). The
implementation plan must decide whether to (a) reproduce that calibration
procedure against a real model, or (b) use the Bennett-curve stand-in as a
first cut, explicitly flagged as an approximation, not silently presented as
the paper's table.

---

## 9. Algorithm 1 — RDKV: Rate-Distortion KV Cache Compression (Appendix C)

```
Input:  prefill cache K^(ℓ), V^(ℓ) per layer ℓ; per-head budget B_head;
        observation window S_w; pooling kernel w; bit-width set 𝔹={0,2,4,8,16};
        empirical distortion tables ε^K(b), ε^V(b)
Output: TriZone packed cache for each (ℓ,h)

for each layer ℓ = 1,…,L do
  # Stage 1: weight computation
  A ← Softmax( Q[τ−S_w:τ]^(ℓ) K^(ℓ)ᵀ / √d )
  for each KV head h = 1,…,H_kv do
    w_t^(h) ← Σ_{τ,g} a_{τ,g,t}^(h) for all t        ▸ V-cache token weight
    w_t^(h) ← AvgPool1d(w_t^(h), w)
    w_c^(h) ← (1/√d)·‖Q_{:,c}^(h)‖_2·‖K_{:,c}^(h)‖_2 for all c   ▸ K-cache channel weight
  end for
  # Stage 2: V-side token allocation (per head)
  B^V ← ½B_head;  B̄^V ← B^V/d
  for each KV head h do
    {b_t^V}_h ← MCKP( w_t^(h), ε^V, B̄^V/T )
    𝒯_kept^(h) ← {t : b_t^V > 0}
  end for
  # Stage 3: K-side channel allocation (per head)
  B^K ← ½B_head
  for each KV head h do
    k_avg^(h) ← B^K / ( |𝒯_kept^(h)| · d )
    {b_c^K}_h ← MCKP( w_c^(h), ε^K, k_avg^(h) )
  end for
  # Stage 4: TriZone packing
  for each KV head h do
    sort 𝒯_kept^(h) by b_t^V into sub-segments 𝒮_2, 𝒮_4, 𝒮_8
    sort channels by b_c^K into segments; permute q to match
    quantize + byte-pack V sub-segments → Zone A(V); K rows of 𝒯_kept^(h) → Zone A(K)
    store {t : b_t^V = 16} V rows in FP16 → Zone B
  end for
end for
```

---

## 10. Algorithm 2 — MCKP: Lagrangian Bisection Knapsack Solver (Appendix C)

```
Input:  weights {w_u}; distortion table ε(b); target avg. bits b̄;
        bit-width set 𝔹; tolerance δ; max iterations I
Output: bit-width assignment {b_u*}

λ_lo ← 0;  λ_hi ← max_u w_u
for i = 1,…,I do
  λ ← (λ_lo + λ_hi) / 2
  for each unit u do
    b_u* ← argmin_{b ∈ 𝔹}  w_u·ε(b) + λ·b
  end for
  b̄_cur ← mean({b_u*})
  if |b̄_cur − b̄| / b̄ < δ then return {b_u*}
  else if b̄_cur > b̄ then λ_lo ← λ
  else λ_hi ← λ
end for
return {b_u*}
```

**Implementation note (not in the paper) — convergence behavior at `b̄ = 0`:**
when the target average bit-width is
exactly zero (e.g. Stage 3 called with `𝒯_kept = ∅`), the relative-tolerance
check `|b̄_cur − b̄| / b̄` is undefined (division by zero). The implementation
must special-case this: converge as soon as `b̄_cur ≤ ε_abs` for a small
absolute tolerance, returning all-zero bit-widths.

---

## 11. Worked example (hand-verifiable, constructed for this spec's test suite)

One query `τ` attending to `T = 4` tokens with softmax weights
`a_τ = [0.5, 0.3, 0.15, 0.05]` (single query ⇒ Eq. 2 collapses to
`w_t = a_{τ,t}` directly).

**Proposition 3.1 check — evict token 1 (a=0.5):**

Eq. (1): `â = [0, 0.6, 0.3, 0.1]` (renormalized, sums to 1).

```
‖a − â‖_TV = ½(|0.5−0| + |0.3−0.6| + |0.15−0.3| + |0.05−0.1|)
           = ½(0.5 + 0.3 + 0.15 + 0.05) = ½ · 1.0 = 0.5
```

Matches `a_{τ,1} = 0.5` exactly.

**Channel weight (Proposition 3.2), `d=2`, `T=4`, `√d = √2 ≈ 1.41421`:**

```
Q[:,1] = [1, 0, 1, 0.5]     K[:,1] = [0.8, 0.6, 0.4, 0.2]
Q[:,2] = [0.3, 0.9, 0.2, 0.1]     K[:,2] = [0.5, 0.5, 0.5, 0.5]

‖Q[:,1]‖ = √2.25 = 1.5,      ‖K[:,1]‖ = √1.2 ≈ 1.0954
  ⇒ w_c1 = 1.5 · 1.0954 / 1.41421 ≈ 1.1619

‖Q[:,2]‖ = √0.95 ≈ 0.9747,   ‖K[:,2]‖ = √1.0 = 1.0
  ⇒ w_c2 = 0.9747 · 1.0 / 1.41421 ≈ 0.6893
```

**Theorem 3.3 applied directly, fixing `λ/ln2 = 0.1`, `σ_u = 1`:**

`b_u* = [log2(w_u σ_u / 0.1)]_+`

| unit | weight | `w_u/0.1` | `b_u*` (continuous) | regime |
|---|---|---|---|---|
| token 1 | 0.500 | 5.00 | log2(5) ≈ 2.322 | quantize |
| token 2 | 0.300 | 3.00 | log2(3) ≈ 1.585 | quantize |
| token 3 | 0.150 | 1.50 | log2(1.5) ≈ 0.585 | quantize (thin) |
| token 4 | 0.050 | 0.50 | log2(0.5) = −1 → 0 | **evicted** |
| channel 1 | 1.1619 | 11.62 | log2(11.62) ≈ 3.539 | quantize |
| channel 2 | 0.6893 | 6.89 | log2(6.89) ≈ 2.785 | quantize |

Token 4 falls below the water level and gets clipped to zero. Both channels
stay above the line at this particular `λ` (no channel eviction in this
construction).

This example is a good source of a first unit test for the continuous
water-filling formula (Theorem 3.3), independent of the MCKP/discretization
layer.

---

## 12. Headline results (§4, for validating an eventual end-to-end reproduction)

From the paper's abstract and §4.1 (Table 1), LLaMA-3.1-8B-Instruct /
LongBench:

| Metric | Value |
|---|---|
| Cache retention | 2.48% |
| Accuracy retained vs. full-cache | 97.81% |
| Decode speedup vs. FlashAttention-2, 128K ctx | 4.5× |
| Peak memory reduction, 128K ctx | 1.9× |
| Avg. improvement over best baseline (LongBench/RULER/InfiniteBench)¹ | +9.1% |

At `B_total = 1024L`, RDKV reaches 49.47 average LongBench score, within
0.35 points of uncompressed FullKV (49.82). At the tightest budget
`B_total = 64L`, its lead over the strongest binary-eviction baseline widens
to 2.7 points. Mechanism claimed by the paper: binary baselines discard
everything below a top-k threshold, while RDKV's larger action space lets
borderline tokens keep a low bit-width instead of being dropped outright.

These numbers require a full model + LongBench harness to reproduce and are
**not** a near-term implementation target; they're recorded here as the
ground truth to eventually validate against.

¹ Sourced from `rdkv-primer.html` only; not independently re-confirmed
against the paper's abstract during this spec's authoring (a fresh fetch of
the abstract text did not surface this figure). Treat as needing
confirmation against the paper's full §4 before being relied on.

---

## 13. Scope boundary for the first implementation pass

This spec covers the full paper mechanism. A first implementation pass
should prioritize, in order:

1. §5 (Theorem 3.3, continuous water-filling) — smallest closed-form unit,
   independently testable against §11's worked example.
2. §6/§10 (Algorithm 2, MCKP bisection) — testable against the same worked
   example once discretized to `𝔹 = {0,2,4,8,16}`.
3. §2–§4 (weight computation, `w_t`, `w_c`, Bennett `σ_u`) — requires real
   attention matrices (a small transformer or synthetic Q/K/V), still no
   GPU kernel work.
4. §7 (three-stage pipeline) — orchestrates 1–3 above.
5. §8/§9 (TriZone packing, fused dequantization, Algorithm 1 end-to-end) —
   requires a real attention kernel; substantially larger scope, likely a
   separate plan.

The empirical `ε_u(b)` calibration (§6, §8's open question) is a
prerequisite for stages 2+ of the pipeline and must be resolved before
writing the implementation plan — this spec deliberately leaves it open
rather than silently substituting the Bennett curve, since that
substitution was explicitly disclosed as a *primer-only* concession, not
something to carry into the actual implementation.

---

## 14. Implementation decisions (this project, not the paper)

Resolved during brainstorming on 2026-08-31, closing the open questions
§8 and §13 raised:

1. **Phasing.** Phase 1 = §5 (Theorem 3.3) + §6/§10 (Algorithm 2/MCKP) +
   §2–§4 (weight computation) + §7 (3-stage pipeline), all pure
   PyTorch/numpy, no custom GPU kernel. Phase 2 (separate future plan) =
   §8/§9 (TriZone packing, fused dequantization attention kernel) —
   mirrors how `turbo-quant/` split its Triton kernel work out from the
   core algorithm.
2. **`ε_u(b)` calibration (§6/§8's open question).** Phase 1 uses the
   Bennett-curve stand-in `ε_u(b) = σ_u · 2^(−b)` from §4/Theorem 3.3,
   explicitly labeled in code and docs as an approximation, not the
   paper's real offline-calibrated table (Appendix B: 32 LongBench
   sequences × 4k tokens per (layer, head) slice). Real calibration is
   deferred to a later task, only after Phase 1's math is validated
   against §11's worked example.
3. **Test fixtures.** Unit tests and math verification use synthetic
   Q/K/V (matching §11's worked example and randomly generated tensors
   shaped `(H_kv, T, d)`) — dependency-free, no model download. Benchmarks
   and end-to-end runs wire in a real small HF model, following
   `turbo-quant/examples/`'s pattern.
4. **Package layout.** Follows `turbo-quant/`'s convention: flat package
   (`rdkv/rdkv/`) with focused modules, a mirrored `tests/` directory, and
   a standard `pyproject.toml` with a `test` extra (a `kernel` extra is
   deferred to Phase 2).

---

## Sources

All equations, propositions, theorem, and algorithms above are transcribed
from `arXiv:2605.08317`'s published source (cross-checked against both the
paper's own HTML rendering and this repository's `rdkv-primer.html`, which
was itself built directly from the paper's LaTeX/HTML source). Appendix B/C
hyperparameter defaults and calibration procedure description were fetched
directly from the paper's arXiv HTML rendering during this spec's authoring.
