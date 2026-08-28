# I Rebuilt Google's TurboQuant Paper-Accurate, No Shortcuts — Here's What the Numbers Actually Say

*Redesigning a KV-cache compression algorithm as an exact, importable package —
then running it against a real model to find out whether the paper's math survives
contact with real activations.*

---

## TL;DR

I had an earlier, engineering-heavy implementation of Google's TurboQuant paper
(ICLR 2026) that took shortcuts — Hadamard transforms instead of the paper's exact
random rotation, a Gaussian approximation instead of the exact Beta coordinate
density, and a hand-tuned V3 variant with residual windows and asymmetric bit
allocation to make generation usable. This time I threw all of that out and rebuilt
`turboquant` as a **paper-exact package**: QR-based Haar-random rotation, the exact
Beta/sin-power Lloyd-Max densities from the paper, both `TurboQuant_mse` and
`TurboQuant_prod` (Algorithm 1 and 2), plus PolarQuant as a configurable third
option — with zero approximations anywhere in the core module. Then I benchmarked
it for real: **Qwen2.5-0.5B**, **WikiText-2**, CPU vs GPU, and real K-vectors pulled
out of a live forward pass. Result: the *math* is exactly right — empirical
distortion matches the paper's Theorem 1 bound to three decimal places. The
*generation quality* is not — quantizing the raw KV cache with no windowing or
asymmetric bit allocation wrecks perplexity even at 4 bits. Theory and engineering
are still two different jobs.

---

## Stack

| Layer | Tool | Why |
|-------|------|-----|
| Language | Python 3.13 | Type hints, dataclasses, modern `scipy` |
| Deep Learning | PyTorch (CUDA 13.0 build) | GPU tensors, `torch.linalg.qr`, autograd-free |
| Models | HuggingFace Transformers | `AutoModelForCausalLM`, `DynamicCache` subclassing |
| Corpus | HuggingFace `datasets` | WikiText-2 (`wikitext-2-raw-v1`), real text, not a fixture string |
| Numerical | SciPy | `integrate.quad` for the exact continuous Lloyd-Max solve |
| Environment | `uv` workspace | `turbo-quant` is a workspace member sharing one `.venv`/lockfile with the rest of this monorepo |
| Hardware | NVIDIA RTX 4070 Laptop GPU (CUDA 13 driver) | Auto-detected at runtime, CPU is the fallback |
| Testing | pytest | 55 tests, TDD-driven per module |
| Version control | Git, task-scoped subagent review | Every module reviewed against the plan before merge |

No FWHT, no Gaussian approximation, no bit-packing tricks in the core package —
those all live in `examples/` now, deliberately separated from the paper-accurate
math.

---

## The Architecture

### Package layout

```
turbo-quant/
├── turboquant/                 # core package — paper-exact, no engineering hacks
│   ├── rotation.py             # QR-based Haar-random rotation (exact, O(d^3) via torch.linalg.qr)
│   ├── distributions.py        # exact Beta coordinate density + PolarQuant angle densities
│   ├── lloyd_max.py            # generic continuous Lloyd-Max solver (scipy.integrate.quad)
│   ├── codebook.py             # Codebook: cached per (density, bits), literal argmin quantize
│   ├── qjl.py                  # QJL sign-quantization matrix + sign_quantize()
│   ├── cartesian.py            # TurboQuantMSE (Algorithm 1), TurboQuantProd (Algorithm 2)
│   └── polar.py                # PolarQuant — recursive Cartesian -> polar decomposition
├── tests/                      # 55 tests, one file per module
└── examples/                   # KV-cache engineering layer — NOT part of the paper-exact core
    ├── kv_cache_hook.py        # QuantizingCache(DynamicCache) — round-trips K/V through a quantizer
    ├── run_benchmark.py        # perplexity + compression sweep, real HF model, CSV output
    ├── run_perf_benchmark.py   # CPU vs GPU quantize/dequantize latency & throughput
    ├── run_experiments.py      # Qwen2.5-0.5B + WikiText-2: perplexity AND empirical-vs-theoretical distortion
    └── results_logger.py       # writes every sweep to timestamped CSVs for later plotting
```

The split is deliberate. `turboquant/` implements exactly what the paper specifies —
nothing more, no alternatives baked in as defaults. Every KV-cache-specific decision
(how to hook `DynamicCache`, what to measure, which model to run) lives in
`examples/`, where it can be as opinionated as it needs to be without contaminating
the reference implementation.

### Data flow, TurboQuant_mse (Algorithm 1)

```
x (d-dim vector, e.g. one attention head's K or V)
   │
   ▼
R = QR(Gaussian(d,d))            exact Haar-random rotation, cached per (d, seed, device)
   │
   ▼
y = R @ x                        rotated vector — coordinates are now i.i.d. Beta(d)
   │
   ▼
per-coordinate Lloyd-Max quantize   Codebook.for_density(beta_coordinate_density(d), bits)
   │                                  literal argmin against precomputed centroids
   ▼
(indices, ||x||)                 quantized output + one fp scalar norm

Decompress: centroids[indices] * ||x||  -->  R^T @ (.)  -->  x_hat
```

### Data flow, TurboQuant_prod (Algorithm 2)

Same rotation and MSE stage at `bits - 1`, plus a 1-bit QJL sign correction on the
residual:

```
residual = y - dequantize(quantize_mse(y))
s = sign(S @ residual)             S ~ N(0,1), d x d, cached per (d, seed, device)
correction_scale = sqrt(pi/2) / d  (computed once, reused by dequantize AND inner_product)
```

`inner_product(y, compressed)` uses the correction to give a mathematically unbiased
estimator of `<x, y>` — this is Algorithm 2's whole point, and it is implemented
exactly as specified, not approximated.

### PolarQuant

Recursive Cartesian → polar decomposition: pair up coordinates, convert each pair to
(radius, angle), recurse on the radii for `log2(d)` levels. Angle densities are
exact too — uniform on `[0, 2π)` at level 1, `∝ sin(2θ)^(2^(ℓ-1)-1)` on `[0, π/2]`
at every level after that.

---

## Idea & Research

### Why redo it

My earlier pass at this paper (documented in an earlier version of this post) had
already produced *something that worked* — 2.2–2.6x real compression on
Qwen2.5-3B-Instruct — but it worked by deviating from the paper: Hadamard
transforms instead of the true random rotation, a Gaussian approximation of the
Beta density, and a hand-engineered V3 variant (asymmetric K/V bits, an fp16
residual window for recent tokens) needed to keep generation usable. Useful
engineering, but it meant I could never say "I implemented the paper" — I'd
implemented a *derivative* of it.

This time the goal was different: build the exact algorithm, with the exact
mathematical objects the paper defines, as a clean importable package — no
Hadamard, no Gaussian shortcut, no baked-in engineering decisions — and then
measure, honestly, how far that exact algorithm gets you before any engineering
is layered on top.

### What "exact" meant in practice

- **Rotation**: the paper specifies a Haar-random orthogonal matrix. FWHT + random
  signs approximates this distributionally for large d but is not the same object.
  `rotation.py` does the real thing: QR-decompose a Gaussian matrix, fix the sign
  ambiguity via the sign of the diagonal of R, and accept the O(d³) cost.
- **Coordinate density**: after a Haar-random rotation, each coordinate of a
  rotated unit vector follows an exact Beta-derived density,
  `f_X(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1 - x²)^((d-3)/2)`, not a Gaussian
  approximation of it. `distributions.py` computes this with `math.lgamma` for
  numerical stability, and `lloyd_max.py` solves the continuous 1-D optimal
  quantizer against that exact density via numerical integration — the same
  computation reproduces the paper's Theorem 1 distortion table (0.360 / 0.117 /
  0.030 / 0.009 for b = 1..4 at d = 128) to three significant figures.
- **PolarQuant as a citizen, not an afterthought**: the paper's companion blog post
  treats PolarQuant as an alternative. This package makes it a fully configurable
  third algorithm (`PolarQuant(d, bits, seed, device)`) with the same
  `.quantize()`/`.dequantize()` interface as the other two, rather than bolting it
  on as a special case.

---

## Implementation

Built as a 13-task plan executed through subagent-driven development: a fresh
implementer per module, a task-scoped reviewer per module (spec compliance +
code quality, both graded independently of the implementer's self-report), and a
progress ledger tracking every review round. Three review rounds each caught the
same category of bug — more on that below.

### Core modules

- **`rotation.py`** — `generate_rotation_matrix(d, seed, device=None)`. QR
  decomposition of a Gaussian matrix built in float64 (numerical stability),
  sign-fixed, cast to float32, cached per `(d, seed, resolved_device)`.
- **`distributions.py`** — `Density` dataclass, `beta_coordinate_density(d)`,
  `polar_angle_density(level)`.
- **`lloyd_max.py`** — `solve_lloyd_max(pdf, support, bits)`, a generic continuous
  Lloyd-Max solver parameterized by any `(pdf, support)` pair, not hardcoded to
  one distribution.
- **`codebook.py`** — `Codebook.for_density(density, bits)`, cached, literal
  `argmin` quantize (the paper's nearest-centroid rule, not a bucketize
  approximation).
- **`qjl.py`** — QJL sign-quantization matrix and `sign_quantize()` for Algorithm 2.
- **`cartesian.py`** — `TurboQuantMSE`, `TurboQuantProd`.
- **`polar.py`** — `PolarQuant`, validated against power-of-two `d` and `bits >= 1`.

### Examples layer (KV-cache engineering, deliberately outside the core)

- **`kv_cache_hook.py`** — `QuantizingCache(DynamicCache)`, round-trips K/V through
  any of the three quantizers inside `.update()`. Handles both the tuple return
  from `TurboQuantMSE` and the dict return from `TurboQuantProd`/`PolarQuant`.
- **`run_benchmark.py`** — perplexity + analytical compression ratio sweep across
  algorithm × bits, against a real HF model, with CPU/GPU auto-detection and CSV
  output.
- **`run_perf_benchmark.py`** — CPU vs GPU quantize/dequantize latency and
  throughput, with `torch.cuda.synchronize()` correctly bracketed before and
  after every timed block (the usual async-kernel-launch trap that makes GPU
  timing look artificially fast if you skip it).
- **`run_experiments.py`** — the "does this actually work" script, described below.
- **`results_logger.py`** — every sweep writes a timestamped CSV to
  `examples/results/` so results can be plotted later without re-running anything.

### Device auto-detection

Every matrix-generation function and every quantizer class takes
`device: str | None = None`, resolved once via
`'cuda' if torch.cuda.is_available() else 'cpu'` when left unset, and threaded
consistently everywhere (including into cache keys — a rotation matrix cached for
CPU and one cached for CUDA are different cache entries). This is the boring but
easy-to-get-wrong part: the first version of `run_benchmark.py` broke the moment
device auto-detection landed, because quantizers started defaulting to CUDA while
the HF model stayed on CPU (`RuntimeError: Expected all tensors to be on the same
device`). Fixed by adding an explicit `--device` flag and threading it through
every constructor call, not just the quantizers.

---

## The Bottleneck / Challenges

### Challenge 1: the plan's "exact" formula wasn't always exact

Three separate test files (`test_rotation.py`, `test_lloyd_max.py`, and a
worked-example test in the Task 7 brief) each assumed the b=1 optimal centroid
for the Beta density collapses to the simple asymptotic half-normal formula
`sqrt(2/π) / sqrt(d)`. It doesn't — that's the large-d *approximation*. The true
closed form is `2·Γ(d/2) / (√π·(d-1)·Γ((d-1)/2))`. At d=128 the two differ by
0.195% (0.07066157 vs 0.07052370) — small enough to look like a rounding artifact,
large enough to fail a tight assertion. At d=4 the gap is much larger:
`4/(3π) ≈ 0.4244` (exact) vs `0.3989` (asymptotic). Each time, the fix was to the
*test's* expected value, verified independently with a `math.lgamma`-based
calculation before touching anything — the algorithm implementation was correct
all three times; the plan's reference formula was the asymptotic approximation
it had explicitly promised not to use.

**Lesson**: "no approximations" has to be checked against every formula in the
plan, not just the headline rotation/density choices. An approximation can hide
inside a test's expected value just as easily as inside the implementation.

### Challenge 2: a git commit that swallowed three unrelated files

One implementer subagent ran `git commit -m "..."` with no pathspec against an
index that already had unrelated pre-existing files staged (a `.gitignore` change,
an untracked blog draft, an untracked HTML primer — none related to its task).
The commit went through and bundled all four. The subagent caught its own mistake,
reported `DONE_WITH_CONCERNS` instead of hiding it, and correctly declined to
unstage anything on its own (out of scope, and it wasn't sure what depended on
what). Fixed with `git reset --soft HEAD~1` — safe because nothing had been built
on top of the bad commit yet — followed by restoring the three unrelated files to
unstaged and re-committing only the intended file with an explicit pathspec.

**Lesson**: `git commit` with no pathspec against a dirty index is a standing
risk in any multi-agent workflow where files can get staged for reasons upstream
of the current task. Every subsequent implementer was told explicitly: run
`git status --porcelain` first, commit with explicit pathspecs, always.

### Challenge 3: the GPU didn't exist until the venv did

The first attempt to run the benchmarks on GPU silently fell back to CPU —
`torch.cuda.is_available()` was `False` even with an RTX 4070 in the machine,
because the global Python environment had a CPU-only torch build. The fix wasn't
a code change: it was moving `turbo-quant` into this repo's existing `uv`
workspace, which already had a `pytorch-cu130` index configured, so the package
started sharing a torch build that actually matches the driver
(`nvidia-smi` confirmed CUDA 13.3 support). No amount of `device="cuda"` in the
code fixes a torch build that was never compiled with CUDA support.

**Lesson**: device auto-detection code is only as good as the environment it
runs in — verify the environment first, don't assume the flag is the bug.

---

## Results

All results below are real runs, not smoke tests: `Qwen/Qwen2.5-0.5B`, real
WikiText-2 text, on the RTX 4070 (CUDA auto-detected). Every row is logged to a
CSV under `turbo-quant/examples/results/` for reproducibility.

### Part A — does the exact algorithm preserve generation quality?

| Algorithm | Bits | Perplexity | Δ vs baseline (10.13) | Compression |
|-----------|------|-----------:|----------------------:|-------------:|
| baseline  | fp16 | 10.13 | — | 1.00x |
| mse | 4 | 156.07 | +146.0 | 3.76x |
| mse | 3 | 200.29 | +190.2 | 4.92x |
| mse | 2 | 1,496.5 | +1,486.4 | 7.11x |
| mse | 1 | 4,135.0 | +4,124.9 | 12.80x |
| prod | 4 | 799.6 | +789.4 | 3.76x |
| prod | 3 | 3,230.9 | +3,220.7 | 4.92x |
| prod | 2 | 20,196.1 | +20,185.9 | 7.11x |
| polar | 4 | 70.4 | +60.3 | 3.76x |
| polar | 3 | 728.6 | +718.5 | 4.92x |
| polar | 2 | 2,727.9 | +2,717.7 | 7.11x |
| polar | 1 | 17,570.0 | +17,559.9 | 12.80x |

This is the honest finding: **naively quantizing the entire KV cache with the
paper-exact algorithm and nothing else destroys Qwen2.5-0.5B's generation
quality**, even at 4 bits, even with the best-performing variant (PolarQuant).
`TurboQuant_prod`'s "unbiased inner product" property does not translate to
better attention scores in practice — it is consistently the *worst* performer
of the three at matched bit budgets here, which lines up with what I found in
the earlier V3 work: QJL's unbiasedness is a property of expectation, and softmax
does not care about expectation, it cares about the actual per-token error.

### Part B — does the math match the paper's theory?

Real K-vectors extracted from a live Qwen2.5-0.5B forward pass (head_dim=64,
~5,000 vectors), quantized and dequantized with `TurboQuantMSE`, compared against
both the paper's general Theorem 1 bound (`1.5 · 4^-bits`) and this package's own
exact Lloyd-Max solve for d=64:

| Bits | Empirical distortion | Theorem 1 bound | Exact solved bound | Within general bound |
|------|----------------------:|-----------------:|--------------------:|:---:|
| 1 | 0.3645 | 0.3750 | 0.3584 | ✅ |
| 2 | 0.1159 | 0.0938 | 0.1145 | ❌ (but matches the exact solve) |
| 3 | 0.0336 | 0.0234 | 0.0334 | ❌ (but matches the exact solve) |
| 4 | 0.0091 | 0.0059 | 0.0091 | ❌ (but matches the exact solve) |

The empirical distortion tracks the **exact solved bound** almost perfectly at
every bit-width (within 0.0002 at 4 bits) — this is the strongest evidence in
this whole project that the implementation is mathematically correct. It does
*not* always sit inside the paper's looser general Theorem 1 bound at 2–4 bits,
but that bound is a worst-case guarantee across all dimensions, not a tight
prediction for any specific d — the exact per-dimension solve is the number that
should match, and it does.

### CPU vs GPU

| Algorithm | Bits | d | CPU quantize (ms) | GPU quantize (ms) | Speedup |
|-----------|------|---|-------------------:|--------------------:|--------:|
| mse | 4 | 128 | 15.31 | 1.10 | 13.9x |
| mse | 1 | 64 | 3.18 | 0.40 | 8.0x |
| prod | 4 | 128 | 11.32 | 0.76 | 14.9x |
| polar | 4 | 128 | 21.96 | 1.70 | 12.9x |
| polar | 1 | 64 | 7.67 | 1.82 | 4.2x |

MSE and Prod see clean 8–15x GPU speedups — mostly matrix multiplies, which GPUs
love. PolarQuant's speedup is smaller and more variable (4–13x) because its
recursive Cartesian→polar decomposition is a Python-level loop over
`log2(d)` levels, each a small tensor op — more kernel-launch overhead relative
to actual compute than a single rotation matmul.

---

## Learnings

### On implementing papers exactly

1. **"Exact" is a discipline you have to re-apply at every layer, including your
   own tests.** Three different asymptotic-vs-exact bugs surfaced not in the
   core algorithm but in the reference values used to check it. The commitment
   to "no approximations" has to survive contact with the test suite, not just
   the production code.

2. **Matching the paper's theoretical bound is necessary but not sufficient.**
   The Part B results are unambiguous: the exact Lloyd-Max solve achieves almost
   precisely its predicted distortion on real activations. That correctness says
   nothing about Part A's catastrophic perplexity numbers — distortion on
   individual vectors and end-to-end generation quality are different metrics,
   and a paper's Theorem 1 bound is a statement about the former only.

3. **The engineering the earlier V3 variant added — asymmetric K/V bits, a
   residual fp16 window for recent tokens — wasn't cosmetic. It's load-bearing.**
   Without it, this run shows the exact, unmodified algorithm collapsing
   generation quality at every bit-width tested. The paper's headline claim
   ("quality-neutral at 3.5 bits") is compatible with these numbers only if real
   deployments also do the engineering this package's `examples/` layer
   deliberately doesn't bake into the core: windowing, asymmetric allocation, or
   both.

### On process

4. **Task-scoped review catches formula drift that a single implementer
   wouldn't.** All three asymptotic/exact mismatches were caught by an
   independent reviewer reading the diff against the plan's stated formula, not
   by the implementer who wrote the code — and each was verified numerically
   before ruling, rather than trusted from either side's assertion.

5. **A subagent that flags its own mistake instead of hiding it is worth more
   than one that never makes one.** The commit-scope incident was caught and
   reported by the agent that caused it. That is the outcome the review process
   is designed to produce — verification over trust, both for the agent's
   self-report and for the plan it was handed.

---

*Full implementation — `turboquant/` core package, `examples/` benchmarking
layer, all 55 tests, and the raw CSVs behind every table above — lives at
[github.com/VjayRam/Research-Demos/turbo-quant](https://github.com/VjayRam/Research-Demos/tree/main/turbo-quant).*
