# Sequential Attention — Design Spec

**Date:** 2026-08-28
**Status:** Approved for planning

## Summary

Add a new `seq-attention/` package, `seqattention`, implementing Google's
**Sequential Attention** feature-selection algorithm — *"Sequential Attention
for Feature Selection"* (Yasuda, Bateni, Chen, Fahrbach, Fu, Mirrokni,
ICLR 2023, [arXiv:2209.14881](https://arxiv.org/abs/2209.14881)). This is a
greedy input-feature-selection method for tabular/pixel-as-feature data,
built on a softmax attention mask over candidate features — not a
transformer/LLM attention mechanism. Scope: the core algorithm (Algorithm 1
+ the one-pass training trick), a numerical demonstration of its proven
equivalence to Orthogonal Matching Pursuit (Theorem 1.1/3.3), and a
benchmark reproduction of the paper's Table 2 results on MNIST,
Fashion-MNIST, and ISOLET with a small MLP. Mirrors `turbo-quant/`'s
project structure and conventions (package + `tests/` + `examples/` +
`README.md`), added as a sibling workspace member. All work stays inside
`seq-attention/`.

## Motivation

The repo's existing interactive primer
(`seq-attention/sequential-attention.html`) explains the algorithm visually
but contains no runnable, paper-accurate implementation. The root
`README.md` already lists Sequential Attention as "Implementation pending."
`turbo-quant/` established this repo's pattern for a from-scratch,
paper-exact implementation with a validated results table — this spec
follows the same pattern for Sequential Attention: correctness first (no
shortcuts on the mask/selection math), then real benchmark numbers compared
against the paper's own reported values.

## Non-Goals

- No transformer/LLM self-attention integration. "Sequential Attention" in
  this paper is an unrelated use of the word "attention" — a differentiable
  feature-selection mask, not a token-mixing mechanism. Nothing in this spec
  touches `turbo-quant/` or any transformer architecture.
- No SequentialAttention++ (the block/structured-sparsification follow-up
  mentioned in the blog for pruning neurons/channels in larger networks).
  Confirmed out of scope for this pass.
- No hyperparameter search framework. The benchmark reproduction uses the
  paper's reported hyperparameters (or the closest documented equivalent)
  rather than re-tuning from scratch.
- No distributed/multi-GPU training. Single local RTX 4070 Laptop (8GB) is
  more than sufficient for MLPs on MNIST/Fashion-MNIST/ISOLET-scale data.
- No new top-level primer HTML rewrite — `sequential-attention.html` stays
  as-is; this spec is about runnable code, not the existing visual artifact.

## Architecture

### Module layout

```
seq-attention/
├── seqattention/
│   ├── __init__.py           # public API re-exports
│   ├── mask.py                # softmax attention mask (Algorithm 1's single logit vector w)
│   ├── selector.py             # Algorithm 1: greedy sequential selection loop
│   ├── onepass.py              # one-pass training trick (phase scheduling within one run)
│   ├── omp.py                  # reference OMP + Sequential LASSO, for the equivalence demo
│   └── models.py               # small MLP with an attention-gated input layer
├── tests/
│   ├── __init__.py
│   ├── test_mask.py
│   ├── test_selector.py
│   ├── test_onepass.py
│   └── test_omp_equivalence.py
├── examples/
│   ├── run_omp_equivalence.py  # synthetic sparse-regression equivalence demo
│   ├── run_benchmark.py        # MNIST / Fashion-MNIST / ISOLET reproduction (Table 2)
│   ├── data.py                 # dataset loading incl. ISOLET fetch/cache
│   ├── results_logger.py       # CSV logging, mirrors turbo-quant's
│   └── results/                # gitignored CSV output dir
├── pyproject.toml
└── README.md                   # existing sequential-attention.html referenced from here
```

`seq-attention` is added to the root `pyproject.toml`'s
`[tool.uv.workspace] members` list alongside `turbo-quant`, and as a
`[tool.uv.sources]` workspace entry, matching how `turboquant` is wired in
today.

### Core algorithm (`mask.py`, `selector.py`, `onepass.py`)

**`mask.py`** implements the attention mask exactly as defined in
Algorithm 1: given a single per-feature attention-logit vector `w`,
already-selected features get a fixed weight of 1 (piecewise, not passed
through softmax), and the remaining unselected features compete via
softmax over their logits. `w` itself is the paper's entire
overparameterization (footnote 2) — there is no second learned weight
vector inside the mask; the model's own parameters (e.g. a linear layer's
weights) supply everything else, per Definition 3.1 and Appendix B.2.4.

**`selector.py`** implements Algorithm 1: repeatedly (1) train the
mask-gated model for some number of steps, (2) pick `argmax` attention
logit among currently-unselected features, (3) freeze that feature into the
selected set `S`, until `|S|` reaches the target feature count `k`.

**`onepass.py`** implements the paper's training-efficiency trick:
partition a single model's training run into `k` phases (one phase per
feature to be selected) rather than training `k` separate models from
scratch. Only the attention logits are reset between phases; model weights
and the running feature set `S` persist across phases.

### OMP equivalence demo (`omp.py`, `examples/run_omp_equivalence.py`)

`omp.py` implements plain Orthogonal Matching Pursuit (greedy pick of the
candidate feature maximizing correlation with the current residual, via
least-squares residual computation) and Sequential LASSO, as independent
reference implementations — not derived from `selector.py`'s code path, so
the comparison is meaningful.

`examples/run_omp_equivalence.py` builds a synthetic sparse linear
regression problem (`k` true generating features among `n` candidates, seeded
for reproducibility — same shape as the existing HTML primer's live
simulator) and runs three algorithms against it: plain OMP, Sequential
LASSO, and Regularized Linear Sequential Attention (`selector.py` in its
linear-regression configuration). Reports whether all three select the same
feature sequence, numerically demonstrating Theorem 1.1/3.3 rather than
just asserting it in prose.

### Benchmark reproduction (`examples/run_benchmark.py`, `models.py`, `examples/data.py`)

`models.py` provides a small MLP whose input layer is gated by the
Sequential Attention mask from `mask.py` — matching the paper's
experimental architecture (attention-gated input, standard MLP body).

`examples/data.py` loads MNIST and Fashion-MNIST via `torchvision`
(already a repo dependency) and ISOLET via a `requests`-based download from
the UCI ML repository (ISOLET isn't in `torchvision`/`torchtext`), caching
the raw files locally under `examples/data_cache/` (gitignored) so repeat
runs don't re-download.

`examples/run_benchmark.py` reproduces Table 2: for each dataset, trains
the MLP both with all features (baseline) and with Sequential
Attention-selected features at the paper's reported `k`, logging accuracy
before/after selection to CSV via `results_logger.py` (mirroring
`turbo-quant/examples/results_logger.py`'s pattern). Target numbers from
the paper (validated against the existing HTML primer's Results section):

| Dataset | Baseline (all features) | Sequential Attention (selected `k`) |
|---|---|---|
| MNIST | 0.944 | 0.956 |
| Fashion-MNIST | 0.843 | 0.854 |
| ISOLET | 0.866 | 0.920 |

These are the values the benchmark run is expected to reproduce (within
normal run-to-run training variance) — not hardcoded into the code, but the
standard this implementation is judged against.

## Testing Strategy

- `test_mask.py`: softmax normalization over unselected features sums to 1;
  selected features are pinned to weight 1 regardless of their logit; the
  mask's `gate()` equals its softmax mask (no second weight vector).
- `test_selector.py`: on a small synthetic problem with a known ground-truth
  sparse support, Algorithm 1 recovers exactly that support in the correct
  greedy order; `|S|` grows by exactly one feature per phase and never
  re-selects an already-selected feature.
- `test_onepass.py`: attention logits reset between phases while model
  weights and `S` persist; final selected set matches the multi-model
  (naively retrained-per-phase) baseline on the same synthetic problem.
- `test_omp_equivalence.py`: on a small synthetic sparse regression problem,
  assert (not just report) that plain OMP, Sequential LASSO, and Regularized
  Linear Sequential Attention select an identical feature sequence — this is
  the automated, CI-safe counterpart to `run_omp_equivalence.py`'s demo.
- Existing repo test conventions (pytest, `tests/__init__.py` package style)
  followed as in `turbo-quant/tests/`.
- Benchmark reproduction (`run_benchmark.py`) is verified by actually running
  it on the project's RTX 4070 and comparing against the Table 2 target
  numbers above — a real run, not just a unit test — before this work is
  considered done, per this repo's established practice of empirically
  validating results rather than trusting code review alone.

## Open Items for the Implementation Plan

- Exact MLP hyperparameters (hidden width/depth, optimizer, learning rate,
  number of training steps per phase) are implementation-plan decisions;
  this spec fixes only the architecture shape (attention-gated input +
  MLP body) and the target `k` per dataset from Table 2.
- ISOLET's exact UCI download URL/format handling is an implementation
  detail for the plan, not fixed here.
- Whether `run_benchmark.py` takes CLI flags (e.g. `--dataset`, `--k`) is an
  implementation-plan decision, following whatever pattern
  `turbo-quant/examples/run_benchmark.py` already uses in this repo.
