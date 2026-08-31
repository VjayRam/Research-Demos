# Sequential Attention

A paper-accurate PyTorch implementation of Google's **Sequential Attention**
feature-selection algorithm, based on *"Sequential Attention for Feature
Selection"* (Yasuda, Bateni, Chen, Fahrbach, Fu, Mirrokni, ICLR 2023,
[arXiv:2209.14881](https://arxiv.org/abs/2209.14881)).

This is a greedy input-feature-selection method built on a softmax attention
mask over candidate features — not a transformer/LLM attention mechanism.
See [`sequential-attention.html`](sequential-attention.html) for an
interactive walkthrough of the math.

## What is implemented

- The softmax attention mask over per-feature attention logits
  (`seqattention/mask.py`)
- Algorithm 1: greedy sequential feature selection, both the naive
  per-phase-retrain reading (`seqattention/selector.py`) and the paper's
  more efficient one-pass training trick (`seqattention/onepass.py`)
- Reference Orthogonal Matching Pursuit and Sequential LASSO
  implementations (`seqattention/omp.py`), used to numerically demonstrate
  their proven equivalence to Regularized Linear Sequential Attention
  (Theorem 1.1/3.3) — see `examples/run_omp_equivalence.py`
- A benchmark reproduction of the paper's Table 2 results on MNIST,
  Fashion-MNIST, and ISOLET with a small MLP (`examples/run_benchmark.py`)

## Results

Run on the project's RTX 4070 Laptop GPU, MLP with `hidden_dim=256`,
`k=50` selected features per dataset (`train_steps_per_phase=200` during
selection, `steps=2000` for final classifier training — fixed, untuned
hyperparameters; no per-dataset search, per this project's scope).

| Dataset | Baseline (all features) | Sequential Attention (k=50) | Paper (Table 2) |
|---|---|---|---|
| MNIST | 0.9782 | 0.9409 | 0.944 -> 0.956 |
| Fashion-MNIST | 0.8876 | 0.8602 | 0.843 -> 0.854 |
| ISOLET | 0.9532 | 0.9089 | 0.866 -> 0.920 |

Selected-feature accuracy comes in below the full-feature baseline on all
three datasets, unlike the paper's Table 2 (where the selected 50 features
*beat* the full-feature baseline on every dataset). This is a real,
honestly-reported shortfall of this reproduction, not an artifact: an
earlier version of this table had baseline numbers depressed by a bug (the
baseline model's feature mask was left untrained/unpinned, so it trained
through a near-uniform ~1/num_features gate instead of a genuine
full-feature pass — caught in review, fixed, and this table re-run after
the fix). With the bug fixed, the full-feature baseline is a strong
classifier in its own right (as expected — an MLP with all pixels/features
available has strictly more information than one restricted to 50), and
this project's simple, single-run, untuned selection setup isn't enough to
close that gap. The paper's own result likely depends on the extensive
per-dataset hyperparameter tuning this project deliberately doesn't do
(see Non-Goals in the design spec) plus possibly other regularization
choices not reproduced here.

The selection procedure itself is still verified to be doing real,
non-random work: substituting 50 *random* ISOLET features into the same
training pipeline scores only 0.83-0.85 across three random seeds, well
below the actual selected features' 0.9089 — the gap to baseline is a
"selected 50 features isn't enough signal, at these hyperparameters, to
match the full 617/784" story, not a broken selection algorithm.

Reproduce with: `python examples/run_benchmark.py --dataset all`. Full CSV:
[`examples/results/run_benchmark_20260831_055324.csv`](examples/results/run_benchmark_20260831_055324.csv).

### Known limitations

- Fixed, untuned hyperparameters shared across all three datasets — no
  per-dataset search (deliberate, per the design spec's Non-Goals).
- Full-batch gradient descent (no minibatching), no train/validation split
  — selection and final-classifier training both see the full training
  set, evaluated once against the held-out test set.
- `select_features_onepass` (the one-pass training trick) is more
  sensitive to learning rate than `select_features_naive` on small
  problems — see the comment on `tests/test_onepass.py`'s ground-truth
  test for a concrete, verified example of this fidelity gap.
- The OMP/Sequential-LASSO/Sequential-Attention equivalence is proven
  exactly (Theorem 1.1/3.3) and CI-tested only on an orthonormal design
  matrix, where OMP's residual refit is closer to a simple correlation
  sort; `examples/run_omp_equivalence.py` additionally demonstrates
  agreement on a non-orthonormal (correlated) design as a less trivial
  check.
- ISOLET's raw features are not standardized (unlike MNIST/Fashion-MNIST's
  [0, 1] pixel scaling), which the softmax gate multiplies directly.

### Investigating the gap to the paper

Candidate explanations for why selected-feature accuracy trails baseline
here (opposite of the paper's Table 2), tested as standalone diagnostic
experiments — not adopted into `run_benchmark.py` itself, since each
applies a *symmetric* change to both baseline and selected models to
isolate its effect on the gap (except where noted):

- **Baseline over-training (falsified).** Hypothesis: the fixed 2000-step,
  no-regularization training budget lets the baseline over-fit relative to
  the 50-feature model. Tested by applying the same held-out-validation,
  patience-based early stopping to both models
  (`examples/run_benchmark_early_stopping.py`). Result: accuracy barely
  moved (e.g. MNIST baseline 0.9782 → 0.9753, selected 0.9409 → 0.9419) and
  the gap only narrowed ~10% relative on each dataset — nowhere near
  flipping direction. Full-batch Adam on these dataset sizes converges
  smoothly enough that validation loss rarely plateaus early within 2000
  steps, so this is not the driver of the baseline's advantage. CSV:
  [`examples/results/run_benchmark_earlystop_experiment_20260831_131827.csv`](examples/results/run_benchmark_earlystop_experiment_20260831_131827.csv).
- **First-layer capacity mismatch (confirmed — the main lever found so
  far; the one asymmetric experiment here).** Hypothesis: with a shared
  `hidden_dim=256`, the baseline's first layer has `num_features *
  hidden_dim` parameters (e.g. 784*256 on MNIST) versus the selected
  model's `k * hidden_dim` (50*256) — a 15.7x difference unrelated to how
  much real signal the extra features carry. Tested by shrinking *only*
  the baseline's `hidden_dim` so its first layer has (approximately) the
  same parameter count as the selected model's: `hidden_dim_baseline =
  round(k * hidden_dim / num_features)` (`examples/run_benchmark_capacity_matched.py`;
  16 for MNIST/Fashion-MNIST, 21 for ISOLET). Result: the gap closed 80%
  on MNIST (0.9782 → 0.9484 vs. selected's 0.9409) and **flipped on
  Fashion-MNIST** (0.8561 baseline vs. 0.8602 selected — selection wins,
  matching the paper's direction), with a smaller 19% narrowing on ISOLET
  (0.9532 → 0.9448 vs. 0.9089). This is the strongest evidence so far that
  this reproduction's baseline was simply over-provisioned relative to the
  selected model, not that the extra features carry proportionally more
  signal. Not folded into `run_benchmark.py` as the new default, since the
  paper's own baseline architecture (Table 2's exact hidden width) hasn't
  been verified — this parameter-matching heuristic is one plausible way
  to approach it, not confirmed to be *the* way. CSV:
  [`examples/results/run_benchmark_capacity_matched_experiment_20260831_133343.csv`](examples/results/run_benchmark_capacity_matched_experiment_20260831_133343.csv).

## File structure

```
seqattention/
├── mask.py       # softmax attention mask over per-feature attention logits
├── selector.py   # Algorithm 1, naive per-phase retrain
├── onepass.py    # Algorithm 1, one-pass training trick
├── omp.py        # OMP + Sequential LASSO reference implementations
└── models.py     # mask-gated linear regression and MLP models
examples/
├── run_omp_equivalence.py   # OMP/LASSO/Sequential Attention equivalence demo
├── run_benchmark.py         # Table 2 reproduction (MNIST/Fashion-MNIST/ISOLET)
├── data.py                  # dataset loading, incl. ISOLET fetch/cache
└── results_logger.py        # CSV result logging
```

## Installation

From the repo root:
```bash
uv sync
```

## Usage

```bash
cd seq-attention
python examples/run_omp_equivalence.py
python examples/run_benchmark.py --dataset all
python -m pytest tests/ -v
```
