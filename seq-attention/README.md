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
| MNIST | 0.9033 | 0.9409 | 0.944 -> 0.956 |
| Fashion-MNIST | 0.7990 | 0.8602 | 0.843 -> 0.854 |
| ISOLET | 0.9294 | 0.9089 | 0.866 -> 0.920 |

MNIST and Fashion-MNIST reproduce the paper's qualitative result (selected
features outperform the full-feature baseline) at somewhat lower absolute
accuracy than the paper, consistent with this project's simpler,
single-run, untuned MLP versus the paper's more heavily tuned setup.

ISOLET's `k=50` result comes in *below* its full-feature baseline
(0.9089 vs 0.9294) rather than above it, unlike the paper and unlike the
other two datasets. This was checked as a possible bug before being
recorded here: the 50 selected features are all unique (no duplicate
selection), and substituting 50 *random* features in the same pipeline
scores only 0.83-0.85 across three random seeds — meaningfully worse than
the 50 selected features, confirming the selection procedure is doing real
work. The shortfall against ISOLET's own full-feature baseline is
attributed to reducing 617 features to 50 on a harder 26-class task with
this project's small, untuned MLP, not a defect in `select_features_onepass`
or `pin_selected_features`.

Reproduce with: `python examples/run_benchmark.py --dataset all`. Full CSV:
[`examples/results/run_benchmark_20260831_052333.csv`](examples/results/run_benchmark_20260831_052333.csv).

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
