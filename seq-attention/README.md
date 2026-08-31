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

- The softmax attention mask over per-feature attention logits (Algorithm 1)
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

<!-- Filled in by Task 12 after a real run on the project's RTX 4070. -->

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
