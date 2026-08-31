"""Diagnostic experiment, NOT the paper-accurate benchmark (see run_benchmark.py
for that). Tests the hypothesis that our baseline beats the selected-feature
model on all three datasets because it's trained to convergence with a fixed
step budget and no regularization, unlike the paper's Table 2 setup (whose
own baseline numbers are notably lower than ours). Applies the SAME
early-stopping regime -- a held-out validation split, patience-based stopping
on validation loss -- to both the baseline and the selected model, so any
change in the baseline-vs-selected gap is attributable to reduced
over-training rather than to treating the two models differently.

Run: python examples/run_benchmark_early_stopping.py [--dataset mnist|fashion_mnist|isolet|all] [--output PATH]
"""

import argparse

import torch

from seqattention.models import AttentionGatedMLP
from seqattention.onepass import select_features_onepass

from data import load_fashion_mnist, load_isolet, load_mnist
from results_logger import default_output_path, write_csv
from run_benchmark import DATASETS, DEVICE, evaluate, pin_selected_features


def train_classifier_early_stopping(
    model, X, y, max_steps=2000, lr=1e-3, val_frac=0.1, patience=5, eval_every=25, seed=0
):
    """Same optimizer/loss as run_benchmark.py's train_classifier, but carves
    off a validation split and stops once val loss hasn't improved for
    `patience` consecutive checks, restoring the best-val-loss weights."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(X.shape[0], generator=generator).to(X.device)
    n_val = max(1, int(X.shape[0] * val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    optimizer = torch.optim.Adam(model.body.parameters(), lr=lr)
    best_val_loss = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    checks_without_improvement = 0

    for step in range(max_steps):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(X_tr), y_tr)
        loss.backward()
        optimizer.step()

        if (step + 1) % eval_every == 0:
            with torch.no_grad():
                val_loss = torch.nn.functional.cross_entropy(model(X_val), y_val).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                checks_without_improvement = 0
            else:
                checks_without_improvement += 1
                if checks_without_improvement >= patience:
                    break

    model.load_state_dict(best_state)
    return model


def run_dataset(name, loader, num_features, num_classes, k, hidden_dim, seed=0):
    X_train, y_train = loader(train=True)
    X_test, y_test = loader(train=False)
    X_train, y_train = X_train.to(DEVICE), y_train.to(DEVICE)
    X_test, y_test = X_test.to(DEVICE), y_test.to(DEVICE)

    torch.manual_seed(seed)
    baseline = AttentionGatedMLP(num_features, hidden_dim, num_classes, seed=seed).to(DEVICE)
    baseline = pin_selected_features(baseline)
    baseline = train_classifier_early_stopping(baseline, X_train, y_train, seed=seed)
    baseline_acc = evaluate(baseline, X_test, y_test)

    selector_model_fn = lambda seed_: AttentionGatedMLP(num_features, hidden_dim, num_classes, seed=seed_).to(DEVICE)
    selected = select_features_onepass(
        model_factory=selector_model_fn,
        loss_fn=lambda y_pred, y_true: torch.nn.functional.cross_entropy(y_pred, y_true.long()),
        X=X_train, y=y_train, k=k, train_steps_per_phase=200, lr=1e-3, seed=seed,
    )

    selected_idx = torch.tensor(selected, dtype=torch.long, device=DEVICE)
    X_train_selected = X_train[:, selected_idx]
    X_test_selected = X_test[:, selected_idx]

    torch.manual_seed(seed)
    selected_model = AttentionGatedMLP(len(selected), hidden_dim, num_classes, seed=seed).to(DEVICE)
    selected_model = pin_selected_features(selected_model)
    selected_model = train_classifier_early_stopping(selected_model, X_train_selected, y_train, seed=seed)
    selected_acc = evaluate(selected_model, X_test_selected, y_test)

    return {
        "dataset": name,
        "k": k,
        "baseline_accuracy": round(baseline_acc, 4),
        "selected_accuracy": round(selected_acc, 4),
        "selected_features": ";".join(str(i) for i in selected),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASETS) + ["all"], default="all")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    rows = []
    for name in names:
        loader, num_features, num_classes, k, hidden_dim = DATASETS[name]
        row = run_dataset(name, loader, num_features, num_classes, k, hidden_dim)
        print(row)
        rows.append(row)

    output = args.output or default_output_path("run_benchmark_earlystop_experiment")
    write_csv(rows, output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
