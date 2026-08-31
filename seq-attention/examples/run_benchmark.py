"""Reproduces Table 2 of the Sequential Attention paper: baseline (all
features) vs Sequential Attention-selected top-k features, on MNIST,
Fashion-MNIST, and ISOLET, with a small MLP. Logs results to CSV.

Run: python examples/run_benchmark.py [--dataset mnist|fashion_mnist|isolet|all] [--output PATH]
"""

import argparse

import torch

from seqattention.models import AttentionGatedMLP
from seqattention.onepass import select_features_onepass

from data import load_fashion_mnist, load_isolet, load_mnist
from results_logger import default_output_path, write_csv

# (loader, num_features, num_classes, k, hidden_dim)
DATASETS = {
    "mnist": (load_mnist, 784, 10, 50, 256),
    "fashion_mnist": (load_fashion_mnist, 784, 10, 50, 256),
    "isolet": (load_isolet, 617, 26, 50, 256),
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_classifier(model, X, y, steps=2000, lr=1e-3):
    optimizer = torch.optim.Adam(model.body.parameters(), lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(X), y)
        loss.backward()
        optimizer.step()
    return model


def evaluate(model, X, y) -> float:
    with torch.no_grad():
        preds = model(X).argmax(dim=1)
        return (preds == y).float().mean().item()


def pin_selected_features(model):
    """Marks every feature in `model` as selected, so the mask's softmax
    gate degenerates to an all-ones vector (no unselected features remain to
    leak softmax mass onto). `model` must already be sized to exactly the
    selected feature count -- see `run_dataset`, which slices X down to the
    selected columns before constructing this model."""
    with torch.no_grad():
        for idx in range(model.mask.num_features):
            model.mask.select(idx)
    model.mask.attention_logits.requires_grad_(False)
    return model


def run_dataset(name, loader, num_features, num_classes, k, hidden_dim, seed=0):
    X_train, y_train = loader(train=True)
    X_test, y_test = loader(train=False)
    X_train, y_train = X_train.to(DEVICE), y_train.to(DEVICE)
    X_test, y_test = X_test.to(DEVICE), y_test.to(DEVICE)

    torch.manual_seed(seed)
    baseline = AttentionGatedMLP(num_features, hidden_dim, num_classes, seed=seed).to(DEVICE)
    baseline = train_classifier(baseline, X_train, y_train)
    baseline_acc = evaluate(baseline, X_test, y_test)

    y_train_float = y_train.float()
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
    selected_model = train_classifier(selected_model, X_train_selected, y_train)
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

    output = args.output or default_output_path("run_benchmark")
    write_csv(rows, output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
