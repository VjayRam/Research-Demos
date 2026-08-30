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


def pin_selected_features(model, selected: list[int]):
    with torch.no_grad():
        model.mask.overparam_weight.zero_()
        for idx in selected:
            model.mask.select(idx)
            model.mask.overparam_weight[idx] = 1.0
    model.mask.attention_logits.requires_grad_(False)
    model.mask.overparam_weight.requires_grad_(False)
    return model


def run_dataset(name, loader, num_features, num_classes, k, hidden_dim, seed=0):
    X_train, y_train = loader(train=True)
    X_test, y_test = loader(train=False)

    torch.manual_seed(seed)
    baseline = AttentionGatedMLP(num_features, hidden_dim, num_classes, seed=seed)
    baseline = train_classifier(baseline, X_train, y_train)
    baseline_acc = evaluate(baseline, X_test, y_test)

    y_train_float = y_train.float()
    selector_model_fn = lambda seed_: AttentionGatedMLP(num_features, hidden_dim, num_classes, seed=seed_)
    selected = select_features_onepass(
        model_factory=selector_model_fn,
        loss_fn=lambda y_pred, y_true: torch.nn.functional.cross_entropy(y_pred, y_true.long()),
        X=X_train, y=y_train, k=k, train_steps_per_phase=200, lr=1e-3, seed=seed,
    )

    torch.manual_seed(seed)
    selected_model = AttentionGatedMLP(num_features, hidden_dim, num_classes, seed=seed)
    selected_model = pin_selected_features(selected_model, selected)
    selected_model = train_classifier(selected_model, X_train, y_train)
    selected_acc = evaluate(selected_model, X_test, y_test)

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
