"""Reference implementations of Orthogonal Matching Pursuit and Sequential
LASSO, independent of seqattention's selector/onepass code path, used to
numerically demonstrate the paper's proven equivalence between Regularized
Linear Sequential Attention, Sequential LASSO, and OMP (Theorem 1.1/3.3)
rather than merely asserting it."""

import torch


def orthogonal_matching_pursuit(X: torch.Tensor, y: torch.Tensor, k: int) -> list[int]:
    """Greedily picks the unselected column of X most correlated with the
    current residual, then re-solves least squares over all selected
    columns to update the residual before the next pick."""
    selected: list[int] = []
    residual = y.clone()
    for _ in range(k):
        correlations = (X.T @ residual).abs()
        correlations[selected] = -float("inf")
        best = torch.argmax(correlations).item()
        selected.append(best)
        X_s = X[:, selected]
        coef = torch.linalg.lstsq(X_s, y.unsqueeze(1)).solution.squeeze(1)
        residual = y - X_s @ coef
    return selected


def sequential_lasso(
    X: torch.Tensor, y: torch.Tensor, k: int, lam: float = 0.01, steps: int = 500, lr: float = 0.05
) -> list[int]:
    """At each phase, fits an L1-regularized regression over all features
    (already-selected features excluded from the penalty), then greedily
    adds the largest-magnitude unselected coefficient to the selected set."""
    n, d = X.shape
    selected: list[int] = []
    for _ in range(k):
        coef = torch.zeros(d, requires_grad=True)
        optimizer = torch.optim.Adam([coef], lr=lr)
        unselected_mask = torch.ones(d, dtype=torch.bool)
        unselected_mask[selected] = False
        for _ in range(steps):
            optimizer.zero_grad()
            y_hat = X @ coef
            penalty = lam * coef[unselected_mask].abs().sum()
            loss = torch.mean((y_hat - y) ** 2) + penalty
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            candidates = coef.abs().clone()
            candidates[selected] = -float("inf")
            best = torch.argmax(candidates).item()
        selected.append(best)
    return selected
