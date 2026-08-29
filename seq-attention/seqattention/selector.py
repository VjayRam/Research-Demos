"""Algorithm 1 (Yasuda et al., ICLR 2023), applied literally: each phase
trains a fresh model from scratch, with previously-selected features
pre-pinned into the mask, then greedily adds the argmax unselected
attention logit to the selected set. This is the "k separate models"
reading of the algorithm; onepass.py implements the paper's more efficient
single-model variant and is tested against this function's output."""

from typing import Callable

import torch


def select_features_naive(
    model_factory: Callable[[int], torch.nn.Module],
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    X: torch.Tensor,
    y: torch.Tensor,
    k: int,
    train_steps: int = 200,
    lr: float = 0.05,
    seed: int = 0,
) -> list[int]:
    selected: list[int] = []
    for phase in range(k):
        torch.manual_seed(seed + phase)
        model = model_factory(seed + phase)
        for idx in selected:
            model.mask.select(idx)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(train_steps):
            optimizer.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            unselected = (~model.mask.selected).nonzero(as_tuple=True)[0]
            best = unselected[torch.argmax(model.mask.attention_logits[unselected])].item()
        selected.append(best)
    return selected
