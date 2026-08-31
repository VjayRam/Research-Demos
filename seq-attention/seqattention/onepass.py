"""The paper's one-pass training trick: instead of training k independent
models (selector.py's select_features_naive), train a single deterministic
model across k phases, resetting the attention logits (not the model's other
weights, and not the growing selected set) between phases -- per Appendix
B.2.4's note that resetting the attention weights each phase is important,
but resetting the network weights is not."""

from typing import Callable

import torch


def select_features_onepass(
    model_factory: Callable[[int], torch.nn.Module],
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    X: torch.Tensor,
    y: torch.Tensor,
    k: int,
    train_steps_per_phase: int = 200,
    lr: float = 0.05,
    seed: int = 0,
) -> list[int]:
    torch.manual_seed(seed)
    model = model_factory(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    selected: list[int] = []
    for phase in range(k):
        for _ in range(train_steps_per_phase):
            optimizer.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            unselected = (~model.mask.selected).nonzero(as_tuple=True)[0]
            best = unselected[torch.argmax(model.mask.attention_logits[unselected])].item()
            model.mask.select(best)
            selected.append(best)
            model.mask.reset_logits(seed=seed + phase + 1)
    return selected
