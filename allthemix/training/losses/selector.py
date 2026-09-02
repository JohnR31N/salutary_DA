from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp

from allthemix.training.losses.cross_entropy import cross_entropy
from allthemix.training.losses.mixup_loss import mixup_loss

CriterionFn = Callable[..., jnp.ndarray]


def get_criterion(name: str) -> CriterionFn:
    """Get criterion."""
    criterion_name = name.lower()

    if criterion_name in {"ce", "cross_entropy"}:
        return cross_entropy

    if criterion_name == "mixup_loss":
        return mixup_loss

    raise ValueError(f"Unsupported criterion: {name}")