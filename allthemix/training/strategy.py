from __future__ import annotations

from typing import Any, Protocol

import jax
import jax.numpy as jnp
import numpy as np


class BatchTrainingStrategy(Protocol):
    """Interface for methods that own a task update for one training batch."""

    def train_step(
        self,
        task_state,
        images: jnp.ndarray,
        labels: jnp.ndarray,
        rng: jax.Array,
    ) -> tuple:
        """Update task state and return loss, accuracy, and scalar metrics."""
        ...


class ValidationAwareStrategy(Protocol):
    """Interface for methods that update state from train and validation batches."""

    def next_meta_batch(
        self,
    ) -> Any:
        """Return the next held-out batch used by the strategy."""
        ...

    def train_step(
        self,
        task_state,
        images: jnp.ndarray,
        labels: jnp.ndarray,
        meta_images: jnp.ndarray,
        meta_labels: jnp.ndarray,
        rng: jax.Array,
    ) -> tuple:
        """Update shared task state and strategy-owned state."""
        ...

    def finish_epoch(
        self,
        pair_sums: np.ndarray,
        pair_counts: np.ndarray,
    ) -> dict[str, float]:
        """Finalize host-side strategy statistics after an epoch."""
        ...
