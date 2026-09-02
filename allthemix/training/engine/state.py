"""Shared train-state container for both engines."""

from __future__ import annotations

from typing import Any

from flax import struct
from flax.training.train_state import TrainState


class TrainStateWithBatchStats(TrainState):
    batch_stats: Any = struct.field(pytree_node=True)

    sumix_apply_fn: Any = struct.field(pytree_node=False)
    sumix_params: Any = struct.field(pytree_node=True)
    sumix_tx: Any = struct.field(pytree_node=False)
    sumix_opt_state: Any = struct.field(pytree_node=True)
