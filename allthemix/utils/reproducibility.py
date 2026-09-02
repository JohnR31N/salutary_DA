from __future__ import annotations

import random

import numpy as np
import tensorflow as tf


def resolve_data_seed(
    experiment_seed: int,
    data_seed: int,
) -> int:
    """Resolve the data seed, using the experiment seed by default."""
    if experiment_seed < 0:
        raise ValueError("seed must be >= 0.")

    if data_seed < -1:
        raise ValueError("data_seed must be -1 or >= 0.")

    return experiment_seed if data_seed == -1 else data_seed


def seed_everything(
    seed: int,
    strict_determinism: bool = False,
) -> None:
    """Seed host RNGs before datasets, models, or policies are created."""
    if seed < 0:
        raise ValueError("seed must be >= 0.")

    random.seed(
        seed,
    )
    np.random.seed(
        seed,
    )
    tf.random.set_seed(
        seed,
    )

    if strict_determinism:
        tf.config.experimental.enable_op_determinism()
