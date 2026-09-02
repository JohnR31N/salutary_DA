"""PyTorch numeric-compatibility constants: the reproduction contract.

Every value in this module exists so that JAX/flax training reproduces
baselines that were originally trained in PyTorch. The names say "PyTorch"
on purpose: they document WHY these exact numbers are used. Do not "clean"
them into neutral names or swap them for flax defaults -- changing any of
them invalidates the benchmark tables trained against them.

- ``PYTORCH_KAIMING_CONV_INIT``          torch.nn.init.kaiming_normal_(fan_out)
- ``PYTORCH_MODULE_DEFAULT_CONV_INIT``   torch.nn.Conv2d default (uniform fan_in, gain 1/3)
- ``PYTORCH_3X3_PADDING`` / ``PYTORCH_7X7_PADDING``
                                         torch's symmetric explicit padding
                                         (flax SAME resolves differently)
- ``pytorch_default_linear_kernel_init`` / ``pytorch_default_linear_bias_init``
                                         torch.nn.Linear default bounds
"""

from __future__ import annotations

import math

import flax.linen as nn
import jax
import jax.numpy as jnp

PYTORCH_KAIMING_CONV_INIT = nn.initializers.variance_scaling(
    scale=2.0,
    mode="fan_out",
    distribution="normal",
)

PYTORCH_MODULE_DEFAULT_CONV_INIT = nn.initializers.variance_scaling(
    scale=1.0 / 3.0,
    mode="fan_in",
    distribution="uniform",
)

PYTORCH_3X3_PADDING = (
    (
        1,
        1,
    ),
    (
        1,
        1,
    ),
)

PYTORCH_7X7_PADDING = (
    (
        3,
        3,
    ),
    (
        3,
        3,
    ),
)


def pytorch_default_linear_kernel_init(
    rng: jax.Array,
    shape: tuple[int, ...],
    dtype=jnp.float32,
) -> jnp.ndarray:
    """Match torch.nn.Linear default weight bounds for a given fan-in."""
    fan_in = shape[0]
    bound = 1.0 / math.sqrt(fan_in)

    return jax.random.uniform(
        rng,
        shape,
        dtype,
        minval=-bound,
        maxval=bound,
    )


def pytorch_default_linear_bias_init(
    fan_in: int,
):
    """Build a torch.nn.Linear default bias initializer."""

    def init(
        rng: jax.Array,
        shape: tuple[int, ...],
        dtype=jnp.float32,
    ) -> jnp.ndarray:
        bound = 1.0 / math.sqrt(fan_in)

        return jax.random.uniform(
            rng,
            shape,
            dtype,
            minval=-bound,
            maxval=bound,
        )

    return init
