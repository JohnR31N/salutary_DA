from __future__ import annotations

import jax
import jax.numpy as jnp

from allthemix.methods.output import MixOutput
from allthemix.methods.utils.validation import (
    validate_labels_match_images,
    validate_nhwc_images,
    validate_num_classes,
)


def baseline_mixer(
    rng: jax.Array,
    images: jnp.ndarray,
    labels: jnp.ndarray,
    num_classes: int,
) -> MixOutput:
    """Return the batch unchanged, in the standard mixer container."""
    del rng

    validate_num_classes(
        num_classes,
    )
    validate_nhwc_images(
        images=images,
        method_name="Baseline",
    )
    validate_labels_match_images(
        labels=labels,
        images=images,
        method_name="Baseline",
    )

    return MixOutput(
        images=images,
        labels_a=labels,
        labels_b=labels,
        lam=jnp.ones(
            (),
            dtype=images.dtype,
        ),
        perm=jnp.arange(
            labels.shape[0],
            dtype=jnp.int32,
        ),
    )
