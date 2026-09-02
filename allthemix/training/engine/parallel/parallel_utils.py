"""Cross-device batch utilities shared by the parallel training engine."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from allthemix.utils.parallel import shard_array


def flatten_cross_device_batch(
    value: jnp.ndarray,
) -> jnp.ndarray:
    """Gather a per-device batch and flatten devices into the batch axis."""
    gathered_value = jax.lax.all_gather(
        value,
        axis_name="batch",
    )

    return gathered_value.reshape(
        (
            gathered_value.shape[0] * gathered_value.shape[1],
        )
        + value.shape[1:],
    )


def global_no_repeat_permutation(
    rng: jax.Array,
    batch_size: int,
) -> jnp.ndarray:
    """Create a global permutation without fixed points when possible."""
    if batch_size <= 1:
        return jnp.arange(
            batch_size,
        )

    order = jax.random.permutation(
        rng,
        batch_size,
    )

    shifted_order = jnp.roll(
        order,
        1,
    )

    return jnp.arange(
        batch_size,
    ).at[
        order
    ].set(
        shifted_order,
    )


def make_cross_device_pairs(
    rng: jax.Array,
    images: jnp.ndarray,
    labels: jnp.ndarray,
    aux_info: dict[str, jnp.ndarray] | None = None,
    no_repeat: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, dict[str, jnp.ndarray]]:
    """Pair local examples with examples sampled from the global device batch."""
    global_images = flatten_cross_device_batch(
        images,
    )
    global_labels = flatten_cross_device_batch(
        labels,
    )
    global_batch_size = global_labels.shape[0]
    local_batch_size = labels.shape[0]

    shared_rng = jax.lax.all_gather(
        rng,
        axis_name="batch",
    )[0]

    if no_repeat:
        global_permutation = global_no_repeat_permutation(
            rng=shared_rng,
            batch_size=global_batch_size,
        )

    else:
        global_permutation = jax.random.permutation(
            shared_rng,
            global_batch_size,
        )
    device_index = jax.lax.axis_index(
        "batch",
    )
    start = device_index * local_batch_size
    local_permutation = jax.lax.dynamic_slice(
        global_permutation,
        (
            start,
        ),
        (
            local_batch_size,
        ),
    )
    local_relative_permutation = (
        local_permutation
        - start
    )

    paired_images = global_images[
        local_permutation
    ]
    paired_labels = global_labels[
        local_permutation
    ]

    paired_aux = {}

    if aux_info is not None:
        for key, value in aux_info.items():
            global_value = flatten_cross_device_batch(
                value,
            )
            paired_aux[f"paired_{key}"] = global_value[
                local_permutation
            ]

    return paired_images, paired_labels, local_relative_permutation, paired_aux


def pad_to_device_multiple(
    array,
    fill_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad a batch so PMAP can shard it while tracking real examples."""
    array = np.asarray(
        array,
    )

    num_devices = jax.local_device_count()
    batch_size = array.shape[0]
    remainder = batch_size % num_devices
    pad_size = 0 if remainder == 0 else num_devices - remainder

    valid_mask = np.concatenate(
        [
            np.ones(
                batch_size,
                dtype=np.float32,
            ),
            np.zeros(
                pad_size,
                dtype=np.float32,
            ),
        ],
        axis=0,
    )

    if pad_size == 0:
        return array, valid_mask

    pad_width = [
        (
            0,
            pad_size,
        )
    ] + [
        (
            0,
            0,
        )
        for _ in array.shape[1:]
    ]

    padded = np.pad(  # Add dummy samples only at the batch tail.
        array,
        pad_width=pad_width,
        mode="constant",
        constant_values=fill_value,
    )

    return padded, valid_mask


def shard_aux_info(
    aux_info: dict[str, Any],
) -> dict[str, jnp.ndarray]:
    """Shard every auxiliary tensor across devices."""
    return {
        key: jnp.asarray(
            shard_array(value),
        )
        for key, value in aux_info.items()
    }


def append_replicated_step_metrics(
    metric_lists: dict[str, list[float]],
    metrics: dict[str, jnp.ndarray],
) -> None:
    """Append first-replica scalar metrics into Python lists.

    The parallel engine holds replicated metric arrays; only replica 0
    is read, after a device_get.
    """
    for key, value in metrics.items():
        metric_lists.setdefault(
            key,
            [],
        ).append(
            float(
                jax.device_get(
                    value[0],
                )
            )
        )
