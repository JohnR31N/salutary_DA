from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import jax_utils
from flax.training.train_state import TrainState


def get_num_devices() -> int:
    """Return the number of local JAX devices."""
    return jax.local_device_count()


def replicate_state(
    state: TrainState,
) -> TrainState:
    """Replicate train state across local devices."""
    return jax_utils.replicate(state)


def unreplicate_state(
    state: TrainState,
) -> TrainState:
    """Return the first local-device copy of a replicated train state."""
    return jax_utils.unreplicate(state)


def create_device_rngs(
    rng: jax.Array,
) -> jax.Array:
    """Split one RNG key into one key per local device."""
    num_devices = get_num_devices()

    return jax.random.split(
        rng,
        num_devices,
    )


def shard_array(
    array: jnp.ndarray,
) -> jnp.ndarray:
    """Reshape a global leading batch dimension over local devices."""
    array = jnp.asarray(
        array,
    )
    num_devices = get_num_devices()
    batch_size = array.shape[0]

    if batch_size % num_devices != 0:
        raise ValueError(
            f"Batch size {batch_size} must be divisible by "
            f"num_devices {num_devices}."
        )

    return array.reshape(
        (
            num_devices,
            batch_size // num_devices,
            *array.shape[1:],
        )
    )


def shard_batch(
    images: jnp.ndarray,
    labels: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Shard image and label batches across local devices."""
    return shard_array(
        images,
    ), shard_array(
        labels,
    )


def can_shard_batch(
    batch_size: int,
) -> bool:
    """Return whether a batch can be evenly sharded across devices."""
    num_devices = get_num_devices()

    return batch_size % num_devices == 0  # Sharding requires no remainder.
