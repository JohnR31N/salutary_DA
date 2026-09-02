from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from flax.core import freeze

from allthemix.utils.checkpoint import (
    _merge_matching_leaves,
    restore_state_file,
    save_state_file,
)


def test_state_file_round_trip(
    tmp_path,
) -> None:
    """Verify stage checkpoints round-trip without an Orbax dependency."""
    state = {
        "step": jnp.asarray(
            3,
            dtype=jnp.int32,
        ),
        "weights": jnp.arange(
            4,
            dtype=jnp.float32,
        ),
    }
    path = save_state_file(
        state=state,
        checkpoint_dir=tmp_path,
        name="stage",
    )
    restored = restore_state_file(
        state={
            "step": jnp.asarray(
                0,
                dtype=jnp.int32,
            ),
            "weights": jnp.zeros(
                (4,),
                dtype=jnp.float32,
            ),
        },
        checkpoint_path=path,
    )

    assert path.name == "stage.msgpack"
    np.testing.assert_array_equal(
        np.asarray(
            restored["weights"],
        ),
        np.arange(
            4,
            dtype=np.float32,
        ),
    )
    assert int(
        restored["step"],
    ) == 3


def test_merge_matching_leaves_loads_only_matching_shapes() -> None:
    """Verify partial checkpoint loading keeps incompatible leaves unchanged."""
    target = freeze(
        {
            "backbone": {
                "kernel": jnp.zeros(
                    (
                        3,
                        3,
                    )
                ),
            },
            "head": {
                "kernel": jnp.zeros(
                    (
                        4,
                        10,
                    )
                ),
            },
        }
    )
    source = freeze(
        {
            "backbone": {
                "kernel": jnp.ones(
                    (
                        3,
                        3,
                    )
                ),
            },
            "head": {
                "kernel": jnp.ones(
                    (
                        4,
                        100,
                    )
                ),
            },
        }
    )

    merged, loaded, skipped = _merge_matching_leaves(
        target=target,
        source=source,
        prefix="params",
    )

    np.testing.assert_array_equal(
        np.asarray(
            merged["backbone"]["kernel"],
        ),
        np.ones(
            (
                3,
                3,
            )
        ),
    )
    np.testing.assert_array_equal(
        np.asarray(
            merged["head"]["kernel"],
        ),
        np.zeros(
            (
                4,
                10,
            )
        ),
    )

    assert "params/backbone/kernel" in loaded
    assert "params/head/kernel" in skipped
