from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from allthemix.methods.catchupmix import (
    CatchupMixContext,
    catchup_mix_features,
    catchupmix,
    make_catchup_mix_feature_hook,
)
from allthemix.methods.selector import get_mixer


def test_catchup_mix_features_keeps_low_relative_influence_source_channels() -> None:
    """Verify that catchup mix features keeps low relative influence source channels."""
    features = jnp.asarray(
        [
            [
                [
                    [4.0, 3.0, 2.0, 1.0],
                ],
            ],
            [
                [
                    [1.0, 2.0, 3.0, 4.0],
                ],
            ],
        ],
        dtype=jnp.float32,
    )

    mixed = catchup_mix_features(
        features=features,
        lam=jnp.asarray(0.5, dtype=jnp.float32),
        perm=jnp.asarray([1, 0], dtype=jnp.int32),
    )

    expected = jnp.asarray(
        [
            [
                [
                    [1.0, 2.0, 2.0, 1.0],
                ],
            ],
            [
                [
                    [1.0, 2.0, 2.0, 1.0],
                ],
            ],
        ],
        dtype=jnp.float32,
    )

    np.testing.assert_allclose(
        np.asarray(mixed),
        np.asarray(expected),
        atol=1e-6,
        rtol=1e-6,
    )


def test_catchup_mix_features_accepts_per_sample_lam() -> None:
    """Verify that catchup mix features supports one lambda per sample."""
    features = jnp.asarray(
        [
            [
                [
                    [4.0, 3.0, 2.0, 1.0],
                ],
            ],
            [
                [
                    [1.0, 2.0, 3.0, 4.0],
                ],
            ],
        ],
        dtype=jnp.float32,
    )

    mixed = catchup_mix_features(
        features=features,
        lam=jnp.asarray([0.25, 0.75], dtype=jnp.float32),
        perm=jnp.asarray([1, 0], dtype=jnp.int32),
    )

    assert mixed.shape == features.shape


def test_catchupmix_samples_valid_layer_and_no_repeat_pairs() -> None:
    """Verify that catchupmix samples valid layer and no repeat pairs."""
    rng = jax.random.PRNGKey(0)
    images = jnp.ones((8, 32, 32, 3), dtype=jnp.float32)
    labels = jnp.arange(8, dtype=jnp.int32)

    output = catchupmix(
        rng=rng,
        images=images,
        labels=labels,
        num_classes=10,
        alpha=0.5,
        num_feature_layers=5,
        no_repeat=True,
    )

    assert output.images.shape == images.shape
    assert output.labels_a.shape == labels.shape
    assert output.labels_b.shape == labels.shape
    assert output.lam.shape == ()
    assert output.layer.shape == ()
    assert int(output.layer) >= 0
    assert int(output.layer) <= 5
    assert jnp.all(output.perm != jnp.arange(8))


def test_selector_exposes_catchupmix_arguments() -> None:
    """Verify that selector exposes catchupmix arguments."""
    rng = jax.random.PRNGKey(1)
    images = jnp.ones((8, 32, 32, 3), dtype=jnp.float32)
    labels = jnp.arange(8, dtype=jnp.int32)

    mixer = get_mixer(
        name="catch-up_mix",
        num_classes=10,
        catchupmix_alpha=0.5,
        catchupmix_num_layers=3,
        catchupmix_no_repeat=True,
    )

    output = mixer(
        rng=rng,
        images=images,
        labels=labels,
    )

    assert output.images.shape == images.shape
    assert int(output.layer) >= 0
    assert int(output.layer) <= 3


def test_selector_passes_cutmix_box_arguments_to_catchupmix() -> None:
    """Verify that CatchUpMix can reuse CutMix box/lambda options."""
    rng = jax.random.PRNGKey(2)
    images = jnp.ones((8, 32, 32, 3), dtype=jnp.float32)
    labels = jnp.arange(8, dtype=jnp.int32)

    mixer = get_mixer(
        name="catchupmix",
        num_classes=10,
        catchupmix_alpha=0.5,
        catchupmix_num_layers=3,
        cutmix_variant="torchbearer_area",
        cutmix_per_sample_lam=True,
        cutmix_min_lam=0.6,
    )

    output = mixer(
        rng=rng,
        images=images,
        labels=labels,
    )

    assert output.images.shape == images.shape
    assert output.lam.shape == (8,)


def test_catchup_mix_context_is_constructible() -> None:
    """Verify that catchup mix context is constructible."""
    context = CatchupMixContext(
        layer=jnp.asarray(1, dtype=jnp.int32),
        lam=jnp.asarray(0.5, dtype=jnp.float32),
        perm=jnp.asarray([1, 0], dtype=jnp.int32),
    )

    assert int(context.layer) == 1


def test_catchup_mix_feature_hook_only_applies_selected_layer() -> None:
    """Verify that catchup mix feature hook only applies selected layer."""
    features = jnp.asarray(
        [
            [
                [
                    [4.0, 3.0, 2.0, 1.0],
                ],
            ],
            [
                [
                    [1.0, 2.0, 3.0, 4.0],
                ],
            ],
        ],
        dtype=jnp.float32,
    )

    context = CatchupMixContext(
        layer=jnp.asarray(2, dtype=jnp.int32),
        lam=jnp.asarray(0.5, dtype=jnp.float32),
        perm=jnp.asarray([1, 0], dtype=jnp.int32),
    )

    feature_hook = make_catchup_mix_feature_hook(
        context,
    )

    np.testing.assert_allclose(
        np.asarray(
            feature_hook(
                features,
                1,
            )
        ),
        np.asarray(features),
        atol=1e-6,
        rtol=1e-6,
    )

    expected = jnp.asarray(
        [
            [
                [
                    [1.0, 2.0, 2.0, 1.0],
                ],
            ],
            [
                [
                    [1.0, 2.0, 2.0, 1.0],
                ],
            ],
        ],
        dtype=jnp.float32,
    )

    np.testing.assert_allclose(
        np.asarray(
            feature_hook(
                features,
                2,
            )
        ),
        np.asarray(expected),
        atol=1e-6,
        rtol=1e-6,
    )
