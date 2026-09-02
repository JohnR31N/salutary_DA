"""Every registered mixer honors the MixOutput contract.

This is the lock behind ``allthemix.methods.output.MixOutput``: one output
container, all fields named, ``perm`` always a real permutation. New methods
must pass this test by construction.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from allthemix.methods.output import MixOutput
from allthemix.methods.selector import get_mixer

BATCH = 8
HEIGHT = WIDTH = 8
CLASSES = 10

SIMPLE_METHODS = (
    "baseline",
    "mixup",
    "cutmix",
    "fmix",
    "resizemix",
    "guided_sr",
)


def _batch():
    rng = np.random.default_rng(0)
    images = jnp.asarray(
        rng.normal(size=(BATCH, HEIGHT, WIDTH, 3)).astype(np.float32)
    )
    labels = jnp.asarray(np.arange(BATCH, dtype=np.int32) % CLASSES)
    return images, labels


def _assert_contract(output, labels):
    assert isinstance(output, MixOutput)
    images, _ = _batch()
    assert output.images.shape == images.shape
    assert output.labels_a.shape == labels.shape
    assert output.labels_b.shape == labels.shape
    perm = np.asarray(output.perm)
    assert perm.shape == (BATCH,)
    np.testing.assert_array_equal(np.sort(perm), np.arange(BATCH))


@pytest.mark.parametrize("method", SIMPLE_METHODS)
def test_registered_mixers_return_mix_output(method: str) -> None:
    images, labels = _batch()
    mixer = get_mixer(
        name=method,
        num_classes=CLASSES,
    )
    output = mixer(
        jax.random.PRNGKey(0),
        images,
        labels,
        None,
    )
    _assert_contract(output, labels)


def test_saliencymix_returns_mix_output_with_aux_maps() -> None:
    images, labels = _batch()
    mixer = get_mixer(
        name="saliencymix",
        num_classes=CLASSES,
    )
    saliency = jnp.asarray(
        np.random.default_rng(1)
        .random((BATCH, HEIGHT, WIDTH))
        .astype(np.float32)
    )
    output = mixer(
        jax.random.PRNGKey(0),
        images,
        labels,
        {"saliency_maps": saliency},
    )
    _assert_contract(output, labels)


def test_baseline_is_identity_in_the_shared_container() -> None:
    images, labels = _batch()
    mixer = get_mixer(
        name="baseline",
        num_classes=CLASSES,
    )
    output = mixer(jax.random.PRNGKey(0), images, labels, None)
    np.testing.assert_array_equal(np.asarray(output.images), np.asarray(images))
    np.testing.assert_array_equal(
        np.asarray(output.labels_a), np.asarray(output.labels_b)
    )
    np.testing.assert_array_equal(
        np.asarray(output.perm), np.arange(BATCH)
    )
