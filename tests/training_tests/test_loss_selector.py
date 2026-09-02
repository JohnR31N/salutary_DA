from __future__ import annotations

import jax.numpy as jnp
import pytest

from allthemix.methods.output import MixOutput
from allthemix.training.losses.loss_selector import compute_train_loss_and_targets


class TestLossSelector:
    def test_baseline_loss_selector(self) -> None:
        """Verify that baseline loss selector."""
        logits = jnp.array(
            [
                [2.0, 1.0, 0.1],
                [0.1, 1.0, 2.0],
            ]
        )

        mixed_images = jnp.ones((2, 32, 32, 3))
        labels = jnp.array([0, 2])

        loss, target_labels = compute_train_loss_and_targets(
            method="baseline",
            logits=logits,
            mixer_output=MixOutput(
                images=mixed_images,
                labels_a=labels,
                labels_b=labels,
                lam=jnp.ones(()),
                perm=jnp.arange(labels.shape[0]),
            ),
            num_classes=3,
        )

        assert loss.shape == ()
        assert loss > 0

        assert jnp.array_equal(
            target_labels,
            labels,
        )

    def test_mixup_loss_selector(self) -> None:
        """Verify that mixup loss selector."""
        logits = jnp.array(
            [
                [2.0, 1.0, 0.1],
                [0.1, 1.0, 2.0],
            ]
        )

        mixed_images = jnp.ones((2, 32, 32, 3))
        labels_a = jnp.array([0, 2])
        labels_b = jnp.array([1, 1])
        lam = jnp.array(0.7)

        loss, target_labels = compute_train_loss_and_targets(
            method="mixup",
            logits=logits,
            mixer_output=MixOutput(
                images=mixed_images,
                labels_a=labels_a,
                labels_b=labels_b,
                lam=lam,
                perm=jnp.arange(labels_a.shape[0]),
            ),
            num_classes=3,
        )

        assert loss.shape == ()
        assert loss > 0

        assert jnp.array_equal(
            target_labels,
            labels_a,
        )

    def test_guided_sr_loss_selector(self) -> None:
        """Verify that guided sr uses mix-style targets."""
        logits = jnp.array(
            [
                [2.0, 1.0, 0.1],
                [0.1, 1.0, 2.0],
            ]
        )

        mixed_images = jnp.ones((2, 32, 32, 3))
        labels_a = jnp.array([0, 2])
        labels_b = jnp.array([1, 1])
        lam = jnp.array([0.7, 0.3])

        loss, target_labels = compute_train_loss_and_targets(
            method="guided_sr",
            logits=logits,
            mixer_output=MixOutput(
                images=mixed_images,
                labels_a=labels_a,
                labels_b=labels_b,
                lam=lam,
                perm=jnp.arange(labels_a.shape[0]),
            ),
            num_classes=3,
        )

        assert loss.shape == ()
        assert loss > 0

        assert jnp.array_equal(
            target_labels,
            labels_a,
        )

    def test_unknown_method_raises_error(self) -> None:
        """Verify that unknown method raises error."""
        logits = jnp.ones((2, 3))
        mixed_images = jnp.ones((2, 32, 32, 3))
        labels = jnp.array([0, 1])

        with pytest.raises(ValueError):
            compute_train_loss_and_targets(
                method="unknown",
                logits=logits,
                mixer_output=MixOutput(
                images=mixed_images,
                labels_a=labels,
                labels_b=labels,
                lam=jnp.ones(()),
                perm=jnp.arange(labels.shape[0]),
            ),
                num_classes=3,
            )
