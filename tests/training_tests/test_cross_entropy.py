from __future__ import annotations

import jax.numpy as jnp
import pytest

from allthemix.training.losses.cross_entropy import (
    cross_entropy,
    hard_cross_entropy,
    soft_cross_entropy,
)
from allthemix.training.losses.selector import get_criterion


class TestCrossEntropy:
    def test_hard_cross_entropy_returns_scalar(self) -> None:
        """Verify that hard cross entropy returns scalar."""
        logits = jnp.array(
            [
                [2.0, 1.0, 0.1],
                [0.1, 1.0, 2.0],
            ]
        )

        labels = jnp.array([0, 2])

        loss = hard_cross_entropy(
            logits=logits,
            labels=labels,
            num_classes=3,
        )

        assert loss.shape == ()
        assert loss > 0

    def test_soft_cross_entropy_returns_scalar(self) -> None:
        """Verify that soft cross entropy returns scalar."""
        logits = jnp.array(
            [
                [2.0, 1.0, 0.1],
                [0.1, 1.0, 2.0],
            ]
        )

        labels = jnp.array(
            [
                [0.7, 0.3, 0.0],
                [0.0, 0.2, 0.8],
            ]
        )

        loss = soft_cross_entropy(
            logits=logits,
            labels=labels,
        )

        assert loss.shape == ()
        assert loss > 0

    def test_cross_entropy_accepts_hard_labels(self) -> None:
        """Verify that cross entropy accepts hard labels."""
        logits = jnp.array(
            [
                [2.0, 1.0, 0.1],
                [0.1, 1.0, 2.0],
            ]
        )

        labels = jnp.array([0, 2])

        loss = cross_entropy(
            logits=logits,
            labels=labels,
            num_classes=3,
        )

        assert loss.shape == ()
        assert loss > 0

    def test_cross_entropy_accepts_soft_labels(self) -> None:
        """Verify that cross entropy accepts soft labels."""
        logits = jnp.array(
            [
                [2.0, 1.0, 0.1],
                [0.1, 1.0, 2.0],
            ]
        )

        labels = jnp.array(
            [
                [0.7, 0.3, 0.0],
                [0.0, 0.2, 0.8],
            ]
        )

        loss = cross_entropy(
            logits=logits,
            labels=labels,
            num_classes=3,
        )

        assert loss.shape == ()
        assert loss > 0

    def test_invalid_label_shape_raises_error(self) -> None:
        """Verify that invalid label shape raises error."""
        logits = jnp.ones((2, 3))
        labels = jnp.ones((2, 3, 1))

        with pytest.raises(ValueError):
            cross_entropy(
                logits=logits,
                labels=labels,
                num_classes=3,
            )

    def test_get_cross_entropy_criterion(self) -> None:
        """Verify that get cross entropy criterion."""
        criterion = get_criterion("cross_entropy")

        assert criterion is not None

    def test_get_ce_criterion(self) -> None:
        """Verify that get ce criterion."""
        criterion = get_criterion("ce")

        assert criterion is not None

    def test_unknown_criterion_raises_error(self) -> None:
        """Verify that unknown criterion raises error."""
        with pytest.raises(ValueError):
            get_criterion("unknown")