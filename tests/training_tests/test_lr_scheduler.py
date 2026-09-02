from __future__ import annotations

import jax.numpy as jnp

from allthemix.training.utils.lr_scheduler import build_lr_schedule, build_step_lr_schedule


class TestLRScheduler:
    def test_step_lr_schedule(self) -> None:
        """Verify that step lr schedule."""
        schedule = build_step_lr_schedule(
            base_learning_rate=0.1,
            steps_per_epoch=10,
            decay_epochs=[2, 4],
            decay_rate=0.1,
        )

        assert jnp.allclose(
            schedule(0),
            0.1,
        )

        assert jnp.allclose(
            schedule(19),
            0.1,
        )

        assert jnp.allclose(
            schedule(20),
            0.01,
        )

        assert jnp.allclose(
            schedule(39),
            0.01,
        )

        assert jnp.allclose(
            schedule(40),
            0.001,
        )

    def test_cosine_lr_schedule_reaches_min_learning_rate(self) -> None:
        """Verify that cosine lr schedule reaches min learning rate."""
        schedule = build_lr_schedule(
            schedule_name="cosine",
            base_learning_rate=0.1,
            steps_per_epoch=10,
            epochs=20,
            decay_epochs=[100, 150],
            decay_rate=0.1,
            min_learning_rate=0.0,
        )

        assert jnp.allclose(
            schedule(0),
            0.1,
        )

        assert schedule(100) < 0.1
        assert schedule(100) > 0.0

        assert jnp.allclose(
            schedule(200),
            0.0,
            atol=1e-7,
        )

    def test_warmup_cosine_lr_schedule_warms_then_decays(self) -> None:
        """Verify that warmup cosine linearly warms up before cosine decay."""
        schedule = build_lr_schedule(
            schedule_name="warmup_cosine",
            base_learning_rate=0.1,
            steps_per_epoch=10,
            epochs=20,
            decay_epochs=[100, 150],
            decay_rate=0.1,
            min_learning_rate=0.0,
            warmup_epochs=5,
        )

        assert jnp.allclose(
            schedule(0),
            0.0,
        )

        assert jnp.allclose(
            schedule(25),
            0.05,
            atol=1e-7,
        )

        assert jnp.allclose(
            schedule(50),
            0.1,
            atol=1e-7,
        )

        assert schedule(100) < 0.1
        assert schedule(100) > 0.0

        assert jnp.allclose(
            schedule(200),
            0.0,
            atol=1e-7,
        )

    def test_warmup_cosine_rejects_full_length_warmup(self) -> None:
        """Verify that warmup cannot consume the whole training schedule."""
        try:
            build_lr_schedule(
                schedule_name="warmup_cosine",
                base_learning_rate=0.1,
                steps_per_epoch=10,
                epochs=5,
                decay_epochs=[100, 150],
                decay_rate=0.1,
                min_learning_rate=0.0,
                warmup_epochs=5,
            )
        except ValueError as exc:
            assert "warmup_steps must be smaller than total_steps" in str(exc)
        else:
            raise AssertionError("Expected ValueError for full-length warmup.")

    def test_step_cosine_lr_schedule_plateaus_then_decays(self) -> None:
        """Verify that step cosine keeps a plateau before cosine decay."""
        schedule = build_lr_schedule(
            schedule_name="step_cosine",
            base_learning_rate=0.1,
            steps_per_epoch=10,
            epochs=30,
            decay_epochs=[
                15,
            ],
            decay_rate=0.1,
            min_learning_rate=0.0001,
        )

        assert jnp.allclose(
            schedule(0),
            0.1,
        )

        assert jnp.allclose(
            schedule(149),
            0.1,
        )

        assert jnp.allclose(
            schedule(150),
            0.1,
        )

        assert schedule(225) < 0.1
        assert schedule(225) > 0.0001

        assert jnp.allclose(
            schedule(300),
            0.0001,
            atol=1e-7,
        )
