from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from allthemix.networks.builder import build_model
from allthemix.training.engine.single.eval import eval_step
from allthemix.training.engine.single.loop import evaluate
from allthemix.training.engine.single.train import create_train_state


class TestEval:
    def test_eval_step_returns_loss_and_metrics(self) -> None:
        """Verify that eval step returns loss and metrics."""
        rng = jax.random.PRNGKey(0)

        model = build_model(
            name="simple_cnn",
            num_classes=10,
        )

        state = create_train_state(
            rng=rng,
            model=model,
            learning_rate=0.1,
            momentum=0.9,
            weight_decay=5e-4,
            input_shape=(4, 32, 32, 3),
        )

        images = jnp.ones((4, 32, 32, 3))
        labels = jnp.array([0, 1, 2, 3])

        loss, top1_acc, top5_acc, top1_error, top5_error = eval_step(
            state=state,
            images=images,
            labels=labels,
            num_classes=10,
        )

        assert loss.shape == ()
        assert loss > 0

        assert top1_acc.shape == ()
        assert top1_acc >= 0.0
        assert top1_acc <= 1.0

        assert top5_acc.shape == ()
        assert top5_acc >= 0.0
        assert top5_acc <= 1.0

        assert top1_error.shape == ()
        assert top1_error >= 0.0
        assert top1_error <= 1.0

        assert top5_error.shape == ()
        assert top5_error >= 0.0
        assert top5_error <= 1.0

        assert jnp.allclose(
            top1_error,
            1.0 - top1_acc,
        )

        assert jnp.allclose(
            top5_error,
            1.0 - top5_acc,
        )

    def test_evaluate_weights_partial_final_batch_by_sample_count(
        self,
        monkeypatch,
    ) -> None:
        """Verify a partial final batch is not weighted as a full batch."""

        def fake_eval_step(
            state,
            images,
            labels,
            num_classes,
        ):
            """Return known batch means for two differently sized batches."""
            del state, images, num_classes

            if labels.shape[0] == 4:
                return (
                    jnp.asarray(1.0),
                    jnp.asarray(0.25),
                    jnp.asarray(0.5),
                    jnp.asarray(0.75),
                    jnp.asarray(0.5),
                )

            return (
                jnp.asarray(3.0),
                jnp.asarray(1.0),
                jnp.asarray(1.0),
                jnp.asarray(0.0),
                jnp.asarray(0.0),
            )

        monkeypatch.setattr(
            "allthemix.training.engine.single.loop.eval_step",
            fake_eval_step,
        )
        evaluation_batches = [
            (
                np.zeros(
                    (
                        4,
                        2,
                        2,
                        1,
                    ),
                    dtype=np.float32,
                ),
                np.zeros(
                    (4,),
                    dtype=np.int64,
                ),
            ),
            (
                np.zeros(
                    (
                        1,
                        2,
                        2,
                        1,
                    ),
                    dtype=np.float32,
                ),
                np.zeros(
                    (1,),
                    dtype=np.int64,
                ),
            ),
        ]

        metrics = evaluate(
            state=None,
            test_ds=evaluation_batches,
            num_classes=2,
            max_eval_steps=-1,
            return_counts=True,
        )

        np.testing.assert_allclose(
            metrics[:5],
            (
                1.4,
                0.4,
                0.6,
                0.6,
                0.4,
            ),
            atol=1e-7,
        )
        assert metrics[5:] == (2, 5)
