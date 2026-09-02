from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from allthemix.methods.selector import get_mixer
from allthemix.networks.builder import build_model
from allthemix.training.engine.single.train import create_train_state, train_step
from allthemix.training.utils.lr_scheduler import build_step_lr_schedule


class TestTrain:
    def _tree_l2_delta(self, before, after) -> float:
        """Support tree l2 delta."""
        leaves_before = jax.tree_util.tree_leaves(
            before,
        )

        leaves_after = jax.tree_util.tree_leaves(
            after,
        )

        total = 0.0

        for before_leaf, after_leaf in zip(
            leaves_before,
            leaves_after,
        ):
            diff = np.asarray(
                after_leaf - before_leaf,
            )

            total += float(
                np.sum(
                    diff * diff,
                )
            )

        return total ** 0.5

    def test_train_step_with_baseline(self) -> None:
        """Verify that train step with baseline."""
        rng = jax.random.PRNGKey(0)
        rng_init, rng_train = jax.random.split(rng)

        model = build_model(
            name="simple_cnn",
            num_classes=10,
        )

        state = create_train_state(
            rng=rng_init,
            model=model,
            learning_rate=0.1,
            momentum=0.9,
            weight_decay=5e-4,
            input_shape=(4, 32, 32, 3),
        )

        images = jnp.ones((4, 32, 32, 3))
        labels = jnp.array([0, 1, 2, 3])

        mixer = get_mixer(
            name="baseline",
            num_classes=10,
            alpha=1.0,
        )

        new_state, loss, accuracy = train_step(
            state=state,
            rng=rng_train,
            images=images,
            labels=labels,
            mixer_fn=mixer,
            method="baseline",
            num_classes=10,
        )

        assert loss.shape == ()
        assert loss > 0

        assert accuracy.shape == ()
        assert accuracy >= 0.0
        assert accuracy <= 1.0

        assert new_state.step == state.step + 1

    def test_train_step_with_mixup(self) -> None:
        """Verify that train step with mixup."""
        rng = jax.random.PRNGKey(0)
        rng_init, rng_train = jax.random.split(rng)

        model = build_model(
            name="simple_cnn",
            num_classes=10,
        )

        state = create_train_state(
            rng=rng_init,
            model=model,
            learning_rate=0.1,
            momentum=0.9,
            weight_decay=5e-4,
            input_shape=(4, 32, 32, 3),
        )

        images = jnp.ones((4, 32, 32, 3))
        labels = jnp.array([0, 1, 2, 3])

        mixer = get_mixer(
            name="mixup",
            num_classes=10,
            alpha=1.0,
        )

        new_state, loss, accuracy = train_step(
            state=state,
            rng=rng_train,
            images=images,
            labels=labels,
            mixer_fn=mixer,
            method="mixup",
            num_classes=10,
        )

        assert loss.shape == ()
        assert loss > 0

        assert accuracy.shape == ()
        assert accuracy >= 0.0
        assert accuracy <= 1.0

        assert new_state.step == state.step + 1

    def test_train_step_with_cutmix_sumix_updates_uncertainty_head(self) -> None:
        """Verify that train step with cutmix sumix updates uncertainty head."""
        rng = jax.random.PRNGKey(0)
        rng_init, rng_train = jax.random.split(rng)

        model = build_model(
            name="simple_cnn",
            num_classes=10,
        )

        state = create_train_state(
            rng=rng_init,
            model=model,
            learning_rate=0.1,
            momentum=0.9,
            weight_decay=5e-4,
            input_shape=(4, 32, 32, 3),
        )

        images = jnp.arange(
            4 * 32 * 32 * 3,
            dtype=jnp.float32,
        ).reshape(
            4,
            32,
            32,
            3,
        )

        images = images / jnp.max(
            images,
        )

        labels = jnp.array([0, 1, 2, 3])

        mixer = get_mixer(
            name="cutmix_sumix",
            num_classes=10,
            cutmix_alpha=0.2,
            cutmix_prob=1.0,
        )

        new_state, loss, accuracy = train_step(
            state=state,
            rng=rng_train,
            images=images,
            labels=labels,
            mixer_fn=mixer,
            method="cutmix_sumix",
            num_classes=10,
            sumix_gamma=0.5,
        )

        assert loss.shape == ()
        assert loss > 0

        assert accuracy.shape == ()
        assert accuracy >= 0.0
        assert accuracy <= 1.0

        assert new_state.step == state.step + 1

        assert self._tree_l2_delta(
            state.params,
            new_state.params,
        ) > 0.0

        assert self._tree_l2_delta(
            state.sumix_params,
            new_state.sumix_params,
        ) > 0.0

    def test_train_step_with_cutmix_sumix_can_return_debug_metrics(self) -> None:
        """Verify that cutmix sumix can return diagnostic metrics."""
        rng = jax.random.PRNGKey(3)
        rng_init, rng_train = jax.random.split(rng)

        model = build_model(
            name="simple_cnn",
            num_classes=10,
        )

        state = create_train_state(
            rng=rng_init,
            model=model,
            learning_rate=0.1,
            momentum=0.9,
            weight_decay=5e-4,
            input_shape=(4, 32, 32, 3),
        )

        images = jnp.ones((4, 32, 32, 3))
        labels = jnp.array([0, 1, 2, 3])

        mixer = get_mixer(
            name="cutmix_sumix",
            num_classes=10,
            cutmix_alpha=0.2,
            cutmix_prob=1.0,
        )

        new_state, loss, accuracy, metrics = train_step(
            state=state,
            rng=rng_train,
            images=images,
            labels=labels,
            mixer_fn=mixer,
            method="cutmix_sumix",
            num_classes=10,
            sumix_gamma=0.5,
            return_sumix_metrics=True,
        )

        del new_state
        del loss
        del accuracy

        assert "lam_a_mean" in metrics
        assert "lam_a_min" in metrics
        assert "lam_a_max" in metrics
        assert "classification_loss" in metrics
        assert "regularization_loss" in metrics

    def test_train_step_with_catchupmix_updates_model(self) -> None:
        """Verify that train step with catchupmix updates model."""
        rng = jax.random.PRNGKey(7)
        rng_init, rng_train = jax.random.split(rng)

        model = build_model(
            name="simple_cnn",
            num_classes=10,
        )

        state = create_train_state(
            rng=rng_init,
            model=model,
            learning_rate=0.1,
            momentum=0.9,
            weight_decay=5e-4,
            input_shape=(4, 32, 32, 3),
        )

        images = jnp.arange(
            4 * 32 * 32 * 3,
            dtype=jnp.float32,
        ).reshape(
            4,
            32,
            32,
            3,
        )

        images = images / jnp.max(
            images,
        )

        labels = jnp.array([0, 1, 2, 3])

        mixer = get_mixer(
            name="catchupmix",
            num_classes=10,
            catchupmix_alpha=0.5,
            catchupmix_num_layers=2,
            catchupmix_no_repeat=True,
        )

        new_state, loss, accuracy = train_step(
            state=state,
            rng=rng_train,
            images=images,
            labels=labels,
            mixer_fn=mixer,
            method="catchupmix",
            num_classes=10,
        )

        assert loss.shape == ()
        assert loss > 0

        assert accuracy.shape == ()
        assert accuracy >= 0.0
        assert accuracy <= 1.0

        assert new_state.step == state.step + 1

        assert self._tree_l2_delta(
            state.params,
            new_state.params,
        ) > 0.0

    def test_train_step_with_fmix_per_sample_updates_model(self) -> None:
        """Verify that train step supports per-sample FMix lambda."""
        rng = jax.random.PRNGKey(9)
        rng_init, rng_train = jax.random.split(rng)

        model = build_model(
            name="simple_cnn",
            num_classes=10,
        )

        state = create_train_state(
            rng=rng_init,
            model=model,
            learning_rate=0.1,
            momentum=0.9,
            weight_decay=5e-4,
            input_shape=(4, 32, 32, 3),
        )

        images = jnp.arange(
            4 * 32 * 32 * 3,
            dtype=jnp.float32,
        ).reshape(
            4,
            32,
            32,
            3,
        )

        images = images / jnp.max(
            images,
        )

        labels = jnp.array([0, 1, 2, 3])

        mixer = get_mixer(
            name="fmix",
            num_classes=10,
            fmix_alpha=1.0,
            fmix_decay=3.0,
            fmix_prob=1.0,
            fmix_per_sample=True,
        )

        new_state, loss, accuracy = train_step(
            state=state,
            rng=rng_train,
            images=images,
            labels=labels,
            mixer_fn=mixer,
            method="fmix",
            num_classes=10,
        )

        assert loss.shape == ()
        assert loss > 0

        assert accuracy.shape == ()
        assert accuracy >= 0.0
        assert accuracy <= 1.0

        assert new_state.step == state.step + 1

        assert self._tree_l2_delta(
            state.params,
            new_state.params,
        ) > 0.0

    def test_train_step_with_fmix_can_return_debug_metrics(self) -> None:
        """Verify that FMix tuple outputs can feed diagnostic metrics."""
        rng = jax.random.PRNGKey(11)
        rng_init, rng_train = jax.random.split(rng)

        model = build_model(
            name="simple_cnn",
            num_classes=10,
        )

        state = create_train_state(
            rng=rng_init,
            model=model,
            learning_rate=0.1,
            momentum=0.9,
            weight_decay=5e-4,
            input_shape=(4, 32, 32, 3),
        )

        images = jnp.arange(
            4 * 32 * 32 * 3,
            dtype=jnp.float32,
        ).reshape(
            4,
            32,
            32,
            3,
        )

        images = images / jnp.max(
            images,
        )

        labels = jnp.array([0, 1, 2, 3])

        mixer = get_mixer(
            name="fmix",
            num_classes=10,
            fmix_alpha=1.0,
            fmix_decay=3.0,
            fmix_prob=1.0,
            fmix_per_sample=True,
        )

        _, _, _, metrics = train_step(
            state=state,
            rng=rng_train,
            images=images,
            labels=labels,
            mixer_fn=mixer,
            method="fmix",
            num_classes=10,
            return_mix_metrics=True,
        )

        assert "mix_lam_mean" in metrics
        assert "mix_lam_min" in metrics
        assert "mix_lam_max" in metrics
        assert "mix_changed_ratio" in metrics
        assert metrics["mix_apply_rate"] >= 0.0

    def test_create_train_state_accepts_lr_schedule(self) -> None:
        """Verify that create train state accepts lr schedule."""
        rng = jax.random.PRNGKey(0)

        model = build_model(
            name="simple_cnn",
            num_classes=10,
        )

        lr_schedule = build_step_lr_schedule(
            base_learning_rate=0.1,
            steps_per_epoch=10,
            decay_epochs=[2, 4],
            decay_rate=0.1,
        )

        state = create_train_state(
            rng=rng,
            model=model,
            learning_rate=lr_schedule,
            momentum=0.9,
            weight_decay=5e-4,
            input_shape=(4, 32, 32, 3),
        )

        assert state.step == 0
