from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from allthemix.methods.baseline import baseline_mixer
from allthemix.methods.cutmix import cutmix
from allthemix.methods.mixup import mixup
from allthemix.methods.selector import get_mixer


class TestMixup:
    def test_baseline_mixer_keeps_image_shape(self) -> None:
        """Verify that baseline mixer keeps image shape."""
        rng = jax.random.PRNGKey(0)

        images = jnp.ones((4, 32, 32, 3))
        labels = jnp.array([0, 1, 2, 3])

        baseline_output = baseline_mixer(
            rng=rng,
            images=images,
            labels=labels,
            num_classes=10,
        )

        assert baseline_output.images.shape == (4, 32, 32, 3)
        assert baseline_output.labels_a.shape == (4,)

    def test_baseline_mixer_keeps_hard_labels(self) -> None:
        """Verify that baseline mixer keeps hard labels."""
        rng = jax.random.PRNGKey(0)

        images = jnp.ones((4, 32, 32, 3))
        labels = jnp.array([0, 1, 2, 3])

        baseline_output = baseline_mixer(
            rng=rng,
            images=images,
            labels=labels,
            num_classes=10,
        )

        assert jnp.array_equal(
            baseline_output.labels_a,
            labels,
        )

    def test_mixup_keeps_image_shape(self) -> None:
        """Verify that mixup keeps image shape."""
        rng = jax.random.PRNGKey(0)

        images = jnp.ones((4, 32, 32, 3))
        labels = jnp.array([0, 1, 2, 3])

        mixed_images, labels_a, labels_b, lam, _perm = mixup(
            rng=rng,
            images=images,
            labels=labels,
            num_classes=10,
            alpha=1.0,
        )

        assert mixed_images.shape == (4, 32, 32, 3)
        assert labels_a.shape == (4,)
        assert labels_b.shape == (4,)
        assert lam.shape == ()

    def test_mixup_returns_original_and_shuffled_labels(self) -> None:
        """Verify that mixup returns original and shuffled labels."""
        rng = jax.random.PRNGKey(0)

        images = jnp.ones((4, 32, 32, 3))
        labels = jnp.array([0, 1, 2, 3])

        _, labels_a, labels_b, _, _perm = mixup(
            rng=rng,
            images=images,
            labels=labels,
            num_classes=10,
            alpha=1.0,
        )

        assert jnp.array_equal(
            labels_a,
            labels,
        )

        assert labels_b.shape == labels.shape

    def test_mixup_lambda_is_valid(self) -> None:
        """Verify that mixup lambda is valid."""
        rng = jax.random.PRNGKey(0)

        images = jnp.ones((4, 32, 32, 3))
        labels = jnp.array([0, 1, 2, 3])

        _, _, _, lam, _perm = mixup(
            rng=rng,
            images=images,
            labels=labels,
            num_classes=10,
            alpha=1.0,
        )

        assert lam >= 0.0
        assert lam <= 1.0

    def test_mixup_can_use_external_partner_batch(self) -> None:
        """Verify that MixUp can use partners supplied by distributed training."""
        rng = jax.random.PRNGKey(0)

        images = jnp.arange(
            4 * 4 * 4 * 1,
            dtype=jnp.float32,
        ).reshape(
            4,
            4,
            4,
            1,
        )
        labels = jnp.array([0, 1, 2, 3])
        paired_images = images[::-1]
        paired_labels = labels[::-1]

        _, labels_a, labels_b, _, _perm = mixup(
            rng=rng,
            images=images,
            labels=labels,
            num_classes=10,
            alpha=1.0,
            paired_images=paired_images,
            paired_labels=paired_labels,
        )

        assert jnp.array_equal(
            labels_a,
            labels,
        )
        assert jnp.array_equal(
            labels_b,
            paired_labels,
        )

    def test_cutmix_no_repeat_avoids_self_pairs(self) -> None:
        """Verify that cutmix no repeat avoids self pairs."""
        rng = jax.random.PRNGKey(0)

        images = jnp.ones((8, 32, 32, 3))
        labels = jnp.arange(8)

        output = cutmix(
            rng=rng,
            images=images,
            labels=labels,
            num_classes=10,
            alpha=1.0,
            prob=1.0,
            no_repeat=True,
        )

        assert jnp.all(
            output.perm != jnp.arange(8),
        )

    def test_cutmix_can_use_external_partner_batch(self) -> None:
        """Verify that CutMix can use partners supplied by distributed training."""
        rng = jax.random.PRNGKey(0)

        images = jnp.ones((4, 32, 32, 3))
        labels = jnp.array([0, 1, 2, 3])
        paired_images = images[::-1]
        paired_labels = labels[::-1]

        output = cutmix(
            rng=rng,
            images=images,
            labels=labels,
            num_classes=10,
            alpha=1.0,
            prob=1.0,
            paired_images=paired_images,
            paired_labels=paired_labels,
            paired_perm=jnp.array([3, 2, 1, 0]),
        )

        assert jnp.array_equal(
            output.labels_a,
            labels,
        )
        assert jnp.array_equal(
            output.labels_b,
            paired_labels,
        )
        assert jnp.array_equal(
            output.perm,
            jnp.array([3, 2, 1, 0]),
        )

    def test_cutmix_torchbearer_variant_keeps_sampled_lambda(self) -> None:
        """Verify Torchbearer-style CutMix uses sampled lambda."""
        rng = jax.random.PRNGKey(2)

        images = jnp.zeros((4, 16, 16, 1))
        labels = jnp.array([0, 1, 2, 3])
        paired_images = jnp.ones_like(images)
        paired_labels = jnp.array([4, 5, 6, 7])

        output = cutmix(
            rng=rng,
            images=images,
            labels=labels,
            num_classes=10,
            alpha=1.0,
            prob=1.0,
            paired_images=paired_images,
            paired_labels=paired_labels,
            variant="torchbearer",
        )

        changed_mask = output.images[..., 0] > 0.5
        changed_ratios = jnp.mean(
            changed_mask,
            axis=(
                1,
                2,
            ),
        )

        assert output.lam.shape == ()
        assert jnp.all(
            output.labels_a == labels,
        )
        assert jnp.all(
            output.labels_b == paired_labels,
        )
        assert jnp.all(
            changed_ratios > 0.0,
        )
        assert jnp.any(
            jnp.abs(
                (1.0 - changed_ratios)
                - output.lam
            )
            > 1e-6
        )

    def test_cutmix_torchbearer_area_recomputes_clipped_lambda(self) -> None:
        """Verify Torchbearer-area CutMix weights labels by actual clipped area."""
        rng = jax.random.PRNGKey(2)

        images = jnp.zeros((4, 16, 16, 1))
        labels = jnp.array([0, 1, 2, 3])
        paired_images = jnp.ones_like(images)
        paired_labels = jnp.array([4, 5, 6, 7])

        output = cutmix(
            rng=rng,
            images=images,
            labels=labels,
            num_classes=10,
            alpha=1.0,
            prob=1.0,
            paired_images=paired_images,
            paired_labels=paired_labels,
            variant="torchbearer_area",
        )

        changed_mask = output.images[..., 0] > 0.5
        changed_ratios = jnp.mean(
            changed_mask,
            axis=(
                1,
                2,
            ),
        )

        assert output.lam.shape == (4,)
        assert jnp.allclose(
            output.lam,
            1.0 - changed_ratios,
        )

    def test_selector_cutmix_sumix_uses_no_repeat_pairs(self) -> None:
        """Verify that selector cutmix sumix uses no repeat pairs."""
        rng = jax.random.PRNGKey(0)

        images = jnp.ones((8, 32, 32, 3))
        labels = jnp.arange(8)

        mixer = get_mixer(
            name="cutmix_sumix",
            num_classes=10,
            cutmix_alpha=1.0,
            cutmix_prob=1.0,
        )

        output = mixer(
            rng=rng,
            images=images,
            labels=labels,
        )

        assert jnp.all(
            output.perm != jnp.arange(8),
        )

    def test_selector_can_use_torchbearer_cutmix_variant(self) -> None:
        """Verify selector passes CutMix variant to the implementation."""
        rng = jax.random.PRNGKey(2)

        images = jnp.zeros((4, 16, 16, 1))
        labels = jnp.array([0, 1, 2, 3])

        mixer = get_mixer(
            name="cutmix",
            num_classes=10,
            cutmix_alpha=1.0,
            cutmix_prob=1.0,
            cutmix_variant="torchbearer",
        )

        output = mixer(
            rng=rng,
            images=images,
            labels=labels,
        )

        assert output.lam.shape == ()
        assert output.images.shape == images.shape

    def test_cutmix_can_use_per_sample_lam_and_min_lam(self) -> None:
        """Verify CutMix can sample one bounded lambda per sample."""
        rng = jax.random.PRNGKey(4)

        images = jnp.zeros((8, 32, 32, 1))
        labels = jnp.arange(8)
        paired_images = jnp.ones_like(images)
        paired_labels = labels[::-1]

        output = cutmix(
            rng=rng,
            images=images,
            labels=labels,
            num_classes=10,
            alpha=1.0,
            prob=1.0,
            paired_images=paired_images,
            paired_labels=paired_labels,
            variant="torchbearer",
            per_sample_lam=True,
            min_lam=0.7,
        )

        changed_mask = output.images[..., 0] > 0.5
        changed_ratios = jnp.mean(
            changed_mask,
            axis=(
                1,
                2,
            ),
        )

        assert output.lam.shape == (8,)
        assert jnp.all(output.lam >= 0.7)
        assert jnp.all(changed_ratios <= 0.3 + 1e-6)
        assert jnp.std(changed_ratios) > 0.0

    def test_cutmix_rejects_per_sample_lam_without_torchbearer(self) -> None:
        """Verify per-sample lambda cannot be paired with a batch-level box path."""
        rng = jax.random.PRNGKey(4)

        images = jnp.zeros((4, 32, 32, 1))
        labels = jnp.arange(4)

        with pytest.raises(
            ValueError,
            match="cutmix_per_sample_lam",
        ):
            cutmix(
                rng=rng,
                images=images,
                labels=labels,
                num_classes=10,
                alpha=1.0,
                prob=1.0,
                variant="standard",
                per_sample_lam=True,
            )

    def test_get_baseline_mixer(self) -> None:
        """Verify that get baseline mixer."""
        mixer = get_mixer(
            name="baseline",
            num_classes=10,
            alpha=1.0,
        )

        rng = jax.random.PRNGKey(0)
        images = jnp.ones((4, 32, 32, 3))
        labels = jnp.array([0, 1, 2, 3])

        baseline_output = mixer(
            rng=rng,
            images=images,
            labels=labels,
        )

        assert baseline_output.images.shape == (4, 32, 32, 3)
        assert baseline_output.labels_a.shape == (4,)

    def test_get_mixup_mixer(self) -> None:
        """Verify that get mixup mixer."""
        mixer = get_mixer(
            name="mixup",
            num_classes=10,
            alpha=1.0,
        )

        rng = jax.random.PRNGKey(0)
        images = jnp.ones((4, 32, 32, 3))
        labels = jnp.array([0, 1, 2, 3])

        mixed_images, labels_a, labels_b, lam, _perm = mixer(
            rng=rng,
            images=images,
            labels=labels,
        )

        assert mixed_images.shape == (4, 32, 32, 3)
        assert labels_a.shape == (4,)
        assert labels_b.shape == (4,)
        assert lam.shape == ()

    def test_unknown_mixer_raises_error(self) -> None:
        """Verify that unknown mixer raises error."""
        with pytest.raises(ValueError):
            get_mixer(
                name="unknown",
                num_classes=10,
                alpha=1.0,
            )
