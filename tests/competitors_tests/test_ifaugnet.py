from __future__ import annotations

import gc
import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn

from allthemix.cli.args import parse_args
from allthemix.config import load_yaml_config
from allthemix.competitors.ifaugnet.influence import (
    compute_s_test,
    s_test_residual_norm,
)
from allthemix.competitors.ifaugnet.models import (
    AugmentationNetwork,
    FeatureDiscriminator,
    ImageDiscriminator,
)
from allthemix.competitors.ifaugnet.transforms import (
    apply_appearance_transform,
    apply_spatial_transform,
    combine_transforms,
)
from allthemix.competitors.ifaugnet.steps import (
    AugNetRetrainStrategy,
    augnet_influence_train_step,
    augnet_pretrain_step,
    classifier_retrain_step,
    compute_batch_s_test,
    create_augment_state,
    create_discriminator_state,
    get_classifier_head_params,
    infer_feature_dim,
    inherit_pretrained_decoder,
)
from allthemix.methods.selector import get_mixer
from allthemix.networks.builder import build_model
from allthemix.training.engine.single.train import create_train_state, train_step


class _TinyBatchNormClassifier(nn.Module):
    """Small deterministic classifier that exercises mutable BatchNorm state."""

    num_classes: int = 4

    @nn.compact
    def __call__(
        self,
        images: jnp.ndarray,
        training: bool = True,
        return_features: bool = False,
        feature_hook=None,
        sync_batch_stats: bool = False,
    ):
        """Return logits and optional pooled features."""
        del feature_hook
        features = nn.Conv(
            features=8,
            kernel_size=(3, 3),
            padding="SAME",
        )(images)
        features = nn.BatchNorm(
            use_running_average=not training,
            momentum=0.9,
            axis_name=("batch" if sync_batch_stats else None),
        )(features)
        features = nn.relu(features)
        features = jnp.mean(features, axis=(1, 2))
        logits = nn.Dense(self.num_classes)(features)

        if return_features:
            return logits, features

        return logits


def _tiny_augnet() -> AugmentationNetwork:
    """Build a lightweight AugNet that retains the production operations."""
    return AugmentationNetwork(
        image_size=8,
        channels=3,
        tau_dim=8,
        tau_dropout=0.0,
        spatial_scale=0.05,
        appearance_scale=0.05,
        encoder_widths=(4, 8),
        decoder_widths=(8,),
        decoder_base_width=8,
    )


def test_ifaugnet_zero_probability_matches_erm_update() -> None:
    """Require disabled learned augmentation to reduce exactly to ERM."""
    classifier = _TinyBatchNormClassifier()
    classifier_state = create_train_state(
        rng=jax.random.PRNGKey(0),
        model=classifier,
        learning_rate=0.01,
        momentum=0.9,
        weight_decay=0.001,
        input_shape=(4, 8, 8, 3),
    )
    augment_model = _tiny_augnet()
    augment_state = create_augment_state(
        rng=jax.random.PRNGKey(1),
        model=augment_model,
        input_shape=(4, 8, 8, 3),
        learning_rate=1.0e-3,
    )
    images = jax.random.uniform(
        jax.random.PRNGKey(2),
        (4, 8, 8, 3),
    )
    labels = jnp.asarray(
        [0, 1, 2, 3],
    )
    rng = jax.random.PRNGKey(3)

    erm_state, erm_loss, erm_accuracy = train_step(
        state=classifier_state,
        rng=rng,
        images=images,
        labels=labels,
        mixer_fn=get_mixer(
            name="baseline",
            num_classes=4,
        ),
        method="baseline",
        num_classes=4,
    )
    ifaugnet_state, ifaugnet_loss, ifaugnet_accuracy, metrics = (
        classifier_retrain_step(
            task_state=classifier_state,
            augment_state=augment_state,
            images=images,
            labels=labels,
            rng=rng,
            mean=jnp.zeros((1, 1, 1, 3)),
            std=jnp.ones((1, 1, 1, 3)),
            learned_aug_probability=0.0,
        )
    )

    np.testing.assert_allclose(ifaugnet_loss, erm_loss, rtol=0.0, atol=1.0e-7)
    np.testing.assert_allclose(
        ifaugnet_accuracy,
        erm_accuracy,
        rtol=0.0,
        atol=0.0,
    )
    jax.tree_util.tree_map(
        lambda actual, expected: np.testing.assert_allclose(
            actual,
            expected,
            rtol=0.0,
            atol=1.0e-7,
        ),
        ifaugnet_state.params,
        erm_state.params,
    )
    jax.tree_util.tree_map(
        lambda actual, expected: np.testing.assert_allclose(
            actual,
            expected,
            rtol=0.0,
            atol=1.0e-7,
        ),
        ifaugnet_state.batch_stats,
        erm_state.batch_stats,
    )
    assert jax.tree_util.tree_leaves(ifaugnet_state.batch_stats)
    assert float(metrics["ifaugnet_learned_aug_fraction"]) == 0.0


def test_ifaugnet_model_and_influence_are_finite() -> None:
    """Check transform shape and the damped iHVP residual."""
    images = jax.random.uniform(
        jax.random.PRNGKey(0),
        (2, 8, 8, 3),
    )
    model = _tiny_augnet()
    variables = model.init(
        {
            "params": jax.random.PRNGKey(1),
            "dropout": jax.random.PRNGKey(2),
        },
        images,
        training=True,
        return_aux=True,
    )
    augmented, aux = model.apply(
        variables,
        images,
        training=False,
        return_aux=True,
    )

    assert augmented.shape == images.shape
    assert aux["tau"].shape == (2, 8)
    assert aux["tau_pre_dropout"].shape == (2, 8)
    assert aux["fields"].shape == (2, 8, 8, 18)
    assert bool(
        jnp.all(
            jnp.isfinite(
                augmented,
            )
        )
    )

    train_features = jax.random.normal(
        jax.random.PRNGKey(3),
        (5, 3),
    )
    validation_features = jax.random.normal(
        jax.random.PRNGKey(4),
        (4, 3),
    )
    train_labels = jnp.asarray(
        [0, 1, 2, 1, 0],
    )
    validation_labels = jnp.asarray(
        [2, 1, 0, 2],
    )
    classifier_params = {
        "kernel": jax.random.normal(
            jax.random.PRNGKey(5),
            (3, 3),
        )
        * 0.05,
        "bias": jnp.zeros(
            (3,),
        ),
    }
    s_test = compute_s_test(
        classifier_params=classifier_params,
        train_features=train_features,
        train_labels=train_labels,
        validation_features=validation_features,
        validation_labels=validation_labels,
        damping=0.05,
        cg_iters=80,
    )
    residual = s_test_residual_norm(
        classifier_params=classifier_params,
        train_features=train_features,
        train_labels=train_labels,
        validation_features=validation_features,
        validation_labels=validation_labels,
        s_test=s_test,
        damping=0.05,
    )

    assert float(
        residual,
    ) < 1.0e-3


def test_ifaugnet_runner_solves_one_aggregate_s_test_system(
    monkeypatch,
) -> None:
    """Concatenate fixed feature batches before one CG solve and residual."""
    from allthemix.competitors.ifaugnet import runner

    train_batches = [
        {
            "raw_images": np.asarray([[101.0], [102.0]], dtype=np.float32),
            "images": np.asarray([[1.0], [2.0]], dtype=np.float32),
            "labels": np.asarray([0, 1], dtype=np.int64),
        },
        {
            "raw_images": np.asarray([[103.0], [104.0]], dtype=np.float32),
            "images": np.asarray([[3.0], [4.0]], dtype=np.float32),
            "labels": np.asarray([1, 0], dtype=np.int64),
        },
    ]
    validation_batches = [
        (
            np.asarray([[5.0], [6.0]], dtype=np.float32),
            np.asarray([0, 1], dtype=np.int64),
        ),
        (
            np.asarray([[7.0], [8.0]], dtype=np.float32),
            np.asarray([1, 0], dtype=np.int64),
        ),
    ]
    classifier_state = SimpleNamespace(
        params={
            "head": {
                "kernel": jnp.zeros((1, 2), dtype=jnp.float32),
                "bias": jnp.zeros((2,), dtype=jnp.float32),
            },
        },
    )
    solve_calls = []
    residual_calls = []

    def fake_extract(**kwargs):
        return (
            kwargs["train_images"],
            kwargs["train_labels"],
            kwargs["validation_images"],
            kwargs["validation_labels"],
        )

    def fake_solve(**kwargs):
        solve_calls.append(kwargs)
        return {
            "kernel": jnp.ones((1, 2), dtype=jnp.float32),
            "bias": jnp.ones((2,), dtype=jnp.float32),
        }

    def fake_residual(**kwargs):
        residual_calls.append(kwargs)
        return jnp.asarray(0.0, dtype=jnp.float32)

    monkeypatch.setattr(runner, "extract_s_test_feature_batch", fake_extract)
    monkeypatch.setattr(runner, "compute_feature_s_test", fake_solve)
    monkeypatch.setattr(
        runner,
        "compute_feature_s_test_residual",
        fake_residual,
    )
    s_test, returned_rng = runner._precompute_s_test(
        args=SimpleNamespace(
            distributed=False,
            ifaugnet_s_test_batches=2,
            ifaugnet_damping=0.01,
            ifaugnet_cg_iters=5,
        ),
        classifier_state=classifier_state,
        paired_dataset=train_batches,
        validation_dataset=validation_batches,
        rng=jax.random.PRNGKey(11),
    )

    assert len(solve_calls) == 1
    assert len(residual_calls) == 1
    np.testing.assert_array_equal(
        solve_calls[0]["train_features"],
        np.asarray([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        solve_calls[0]["validation_features"],
        np.asarray([[5.0], [6.0], [7.0], [8.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        residual_calls[0]["train_features"],
        solve_calls[0]["train_features"],
    )
    np.testing.assert_array_equal(returned_rng, jax.random.PRNGKey(11))
    assert set(s_test) == {"kernel", "bias"}


def test_ifaugnet_runner_does_not_repeat_validation_for_s_test(
    monkeypatch,
) -> None:
    """Stop the aggregate solve at the end of one finite validation pass."""
    from allthemix.competitors.ifaugnet import runner

    train_batches = [
        {
            "raw_images": np.asarray([[1.0], [2.0]], dtype=np.float32),
            "images": np.asarray([[1.0], [2.0]], dtype=np.float32),
            "labels": np.asarray([0, 1], dtype=np.int64),
        }
    ]
    validation_batches = [
        (
            np.asarray([[3.0], [4.0]], dtype=np.float32),
            np.asarray([0, 1], dtype=np.int64),
        ),
        (
            np.asarray([[5.0]], dtype=np.float32),
            np.asarray([1], dtype=np.int64),
        ),
    ]
    classifier_state = SimpleNamespace(
        params={
            "head": {
                "kernel": jnp.zeros((1, 2)),
                "bias": jnp.zeros((2,)),
            }
        }
    )
    solve_calls = []

    monkeypatch.setattr(
        runner,
        "extract_s_test_feature_batch",
        lambda **kwargs: (
            kwargs["train_images"],
            kwargs["train_labels"],
            kwargs["validation_images"],
            kwargs["validation_labels"],
        ),
    )

    def fake_solve(**kwargs):
        solve_calls.append(kwargs)
        return {
            "kernel": jnp.ones((1, 2)),
            "bias": jnp.ones((2,)),
        }

    monkeypatch.setattr(runner, "compute_feature_s_test", fake_solve)
    monkeypatch.setattr(
        runner,
        "compute_feature_s_test_residual",
        lambda **_: jnp.asarray(0.0),
    )

    runner._precompute_s_test(
        args=SimpleNamespace(
            distributed=False,
            ifaugnet_s_test_batches=4,
            ifaugnet_damping=0.01,
            ifaugnet_cg_iters=5,
        ),
        classifier_state=classifier_state,
        paired_dataset=train_batches,
        validation_dataset=validation_batches,
        rng=jax.random.PRNGKey(0),
    )

    assert len(solve_calls) == 1
    np.testing.assert_array_equal(
        solve_calls[0]["validation_features"],
        np.asarray([[3.0], [4.0], [5.0]], dtype=np.float32),
    )


def test_ifaugnet_validation_padding_only_adds_edge_copies() -> None:
    """Make a partial validation batch shardable without changing its prefix."""
    from allthemix.competitors.ifaugnet import runner

    values = jnp.asarray(
        [1, 2, 3, 4, 5],
        dtype=jnp.int32,
    )
    padded = runner._pad_leading_axis_to_multiple(
        values,
        multiple=4,
    )

    np.testing.assert_array_equal(
        padded,
        np.asarray([1, 2, 3, 4, 5, 5, 5, 5], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        padded[: values.shape[0]],
        values,
    )


def test_ifaugnet_pretrain_tau_sampling_matches_inverted_dropout() -> None:
    """Sample pretrain codes on the support G sees behind tau dropout."""
    from allthemix.competitors.ifaugnet import runner

    rng = jax.random.PRNGKey(
        0,
    )
    plain = runner._sample_pretrain_tau(
        rng=rng,
        batch_size=64,
        tau_dim=128,
    )
    legacy = jax.random.uniform(
        rng,
        shape=(64, 128),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )

    # The default path must stay bit-identical to the previous sampler so
    # seeded reproductions of earlier pretraining runs are unaffected.
    np.testing.assert_array_equal(
        np.asarray(plain),
        np.asarray(legacy),
    )

    matched = runner._sample_pretrain_tau(
        rng=rng,
        batch_size=64,
        tau_dim=128,
        dropout_rate=0.5,
    )
    matched_values = np.asarray(
        matched,
    )
    zero_fraction = float(
        np.mean(
            matched_values == 0.0,
        )
    )
    nonzero_values = matched_values[matched_values != 0.0]

    assert 0.4 < zero_fraction < 0.6
    assert np.all(np.abs(nonzero_values) <= 2.0)
    assert np.any(np.abs(nonzero_values) > 1.0)

    saturated = runner._sample_pretrain_tau(
        rng=rng,
        batch_size=8,
        tau_dim=16,
        dropout_rate=1.0,
    )

    np.testing.assert_array_equal(
        np.asarray(saturated),
        np.zeros((8, 16), dtype=np.float32),
    )


def test_ifaugnet_paper_spatial_transform_uses_pixel_affine_formula() -> None:
    """Match Eq. 20 without tanh, fixed scaling, or boundary clipping."""
    images = jnp.arange(
        9,
        dtype=jnp.float32,
    ).reshape(
        (1, 3, 3, 1),
    )
    fields = jnp.zeros(
        (1, 3, 3, 6),
        dtype=jnp.float32,
    ).at[..., 4].set(
        1.0,
    )
    transformed, sample_grid = apply_spatial_transform(
        images=images,
        spatial_params=fields,
        spatial_scale=0.001,
        smoothing_kernel=1,
        parameterization="paper",
    )

    np.testing.assert_array_equal(
        transformed[0, ..., 0],
        np.asarray(
            [
                [3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        sample_grid[0, ..., 0],
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
            ],
            dtype=np.float32,
        ),
    )


def test_ifaugnet_paper_spatial_transform_smooths_final_flow() -> None:
    """Pool Eq. 20 displacement flow rather than its affine parameter fields."""
    images = jnp.zeros((1, 3, 3, 1), dtype=jnp.float32)
    fields = jnp.zeros((1, 3, 3, 6), dtype=jnp.float32)
    fields = fields.at[0, 1, 1, 0].set(1.0)

    _, sample_grid = apply_spatial_transform(
        images=images,
        spatial_params=fields,
        smoothing_kernel=3,
        parameterization="paper",
    )

    expected_y = np.asarray(
        [
            [-1.0 + 1.0 / 9.0, -1.0 + 1.0 / 9.0, -1.0 + 1.0 / 9.0],
            [1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0],
            [1.0 + 1.0 / 9.0, 1.0 + 1.0 / 9.0, 1.0 + 1.0 / 9.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(
        sample_grid[0, ..., 0],
        expected_y,
        rtol=0.0,
        atol=1.0e-6,
    )


def test_ifaugnet_paper_appearance_transform_is_unscaled_residual() -> None:
    """Apply x + delta directly and retain values outside display range."""
    images = jnp.full(
        (1, 2, 2, 3),
        0.9,
        dtype=jnp.float32,
    )
    fields = jnp.zeros(
        (1, 2, 2, 12),
        dtype=jnp.float32,
    ).at[..., 9:].set(
        0.2,
    )
    transformed, delta = apply_appearance_transform(
        images=images,
        appearance_params=fields,
        appearance_scale=0.001,
        smoothing_kernel=1,
        parameterization="paper",
    )

    np.testing.assert_allclose(delta, 0.2, rtol=0.0, atol=1.0e-7)
    np.testing.assert_allclose(transformed, 1.1, rtol=0.0, atol=1.0e-7)


def test_ifaugnet_parallel_composition_adds_appearance_residual() -> None:
    """Compute spatial(x) + appearance(x) - x for the paper's parallel form."""
    images = jnp.full((1, 2, 2, 1), 0.5)
    spatial_images = jnp.full((1, 2, 2, 1), 0.7)
    appearance_images = jnp.full((1, 2, 2, 1), 0.6)

    combined = combine_transforms(
        images=images,
        spatial_images=spatial_images,
        appearance_images=appearance_images,
        composition="parallel",
        clip_output=False,
    )

    np.testing.assert_allclose(combined, 0.8, rtol=0.0, atol=1.0e-7)


def test_ifaugnet_imagenet_profile_has_deep_224_architecture() -> None:
    """Use the supplement's seven conv/deconv blocks around the dense layer."""
    model = AugmentationNetwork(
        image_size=224,
        channels=3,
        tau_dropout=0.0,
        parameterization="paper",
        composition="parallel",
        architecture="imagenet",
    )
    images = jnp.zeros(
        (1, 224, 224, 3),
        dtype=jnp.float32,
    )
    variables = model.init(
        {
            "params": jax.random.PRNGKey(30),
            "dropout": jax.random.PRNGKey(31),
        },
        images,
        training=False,
        return_aux=True,
    )
    augmented, aux = model.apply(
        variables,
        images,
        training=False,
        return_aux=True,
    )

    assert augmented.shape == images.shape
    assert aux["fields"].shape == (1, 224, 224, 18)
    assert {
        key
        for key in variables["params"]["encoder"]
        if key.startswith("conv_")
    } == {
        f"conv_{index}"
        for index in range(7)
    }
    assert {
        key
        for key in variables["params"]["decoder"]
        if key.startswith("deconv_")
    } == {
        f"deconv_{index}"
        for index in range(6)
    }


def test_ifaugnet_influence_state_inherits_only_pretrained_decoder() -> None:
    """Keep a fresh encoder and optimizer while transferring pretrained G."""
    model = _tiny_augnet()
    fresh_state = create_augment_state(
        rng=jax.random.PRNGKey(20),
        model=model,
        input_shape=(2, 8, 8, 3),
        learning_rate=1.0e-3,
    )
    pretrained_state = create_augment_state(
        rng=jax.random.PRNGKey(21),
        model=model,
        input_shape=(2, 8, 8, 3),
        learning_rate=1.0e-3,
    )
    transferred_state = inherit_pretrained_decoder(
        fresh_state=fresh_state,
        pretrained_state=pretrained_state,
    )

    jax.tree_util.tree_map(
        lambda actual, expected: np.testing.assert_array_equal(
            actual,
            expected,
        ),
        transferred_state.params["encoder"],
        fresh_state.params["encoder"],
    )
    jax.tree_util.tree_map(
        lambda actual, expected: np.testing.assert_array_equal(
            actual,
            expected,
        ),
        transferred_state.params["decoder"],
        pretrained_state.params["decoder"],
    )
    jax.tree_util.tree_map(
        lambda actual, expected: np.testing.assert_array_equal(
            actual,
            expected,
        ),
        transferred_state.opt_state,
        fresh_state.opt_state,
    )
    assert int(
        transferred_state.step,
    ) == 0


def test_ifaugnet_full_differentiable_chain_updates_states() -> None:
    """Run pretrain, iHVP, influence, and fresh-classifier update once."""
    classifier = build_model(
        name="simple_cnn",
        num_classes=4,
    )
    classifier_state = create_train_state(
        rng=jax.random.PRNGKey(0),
        model=classifier,
        learning_rate=0.01,
        momentum=0.9,
        weight_decay=0.0,
        input_shape=(4, 8, 8, 3),
    )
    augment_model = _tiny_augnet()
    augment_state = create_augment_state(
        rng=jax.random.PRNGKey(1),
        model=augment_model,
        input_shape=(4, 8, 8, 3),
        learning_rate=1.0e-3,
    )
    image_discriminator = ImageDiscriminator(
        widths=(4, 8),
    )
    feature_discriminator = FeatureDiscriminator()
    image_discriminator_state = create_discriminator_state(
        rng=jax.random.PRNGKey(2),
        model=image_discriminator,
        input_shape=(4, 8, 8, 3),
        learning_rate=1.0e-3,
    )
    feature_dim = infer_feature_dim(
        classifier_state=classifier_state,
        input_shape=(4, 8, 8, 3),
    )
    feature_discriminator_state = create_discriminator_state(
        rng=jax.random.PRNGKey(3),
        model=feature_discriminator,
        input_shape=(4, feature_dim),
        learning_rate=1.0e-3,
    )
    raw_images = jax.random.uniform(
        jax.random.PRNGKey(4),
        (4, 8, 8, 3),
    )
    real_images = raw_images[:, ::-1]
    labels = jnp.asarray(
        [0, 1, 2, 3],
    )
    mean = jnp.zeros(
        (1, 1, 1, 3),
    )
    std = jnp.ones(
        (1, 1, 1, 3),
    )
    pretrain_params = augment_state.params
    discriminator_tau = jax.random.uniform(
        jax.random.PRNGKey(5),
        (4, 8),
        minval=-1.0,
        maxval=1.0,
    )
    generator_tau = jax.random.uniform(
        jax.random.PRNGKey(6),
        (4, 8),
        minval=-1.0,
        maxval=1.0,
    )
    (
        augment_state,
        image_discriminator_state,
        feature_discriminator_state,
        pretrain_metrics,
    ) = augnet_pretrain_step(
        augment_state=augment_state,
        image_discriminator_state=image_discriminator_state,
        feature_discriminator_state=feature_discriminator_state,
        classifier_state=classifier_state,
        raw_images=raw_images,
        real_images=real_images,
        discriminator_tau=discriminator_tau,
        generator_tau=generator_tau,
        mean=mean,
        std=std,
        identity_l2_weight=0.001,
    )
    s_test = compute_batch_s_test(
        classifier_state=classifier_state,
        train_images=raw_images,
        train_labels=labels,
        validation_images=raw_images[::-1],
        validation_labels=labels[::-1],
        damping=0.05,
        cg_iters=5,
    )
    influence_params = augment_state.params
    augment_state, influence_metrics = augnet_influence_train_step(
        augment_state=augment_state,
        classifier_state=classifier_state,
        raw_images=raw_images,
        labels=labels,
        s_test=s_test,
        rng=jax.random.PRNGKey(6),
        mean=mean,
        std=std,
        identity_l2_weight=0.001,
        influence_clip_value=10.0,
        label_preservation_weight=0.1,
    )
    strategy = AugNetRetrainStrategy(
        augment_state=augment_state,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        learned_aug_probability=1.0,
    )
    classifier_params = classifier_state.params
    classifier_state, loss, _, retrain_metrics = strategy.train_step(
        task_state=classifier_state,
        images=raw_images,
        labels=labels,
        rng=jax.random.PRNGKey(7),
    )

    def tree_delta(before, after) -> float:
        """Return the L2 distance between two matching parameter trees."""
        squared = jax.tree_util.tree_leaves(
            jax.tree_util.tree_map(
                lambda left, right: jnp.sum(
                    jnp.square(
                        right - left,
                    )
                ),
                before,
                after,
            )
        )

        return float(
            jnp.sqrt(
                sum(
                    squared,
                )
            )
        )

    assert tree_delta(
        pretrain_params,
        influence_params,
    ) > 0.0
    assert tree_delta(
        pretrain_params["encoder"],
        influence_params["encoder"],
    ) == 0.0
    assert tree_delta(
        pretrain_params["decoder"],
        influence_params["decoder"],
    ) > 0.0
    assert tree_delta(
        influence_params,
        augment_state.params,
    ) > 0.0
    assert tree_delta(
        classifier_params,
        classifier_state.params,
    ) > 0.0
    assert float(
        retrain_metrics["ifaugnet_learned_aug_fraction"],
    ) == 1.0
    assert bool(
        jnp.isfinite(
            loss,
        )
    )
    assert all(
        bool(
            jnp.isfinite(
                value,
            )
        )
        for value in {
            **pretrain_metrics,
            **influence_metrics,
        }.values()
    )
    assert set(
        get_classifier_head_params(
            classifier_state.params,
        )
    ) == {
        "kernel",
        "bias",
    }


def test_ifaugnet_distributed_chain_matches_global_batch_semantics() -> None:
    """Exercise every PMAP IF-AugNet update on two forced CPU devices."""
    script = textwrap.dedent(
        """
        import flax.linen as nn
        import jax
        import jax.numpy as jnp
        import numpy as np
        from flax import jax_utils

        from allthemix.competitors.ifaugnet.models import (
            AugmentationNetwork,
            FeatureDiscriminator,
            ImageDiscriminator,
        )
        from allthemix.competitors.ifaugnet.steps import (
            AugNetRetrainStrategy,
            augnet_influence_train_step,
            augnet_pretrain_step,
            classifier_retrain_step,
            compute_batch_s_test,
            create_augment_state,
            create_discriminator_state,
            infer_feature_dim,
            parallel_augnet_influence_train_step,
            parallel_augnet_pretrain_step,
            parallel_compute_batch_s_test,
            parallel_compute_batch_s_test_residual,
        )
        from allthemix.methods.selector import get_mixer
        from allthemix.networks.batch_norm import batch_norm
        from allthemix.training.engine.parallel.parallel_loop import (
            parallel_train_one_epoch,
        )
        from allthemix.training.engine.single.train import create_train_state
        from allthemix.utils.parallel import shard_array

        class TinyClassifier(nn.Module):
            '''Classifier with a real synchronized BatchNorm collection.'''

            @nn.compact
            def __call__(
                self,
                images,
                training=True,
                return_features=False,
                sync_batch_stats=False,
            ):
                features = nn.Conv(
                    features=4,
                    kernel_size=(3, 3),
                    padding="SAME",
                )(images)
                features = batch_norm(
                    features,
                    training=training,
                    sync_batch_stats=sync_batch_stats,
                )
                features = nn.relu(features).mean(axis=(1, 2))
                logits = nn.Dense(4, name="head")(features)

                if return_features:
                    return logits, features

                return logits

        def replica_spread(tree):
            '''Return the largest difference from replica zero.'''
            return max(
                (
                    float(
                        np.max(
                            np.abs(np.asarray(leaf) - np.asarray(leaf)[0])
                        )
                    )
                    for leaf in jax.tree_util.tree_leaves(tree)
                ),
                default=0.0,
            )

        def assert_replicated_tree_matches(
            distributed_tree,
            single_tree,
            stage,
            ignored_paths=(),
        ):
            '''Compare replica zero with one global-batch update.'''
            distributed_leaves, distributed_tree_def = (
                jax.tree_util.tree_flatten_with_path(distributed_tree)
            )
            single_leaves, single_tree_def = jax.tree_util.tree_flatten_with_path(
                single_tree
            )
            assert distributed_tree_def == single_tree_def

            for (path, distributed_leaf), (_, single_leaf) in zip(
                distributed_leaves,
                single_leaves,
            ):
                if jax.tree_util.keystr(path) in ignored_paths:
                    continue

                try:
                    np.testing.assert_allclose(
                        np.asarray(distributed_leaf)[0],
                        np.asarray(single_leaf),
                        rtol=2.0e-5,
                        atol=2.0e-5,
                    )
                except AssertionError as error:
                    raise AssertionError(
                        f"{stage} mismatch at {jax.tree_util.keystr(path)}"
                    ) from error

        assert jax.local_device_count() == 2
        batch_size = 4
        classifier = TinyClassifier()
        classifier_state = create_train_state(
            rng=jax.random.PRNGKey(0),
            model=classifier,
            learning_rate=0.01,
            momentum=0.0,
            weight_decay=0.0,
            input_shape=(batch_size, 8, 8, 3),
        )
        augment_model = AugmentationNetwork(
            image_size=8,
            channels=3,
            tau_dim=8,
            tau_dropout=0.0,
            spatial_scale=0.05,
            appearance_scale=0.05,
            encoder_widths=(4, 8),
            decoder_widths=(8,),
            decoder_base_width=8,
        )
        augment_state = create_augment_state(
            rng=jax.random.PRNGKey(1),
            model=augment_model,
            input_shape=(batch_size, 8, 8, 3),
            learning_rate=1.0e-3,
        )
        feature_dim = infer_feature_dim(
            classifier_state=classifier_state,
            input_shape=(batch_size, 8, 8, 3),
        )
        image_discriminator_state = create_discriminator_state(
            rng=jax.random.PRNGKey(2),
            model=ImageDiscriminator(widths=(4, 8)),
            input_shape=(batch_size, 8, 8, 3),
            learning_rate=1.0e-3,
        )
        feature_discriminator_state = create_discriminator_state(
            rng=jax.random.PRNGKey(3),
            model=FeatureDiscriminator(),
            input_shape=(batch_size, feature_dim),
            learning_rate=1.0e-3,
        )
        raw_images = jax.random.uniform(
            jax.random.PRNGKey(4),
            (batch_size, 8, 8, 3),
        )
        real_images = raw_images[:, ::-1]
        labels = jnp.arange(batch_size, dtype=jnp.int32)
        mean = jnp.zeros((1, 1, 1, 3), dtype=jnp.float32)
        std = jnp.ones((1, 1, 1, 3), dtype=jnp.float32)
        discriminator_tau = jax.random.uniform(
            jax.random.PRNGKey(5),
            (batch_size, 8),
            minval=-1.0,
            maxval=1.0,
        )
        generator_tau = jax.random.uniform(
            jax.random.PRNGKey(6),
            (batch_size, 8),
            minval=-1.0,
            maxval=1.0,
        )
        replicated_classifier = jax_utils.replicate(classifier_state)
        (
            replicated_augment,
            replicated_image_discriminator,
            replicated_feature_discriminator,
            pretrain_metrics,
        ) = parallel_augnet_pretrain_step(
            jax_utils.replicate(augment_state),
            jax_utils.replicate(image_discriminator_state),
            jax_utils.replicate(feature_discriminator_state),
            replicated_classifier,
            shard_array(raw_images),
            shard_array(real_images),
            shard_array(discriminator_tau),
            shard_array(generator_tau),
            mean,
            std,
            1.0,
            1.0,
            0.001,
        )
        (
            single_augment,
            single_image_discriminator,
            single_feature_discriminator,
            _,
        ) = augnet_pretrain_step(
            augment_state=augment_state,
            image_discriminator_state=image_discriminator_state,
            feature_discriminator_state=feature_discriminator_state,
            classifier_state=classifier_state,
            raw_images=raw_images,
            real_images=real_images,
            discriminator_tau=discriminator_tau,
            generator_tau=generator_tau,
            mean=mean,
            std=std,
            image_loss_weight=1.0,
            feature_loss_weight=1.0,
            identity_l2_weight=0.001,
        )
        assert replica_spread(replicated_augment.params) == 0.0
        assert replica_spread(replicated_image_discriminator.params) == 0.0
        assert replica_spread(replicated_feature_discriminator.params) == 0.0
        assert all(
            np.isfinite(np.asarray(value)).all()
            for value in pretrain_metrics.values()
        )
        assert_replicated_tree_matches(
            replicated_augment.params,
            single_augment.params,
            "pretrain augment",
        )
        assert_replicated_tree_matches(
            replicated_image_discriminator.params,
            single_image_discriminator.params,
            "pretrain image discriminator",
            ignored_paths=("['logit']['bias']",),
        )
        assert_replicated_tree_matches(
            replicated_feature_discriminator.params,
            single_feature_discriminator.params,
            "pretrain feature discriminator",
            ignored_paths=("['logit']['bias']",),
        )
        distributed_image_logits = replicated_image_discriminator.apply_fn(
            {
                "params": jax_utils.unreplicate(
                    replicated_image_discriminator,
                ).params,
            },
            raw_images,
        )
        single_image_logits = single_image_discriminator.apply_fn(
            {
                "params": single_image_discriminator.params,
            },
            raw_images,
        )
        np.testing.assert_allclose(
            distributed_image_logits - distributed_image_logits.mean(),
            single_image_logits - single_image_logits.mean(),
            rtol=2.0e-5,
            atol=2.0e-5,
        )
        feature_probe = jnp.arange(
            batch_size * feature_dim,
            dtype=jnp.float32,
        ).reshape(batch_size, feature_dim)
        distributed_feature_logits = (
            replicated_feature_discriminator.apply_fn(
                {
                    "params": jax_utils.unreplicate(
                        replicated_feature_discriminator,
                    ).params,
                },
                feature_probe,
            )
        )
        single_feature_logits = single_feature_discriminator.apply_fn(
            {
                "params": single_feature_discriminator.params,
            },
            feature_probe,
        )
        np.testing.assert_allclose(
            distributed_feature_logits - distributed_feature_logits.mean(),
            single_feature_logits - single_feature_logits.mean(),
            rtol=2.0e-5,
            atol=2.0e-5,
        )

        distributed_s_test = parallel_compute_batch_s_test(
            replicated_classifier,
            shard_array(raw_images),
            shard_array(labels),
            shard_array(real_images),
            shard_array(labels[::-1]),
            0.05,
            3,
        )
        single_s_test = compute_batch_s_test(
            classifier_state=classifier_state,
            train_images=raw_images,
            train_labels=labels,
            validation_images=real_images,
            validation_labels=labels[::-1],
            damping=0.05,
            cg_iters=3,
        )
        for distributed_leaf, single_leaf in zip(
            jax.tree_util.tree_leaves(distributed_s_test),
            jax.tree_util.tree_leaves(single_s_test),
        ):
            np.testing.assert_allclose(
                np.asarray(distributed_leaf)[0],
                np.asarray(single_leaf),
                rtol=1.0e-5,
                atol=1.0e-5,
            )
        residual = parallel_compute_batch_s_test_residual(
            replicated_classifier,
            shard_array(raw_images),
            shard_array(labels),
            shard_array(real_images),
            shard_array(labels[::-1]),
            distributed_s_test,
            0.05,
        )
        assert replica_spread(distributed_s_test) == 0.0
        assert np.isfinite(np.asarray(residual)).all()

        replicated_augment, influence_metrics = (
            parallel_augnet_influence_train_step(
                replicated_augment,
                replicated_classifier,
                shard_array(raw_images),
                shard_array(labels),
                distributed_s_test,
                jax.random.split(jax.random.PRNGKey(6), 2),
                mean,
                std,
                0.001,
                10.0,
                0.1,
            )
        )
        single_augment, _ = augnet_influence_train_step(
            augment_state=single_augment,
            classifier_state=classifier_state,
            raw_images=raw_images,
            labels=labels,
            s_test=single_s_test,
            rng=jax.random.PRNGKey(6),
            mean=mean,
            std=std,
            identity_l2_weight=0.001,
            influence_clip_value=10.0,
            label_preservation_weight=0.1,
        )
        assert replica_spread(replicated_augment.params) == 0.0
        assert all(
            np.isfinite(np.asarray(value)).all()
            for value in influence_metrics.values()
        )
        assert_replicated_tree_matches(
            replicated_augment.params,
            single_augment.params,
            "influence augment",
        )

        strategy = AugNetRetrainStrategy(
            augment_state=jax_utils.unreplicate(replicated_augment),
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
            learned_aug_probability=1.0,
            distributed=True,
            sync_batch_stats=True,
        )
        trained_state, _, loss, _, retrain_metrics = parallel_train_one_epoch(
            state=jax_utils.replicate(classifier_state),
            rngs=jax.random.split(jax.random.PRNGKey(7), 2),
            train_ds=[(np.asarray(raw_images), np.asarray(labels))],
            mixer_fn=get_mixer("baseline", 4),
            method="ifaugnet",
            num_classes=4,
            max_train_steps=-1,
            sync_batch_stats=True,
            batch_training_strategy=strategy,
        )
        single_trained_state, _, _, _ = classifier_retrain_step(
            task_state=classifier_state,
            augment_state=single_augment,
            images=raw_images,
            labels=labels,
            rng=jax.random.PRNGKey(7),
            mean=mean,
            std=std,
            learned_aug_probability=1.0,
        )
        assert np.isfinite(loss)
        assert replica_spread(trained_state.params) == 0.0
        assert replica_spread(trained_state.batch_stats) == 0.0
        assert retrain_metrics["ifaugnet_learned_aug_fraction"] == 1.0
        assert_replicated_tree_matches(
            trained_state.params,
            single_trained_state.params,
            "retrain params",
        )
        assert_replicated_tree_matches(
            trained_state.batch_stats,
            single_trained_state.batch_stats,
            "retrain batch stats",
        )
        """
    )
    jax.clear_caches()
    gc.collect()
    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu"
    environment["XLA_FLAGS"] = (
        "--xla_force_host_platform_device_count=2 "
        "--xla_cpu_multi_thread_eigen=false"
    )
    environment["OMP_NUM_THREADS"] = "1"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"

    subprocess.run(
        [
            sys.executable,
            "-c",
            script,
        ],
        check=True,
        cwd=str(
            Path(__file__).resolve().parents[2],
        ),
        env=environment,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "distributed",
    [
        False,
        True,
    ],
)
def test_ifaugnet_integrated_runner_smoke(
    monkeypatch,
    tmp_path,
    distributed: bool,
) -> None:
    """Exercise all four stages through the unified runner without data I/O."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--config",
            "configs/cifar10/preact_resnet18/ifaugnet.yaml",
        ],
    )
    args = parse_args()
    assert args.distributed is True
    assert args.sync_batch_stats is True
    assert args.cross_device_shuffle is False
    args.model = "simple_cnn"
    args.batch_size = 4
    args.epochs = 1
    args.max_train_steps = 1
    args.max_eval_steps = 1
    args.ifaugnet_pretrain_steps = 1 if distributed else 0
    args.ifaugnet_influence_steps = 1
    args.ifaugnet_s_test_batches = 1
    args.ifaugnet_cg_iters = 2
    args.ifaugnet_retrain_epochs = 1
    args.ifaugnet_tau_dim = 8
    args.ifaugnet_tau_dropout = 0.0
    args.ifaugnet_encoder_widths = [4, 8]
    args.ifaugnet_decoder_widths = [8]
    args.ifaugnet_decoder_base_width = 8
    args.ifaugnet_architecture = "custom"
    args.ifaugnet_min_accuracy_retention = 0.0
    args.ifaugnet_max_tau_saturation_fraction = 1.0
    args.save_checkpoint = False
    args.resume_checkpoint = ""
    args.wandb = False
    args.log_time = False
    args.output_dir = str(
        tmp_path,
    )
    args.output_name = "ifaugnet.csv"
    args.run_name = f"ifaugnet_smoke_{distributed}"
    args.distributed = distributed
    args.sync_batch_stats = distributed
    generator = np.random.default_rng(
        0,
    )
    images = generator.normal(
        size=(4, 32, 32, 3),
    ).astype(
        np.float32,
    )
    labels = np.asarray(
        [0, 1, 2, 3],
        dtype=np.int64,
    )
    train_dataset = [
        (
            images[:, ::-1].copy(),
            labels,
        )
    ]
    validation_dataset = [
        (
            images,
            labels,
        )
    ]
    paired_dataset = [
        {
            "images": images[:, ::-1].copy(),
            "raw_images": images,
            "labels": labels,
        }
    ]

    @dataclass
    class Metadata:
        """Small in-memory dataset metadata for the runner smoke test."""

        num_classes: int = 4
        image_size: int = 32
        channels: int = 3
        num_train_examples: int = 8
        num_test_examples: int = 4

    import allthemix.competitors.ifaugnet.runner as runner

    monkeypatch.setattr(
        runner,
        "get_metadata",
        lambda _name: Metadata(),
    )
    monkeypatch.setattr(
        runner,
        "build_dataset_pipeline",
        lambda **_kwargs: (
            train_dataset,
            validation_dataset,
        ),
    )
    monkeypatch.setattr(
        runner,
        "build_raw_augmented_train_pipeline",
        lambda **_kwargs: paired_dataset,
    )
    monkeypatch.setattr(
        runner,
        "build_meta_validation_pipeline",
        lambda **_kwargs: validation_dataset,
    )
    monkeypatch.setattr(
        runner,
        "build_test_pipeline",
        lambda **_kwargs: validation_dataset,
    )
    runner.run_ifaugnet(
        args,
    )

    expected = {
        "ifaugnet.csv",
        "ifaugnet_classifier.csv",
        "ifaugnet_pretrain.csv",
        "ifaugnet_influence.csv",
        "ifaugnet_final_test.csv",
    }

    assert expected == {
        path.name
        for path in tmp_path.iterdir()
    }


def test_cifar100_ifaugnet_config_uses_paper_method_schedule(
    monkeypatch,
) -> None:
    """Use the paper transform, optimizer, pretraining, and retraining rate."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--config",
            "configs/cifar100/preact_resnet18/ifaugnet.yaml",
        ],
    )
    args = parse_args()

    assert args.dataset == "cifar100"
    assert args.method == "ifaugnet"
    assert args.ifaugnet_pretrain_steps == 2_000
    assert args.ifaugnet_influence_steps == 10_000
    assert args.ifaugnet_pretrain_learning_rate == pytest.approx(2.0e-4)
    assert args.ifaugnet_pretrain_beta1 == pytest.approx(0.5)
    assert args.ifaugnet_pretrain_beta2 == pytest.approx(0.999)
    assert args.ifaugnet_learning_rate == pytest.approx(0.01)
    assert args.ifaugnet_lr_schedule == "constant"
    assert args.ifaugnet_min_learning_rate == pytest.approx(0.0)
    assert args.ifaugnet_warmup_steps == 0
    assert args.ifaugnet_beta1 == pytest.approx(0.9)
    assert args.ifaugnet_beta2 == pytest.approx(0.99)
    assert args.ifaugnet_tau_dim == 128
    assert args.ifaugnet_tau_dropout == pytest.approx(0.5)
    assert args.ifaugnet_transform_parameterization == "paper"
    assert args.ifaugnet_composition == "parallel"
    assert args.ifaugnet_architecture == "cifar"
    assert args.ifaugnet_policy_classifier_checkpoint == "final"
    assert args.ifaugnet_pretrain_identity_l2_weight == pytest.approx(0.0)
    assert args.ifaugnet_identity_l2_weight == pytest.approx(0.0)
    assert args.ifaugnet_label_preservation_weight == pytest.approx(0.0)
    assert args.ifaugnet_influence_clip_value == pytest.approx(0.0)
    # Val-selected probability for cifar100 (see the p-sweep log entry).
    assert args.ifaugnet_learned_aug_probability == pytest.approx(0.25)
    assert args.ifaugnet_gradient_clip_norm == pytest.approx(0.0)
    assert args.ifaugnet_zero_nonfinite_grads is False
    assert args.ifaugnet_restore_last_healthy_pretrain is False
    assert args.ifaugnet_restore_best_healthy is True
    assert args.ifaugnet_collapse_patience == 3
    assert args.distributed is True
    assert args.sync_batch_stats is True
    assert args.cross_device_shuffle is False


def test_cifar100_ifaugnet_stable_config_matches_signal_benchmark(
    monkeypatch,
) -> None:
    """Keep the CIFAR-100 signal screen matched to the stable JAX profile."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--config",
            "configs/cifar100/preact_resnet18/ifaugnet_jax_stable.yaml",
        ],
    )

    args = parse_args()

    assert args.dataset == "cifar100"
    assert args.validation_split == pytest.approx(0.1)
    assert args.eval_on_test_each_epoch is False
    assert args.ifaugnet_influence_steps == 1_000
    assert args.ifaugnet_learning_rate == pytest.approx(1.0e-4)
    assert args.ifaugnet_lr_schedule == "warmup_cosine"
    assert args.ifaugnet_min_learning_rate == pytest.approx(1.0e-5)
    assert args.ifaugnet_warmup_steps == 100
    assert args.ifaugnet_s_test_batches == 40
    assert args.ifaugnet_policy_classifier_checkpoint == "best"
    assert args.ifaugnet_learned_aug_probability == pytest.approx(0.25)
    assert args.ifaugnet_retrain_policy_source == "influence"
    assert args.distributed is True
    assert args.sync_batch_stats is True
    assert args.cross_device_shuffle is False


def test_cars196_ifaugnet_config_matches_fine_grained_protocol(
    monkeypatch,
) -> None:
    """Verify Cars IF-AugNet inherits the shared classifier protocol."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--config",
            "configs/cars196/preact_resnet18/ifaugnet.yaml",
        ],
    )
    args = parse_args()

    assert args.dataset == "cars196"
    assert args.method == "ifaugnet"
    assert args.batch_size == 64
    assert args.epochs == 200
    assert args.validation_split == 0.1
    assert args.final_test_checkpoint == "best"
    assert args.learning_rate == 0.05
    assert args.weight_decay == 5.0e-4
    assert args.lr_schedule == "cosine"
    assert args.aug_recipe == "fine_grained"
    assert args.ifaugnet_retrain_epochs == 200
    assert args.ifaugnet_retrain_learning_rate == 0.05
    assert args.ifaugnet_pretrain_steps == 2_000
    assert args.ifaugnet_transform_parameterization == "paper"
    assert args.ifaugnet_composition == "parallel"
    assert args.ifaugnet_architecture == "imagenet"
    # Val-selected probability for cars196 (see the p-sweep log entry).
    assert args.ifaugnet_learned_aug_probability == pytest.approx(0.25)
    assert args.distributed is True
    assert args.sync_batch_stats is True
    assert args.cross_device_shuffle is False


@pytest.mark.parametrize(
    ("config_path", "expected_architecture", "expected_probability"),
    (
        ("configs/cifar10/preact_resnet18/ifaugnet.yaml", "cifar", 1.0),
        ("configs/cifar100/preact_resnet18/ifaugnet.yaml", "cifar", 0.25),
        ("configs/stl10/preact_resnet18/ifaugnet.yaml", "cifar", 0.1),
        ("configs/cars196/preact_resnet18/ifaugnet.yaml", "imagenet", 0.25),
    ),
)
def test_ifaugnet_reportable_configs_enable_all_paper_path_fixes(
    monkeypatch,
    config_path: str,
    expected_architecture: str,
    expected_probability: float,
) -> None:
    """Lock the five audited paper-path corrections across every dataset."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--config",
            config_path,
        ],
    )
    args = parse_args()

    assert args.ifaugnet_transform_parameterization == "paper"
    assert args.ifaugnet_composition == "parallel"
    assert args.ifaugnet_architecture == expected_architecture
    assert args.ifaugnet_pretrain_steps == 2_000
    # Per-dataset val-selected learned-aug probabilities.
    assert args.ifaugnet_learned_aug_probability == pytest.approx(
        expected_probability
    )
    assert args.ifaugnet_restore_best_healthy is True
    assert args.ifaugnet_collapse_patience == 3


def test_cub200_ifaugnet_config_stabilizes_paper_architecture(
    monkeypatch,
) -> None:
    """Keep the CUB paper model while preventing raw-field optimizer collapse."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--config",
            "configs/cub200/preact_resnet18/ifaugnet.yaml",
        ],
    )
    args = parse_args()

    assert args.dataset == "caltech_birds2011"
    assert args.method == "ifaugnet"
    assert args.ifaugnet_transform_parameterization == "paper"
    assert args.ifaugnet_composition == "parallel"
    assert args.ifaugnet_architecture == "imagenet"
    assert args.ifaugnet_pretrain_steps == 2_000
    assert args.ifaugnet_influence_steps == 1_000
    assert args.ifaugnet_learning_rate == pytest.approx(1.0e-4)
    assert args.ifaugnet_pretrain_identity_l2_weight == pytest.approx(0.001)
    assert args.ifaugnet_identity_l2_weight == pytest.approx(0.02)
    assert args.ifaugnet_label_preservation_weight == pytest.approx(0.1)
    assert args.ifaugnet_influence_clip_value == pytest.approx(1.0)
    assert args.ifaugnet_learned_aug_probability == pytest.approx(0.1)
    assert args.ifaugnet_gradient_clip_norm == pytest.approx(1.0)
    assert args.ifaugnet_zero_nonfinite_grads is True
    assert args.ifaugnet_restore_last_healthy_pretrain is True
    assert args.ifaugnet_restore_best_healthy is True
    assert args.ifaugnet_min_accuracy_retention == pytest.approx(0.90)
    assert args.ifaugnet_max_tau_saturation_fraction == pytest.approx(0.50)
    assert args.ifaugnet_collapse_patience == 5
    assert args.distributed is True
    assert args.sync_batch_stats is True
    assert args.cross_device_shuffle is False


def test_stl10_ifaugnet_config_uses_paper_method_schedule(
    monkeypatch,
) -> None:
    """Apply the paper method schedule to the matched STL extension."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--config",
            "configs/stl10/preact_resnet18/ifaugnet.yaml",
        ],
    )
    args = parse_args()

    assert args.dataset == "stl10"
    assert args.method == "ifaugnet"
    assert args.ifaugnet_pretrain_steps == 2_000
    assert args.ifaugnet_influence_steps == 10_000
    assert args.ifaugnet_pretrain_learning_rate == pytest.approx(2.0e-4)
    assert args.ifaugnet_pretrain_beta1 == pytest.approx(0.5)
    assert args.ifaugnet_pretrain_beta2 == pytest.approx(0.999)
    assert args.ifaugnet_learning_rate == pytest.approx(0.01)
    assert args.ifaugnet_beta1 == pytest.approx(0.9)
    assert args.ifaugnet_beta2 == pytest.approx(0.99)
    assert args.ifaugnet_tau_dim == 128
    assert args.ifaugnet_tau_dropout == pytest.approx(0.5)
    assert args.ifaugnet_transform_parameterization == "paper"
    assert args.ifaugnet_composition == "parallel"
    assert args.ifaugnet_architecture == "cifar"
    assert args.ifaugnet_pretrain_identity_l2_weight == pytest.approx(0.0)
    assert args.ifaugnet_identity_l2_weight == pytest.approx(0.0)
    assert args.ifaugnet_label_preservation_weight == pytest.approx(0.0)
    assert args.ifaugnet_influence_clip_value == pytest.approx(0.0)
    # Val-selected probability for stl10 (see the p-sweep log entry).
    assert args.ifaugnet_learned_aug_probability == pytest.approx(0.1)
    assert args.ifaugnet_gradient_clip_norm == pytest.approx(0.0)
    assert args.ifaugnet_zero_nonfinite_grads is False
    assert args.ifaugnet_restore_last_healthy_pretrain is False
    assert args.ifaugnet_restore_best_healthy is True
    assert args.ifaugnet_collapse_patience == 3
    assert args.distributed is True
    assert args.sync_batch_stats is True
    assert args.cross_device_shuffle is False


def test_stl10_ifaugnet_jax_tuned_config_stabilizes_paper_path(
    monkeypatch,
) -> None:
    """Tune JAX optimization without changing the audited method family."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--config",
            "configs/stl10/preact_resnet18/ifaugnet_jax_tuned.yaml",
        ],
    )
    args = parse_args()

    assert args.dataset == "stl10"
    assert args.method == "ifaugnet"
    assert args.ifaugnet_transform_parameterization == "paper"
    assert args.ifaugnet_composition == "parallel"
    assert args.ifaugnet_architecture == "cifar"
    assert args.ifaugnet_pretrain_steps == 2_000
    assert args.ifaugnet_influence_steps == 1_000
    assert args.ifaugnet_learning_rate == pytest.approx(1.0e-4)
    assert args.ifaugnet_identity_l2_weight == pytest.approx(0.02)
    assert args.ifaugnet_label_preservation_weight == pytest.approx(0.1)
    assert args.ifaugnet_influence_clip_value == pytest.approx(1.0)
    assert args.ifaugnet_learned_aug_probability == pytest.approx(0.25)
    assert args.ifaugnet_gradient_clip_norm == pytest.approx(1.0)
    assert args.ifaugnet_zero_nonfinite_grads is True
    assert args.ifaugnet_restore_last_healthy_pretrain is True
    assert args.ifaugnet_restore_best_healthy is True
    assert args.ifaugnet_min_accuracy_retention == pytest.approx(0.90)
    assert args.ifaugnet_max_tau_saturation_fraction == pytest.approx(0.50)
    assert args.ifaugnet_collapse_patience == 5
    assert args.distributed is True
    assert args.sync_batch_stats is True
    assert args.cross_device_shuffle is False


def test_cifar10_ifaugnet_jax_stable_config_stabilizes_paper_path(
    monkeypatch,
) -> None:
    """The CIFAR-10 stable profile must retain the paper transform family."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            "configs/cifar10/preact_resnet18/ifaugnet_jax_stable.yaml",
        ],
    )

    args = parse_args()

    assert args.dataset == "cifar10"
    assert args.method == "ifaugnet"
    assert args.ifaugnet_transform_parameterization == "paper"
    assert args.ifaugnet_composition == "parallel"
    assert args.ifaugnet_architecture == "cifar"
    assert args.ifaugnet_pretrain_steps == 2000
    assert args.ifaugnet_influence_steps == 1000
    assert args.ifaugnet_learning_rate == pytest.approx(1.0e-4)
    assert args.ifaugnet_lr_schedule == "warmup_cosine"
    assert args.ifaugnet_min_learning_rate == pytest.approx(1.0e-5)
    assert args.ifaugnet_warmup_steps == 100
    assert args.ifaugnet_pretrain_identity_l2_weight == pytest.approx(0.001)
    assert args.ifaugnet_identity_l2_weight == pytest.approx(0.02)
    assert args.ifaugnet_label_preservation_weight == pytest.approx(0.1)
    assert args.ifaugnet_influence_clip_value == pytest.approx(1.0)
    assert args.ifaugnet_gradient_clip_norm == pytest.approx(1.0)
    assert args.ifaugnet_zero_nonfinite_grads is True
    assert args.ifaugnet_learned_aug_probability == pytest.approx(0.25)
    assert args.ifaugnet_restore_last_healthy_pretrain is True
    assert args.ifaugnet_restore_best_healthy is True
    assert args.ifaugnet_min_accuracy_retention == pytest.approx(0.90)
    assert args.ifaugnet_max_tau_saturation_fraction == pytest.approx(0.50)
    assert args.ifaugnet_collapse_patience == 5
    assert args.distributed is True
    assert args.sync_batch_stats is True
    assert args.cross_device_shuffle is False


def test_ifaugnet_influence_warmup_cosine_schedule() -> None:
    """Warm up the sensitive policy optimizer, then decay to its floor."""
    from allthemix.competitors.ifaugnet.runner import (
        _build_influence_lr_schedule,
    )

    schedule = _build_influence_lr_schedule(
        SimpleNamespace(
            ifaugnet_lr_schedule="warmup_cosine",
            ifaugnet_learning_rate=1.0e-4,
            ifaugnet_min_learning_rate=1.0e-5,
            ifaugnet_warmup_steps=100,
            ifaugnet_influence_steps=1000,
        )
    )

    assert float(schedule(0)) == pytest.approx(0.0)
    assert float(schedule(100)) == pytest.approx(1.0e-4)
    assert float(schedule(1000)) == pytest.approx(1.0e-5)


def test_cifar10_ifaugnet_lowpeak_config_changes_only_policy_lr(
    monkeypatch,
) -> None:
    """Keep the matched protocol while staying below the observed LR edge."""
    stable_config = load_yaml_config(
        "configs/cifar10/preact_resnet18/ifaugnet_jax_stable.yaml"
    )
    lowpeak_config = load_yaml_config(
        "configs/cifar10/preact_resnet18/ifaugnet_jax_lowpeak.yaml"
    )
    changed_keys = {
        key
        for key in stable_config.keys() | lowpeak_config.keys()
        if stable_config.get(key) != lowpeak_config.get(key)
    }

    assert changed_keys == {
        "ifaugnet_learning_rate",
        "ifaugnet_min_learning_rate",
        "run_name",
    }

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            "configs/cifar10/preact_resnet18/ifaugnet_jax_lowpeak.yaml",
        ],
    )

    args = parse_args()

    assert args.dataset == "cifar10"
    assert args.method == "ifaugnet"
    assert args.ifaugnet_transform_parameterization == "paper"
    assert args.ifaugnet_influence_steps == 1000
    assert args.ifaugnet_lr_schedule == "warmup_cosine"
    assert args.ifaugnet_learning_rate == pytest.approx(5.0e-5)
    assert args.ifaugnet_min_learning_rate == pytest.approx(5.0e-6)
    assert args.ifaugnet_warmup_steps == 100
    assert args.ifaugnet_learned_aug_probability == pytest.approx(0.25)
    assert args.distributed is True
    assert args.sync_batch_stats is True
    assert args.cross_device_shuffle is False


def test_cifar10_ifaugnet_100e_ablation_keeps_test_sealed() -> None:
    """Compare matched short runs without touching the official test split."""
    import ast
    import re

    script = Path(
        "scripts/experiment_run/ablate_cifar10_ifaugnet_100e.sh"
    ).read_text()

    assert "--ifaugnet_retrain_epochs \"${RETRAIN_EPOCHS}\"" in script
    assert "RETRAIN_EPOCHS=\"${RETRAIN_EPOCHS:-100}\"" in script
    assert "--final_test false" in script
    assert "--final_test true" not in script
    assert '${REPO_ROOT}/.venv/bin/python' in script
    assert "current_p025" in script
    assert "current_p000" in script
    assert "lowpeak_p025" in script
    assert '"selection_metric": "best_validation_top1_error"' in script
    assert 'selected = min(results, key=selection_key)' in script
    assert 'run_dir / "selected.env"' in script
    assert '"OFFICIAL_TEST_USED": "false"' in script

    python_blocks = re.findall(
        r"<<'PY'\n(.*?)\nPY",
        script,
        flags=re.DOTALL,
    )
    summary_block = next(
        block
        for block in python_blocks
        if "validation_selection.json" in block
    )
    summary_tree = ast.parse(summary_block)
    imported_modules = {
        alias.name
        for node in summary_tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert {"csv", "json", "shlex", "sys"} <= imported_modules


def test_stl10_ifaugnet_jax_tuning_keeps_test_split_sealed() -> None:
    """Use validation for selection and reserve test for the selected run."""
    tuning_script = Path(
        "scripts/experiment_run/tune_stl10_ifaugnet_jax.sh"
    ).read_text()
    selected_script = Path(
        "scripts/experiment_run/run_stl10_ifaugnet_jax_selected.sh"
    ).read_text()

    assert "--final_test false" in tuning_script
    assert "--final_test true" not in tuning_script
    assert "--final_test true" in selected_script
    assert "--final_test_checkpoint best" in selected_script


def test_ifaugnet_guard_rejects_stl_paper_probe_collapse() -> None:
    """Reject the semantic loss and tau saturation observed in the failed run."""
    from allthemix.competitors.ifaugnet.runner import _policy_is_healthy

    assert not _policy_is_healthy(
        metrics={
            "loss": -0.642,
            "accuracy_retention": 0.211,
            "tau_saturation_fraction": 0.0,
        },
        min_accuracy_retention=0.8,
        max_tau_saturation_fraction=0.5,
    )
    assert not _policy_is_healthy(
        metrics={
            "loss": 8.62,
            "accuracy_retention": 0.18,
            "tau_saturation_fraction": 0.994,
        },
        min_accuracy_retention=0.8,
        max_tau_saturation_fraction=0.5,
    )


@pytest.mark.parametrize(
    ("dataset", "config_path", "architecture"),
    (
        (
            "stl10",
            "configs/stl10/preact_resnet18/ifaugnet_paper_probe.yaml",
            "cifar",
        ),
        (
            "cifar100",
            "configs/cifar100/preact_resnet18/ifaugnet_paper_probe.yaml",
            "cifar",
        ),
        (
            "caltech_birds2011",
            "configs/cub200/preact_resnet18/ifaugnet_paper_probe.yaml",
            "imagenet",
        ),
    ),
)
def test_ifaugnet_paper_probe_preserves_strict_reference_settings(
    monkeypatch,
    dataset: str,
    config_path: str,
    architecture: str,
) -> None:
    """Keep the failed strict schedule available as a diagnostic, not default."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--config",
            config_path,
        ],
    )
    args = parse_args()

    assert args.dataset == dataset
    assert args.ifaugnet_pretrain_steps == 2_000
    assert args.ifaugnet_influence_steps == 10_000
    assert args.ifaugnet_learning_rate == pytest.approx(0.01)
    assert args.ifaugnet_learned_aug_probability == pytest.approx(1.0)
    assert args.ifaugnet_transform_parameterization == "paper"
    assert args.ifaugnet_composition == "parallel"
    assert args.ifaugnet_architecture == architecture
    assert args.ifaugnet_restore_best_healthy is False
    assert args.ifaugnet_collapse_patience > args.ifaugnet_influence_steps


def test_retrain_promotes_saved_best_healthy_policy(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Resume a strict run that stopped before writing its final policy."""
    from allthemix.competitors.ifaugnet import runner

    healthy_path = tmp_path / "ifaugnet_influence_best_healthy.msgpack"
    healthy_path.touch()
    restored_state = object()
    restored_names: list[str] = []
    saved_names: list[str] = []

    def fake_restore_named(*, template, checkpoint_root, name):
        del template, checkpoint_root
        restored_names.append(name)
        return restored_state

    def fake_save_named(*, state, checkpoint_root, name):
        assert state is restored_state
        assert checkpoint_root == tmp_path
        saved_names.append(name)

    monkeypatch.setattr(
        runner,
        "_restore_named_model_state",
        fake_restore_named,
    )
    monkeypatch.setattr(runner, "_save_named", fake_save_named)

    result = runner._restore_influence_for_retrain(
        template=object(),
        checkpoint_root=tmp_path,
        restore_best_healthy=True,
    )

    assert result is restored_state
    assert restored_names == ["ifaugnet_influence_best_healthy"]
    assert saved_names == ["ifaugnet_influence_final"]


def test_stage_dependency_restore_ignores_legacy_optimizer_structure(
    tmp_path: Path,
) -> None:
    """Load policy parameters across Optax-chain changes without ambiguity."""
    from flax import struct

    from allthemix.utils.checkpoint import (
        restore_model_state_file,
        restore_state_file,
        save_state_file,
    )

    @struct.dataclass
    class AugmentCheckpoint:
        augment_state: object

    model = _tiny_augnet()
    legacy_state = create_augment_state(
        rng=jax.random.PRNGKey(101),
        model=model,
        input_shape=(2, 8, 8, 3),
        learning_rate=1.0e-4,
        gradient_clip_norm=0.0,
        zero_nonfinite_grads=False,
    )
    current_state = create_augment_state(
        rng=jax.random.PRNGKey(202),
        model=model,
        input_shape=(2, 8, 8, 3),
        learning_rate=5.0e-5,
        gradient_clip_norm=1.0,
        zero_nonfinite_grads=True,
    )
    legacy_checkpoint = AugmentCheckpoint(
        augment_state=legacy_state,
    )
    current_checkpoint = AugmentCheckpoint(
        augment_state=current_state,
    )
    checkpoint_path = save_state_file(
        state=legacy_checkpoint,
        checkpoint_dir=tmp_path,
        name="legacy_policy",
    )

    with pytest.raises(ValueError, match="size of the list"):
        restore_state_file(
            state=current_checkpoint,
            checkpoint_path=checkpoint_path,
        )

    restored, loaded = restore_model_state_file(
        state=current_checkpoint,
        checkpoint_path=checkpoint_path,
    )

    assert loaded == ["augment_state/params"]
    assert jax.tree_util.tree_structure(
        restored.augment_state.opt_state
    ) == jax.tree_util.tree_structure(current_state.opt_state)
    for restored_leaf, legacy_leaf in zip(
        jax.tree_util.tree_leaves(restored.augment_state.params),
        jax.tree_util.tree_leaves(legacy_state.params),
    ):
        np.testing.assert_array_equal(restored_leaf, legacy_leaf)
    for restored_leaf, current_leaf in zip(
        jax.tree_util.tree_leaves(restored.augment_state.opt_state),
        jax.tree_util.tree_leaves(current_state.opt_state),
    ):
        np.testing.assert_array_equal(restored_leaf, current_leaf)


@pytest.mark.parametrize(
    "config_path",
    (
        "configs/stl10/preact_resnet18/ifaugnet_guarded.yaml",
        "configs/cifar100/preact_resnet18/ifaugnet_guarded.yaml",
    ),
)
def test_ifaugnet_guarded_configs_preserve_previous_behavior(
    monkeypatch,
    config_path: str,
) -> None:
    """Keep the old stabilization profile explicit for audit reruns."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--config",
            config_path,
        ],
    )
    args = parse_args()

    assert args.ifaugnet_transform_parameterization == "guarded"
    assert args.ifaugnet_composition == "serial"
    assert args.ifaugnet_pretrain_steps == 0
    assert args.ifaugnet_influence_steps == 300
    assert args.ifaugnet_learning_rate == pytest.approx(3.0e-5)
    assert args.ifaugnet_learned_aug_probability == pytest.approx(0.05)


def test_ifaugnet_distributed_mode_requires_sync_batch_stats(
    monkeypatch,
) -> None:
    """Reject distributed classifier stages with local-only BatchNorm."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--method",
            "ifaugnet",
            "--validation_split",
            "0.1",
            "--eval_on_test_each_epoch",
            "false",
            "--distributed",
            "true",
        ],
    )

    with pytest.raises(
        ValueError,
        match="sync_batch_stats=true",
    ):
        parse_args()


def test_ifaugnet_accepts_synchronized_distributed_mode(
    monkeypatch,
) -> None:
    """Accept IF-AugNet PMAP when global BatchNorm is explicit."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--method",
            "ifaugnet",
            "--validation_split",
            "0.1",
            "--eval_on_test_each_epoch",
            "false",
            "--distributed",
            "true",
            "--sync_batch_stats",
            "true",
        ],
    )

    args = parse_args()

    assert args.distributed is True
    assert args.sync_batch_stats is True
