from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow as tf
from flax import serialization

from allthemix.competitors.metaaugment.augmentations import (
    NUM_OPS,
    apply_op,
    cutout,
    initial_sampler_probs,
    sample_transformations,
    transformation_embedding,
)
from allthemix.competitors.metaaugment.runtime import (
    create_metaaugment_context,
    normalized_policy_weights,
)
from allthemix.cli.args import parse_args
from allthemix.data.pipeline import build_meta_validation_pipeline
from allthemix.networks.builder import build_model
from allthemix.training.engine.single.loop import train_one_epoch
from allthemix.training.engine.single.train import create_train_state
from allthemix.utils.checkpoint import restore_checkpoint, save_best_checkpoint


def _tree_delta(
    before,
    after,
) -> float:
    """Return the L2 distance between matching parameter trees."""
    squared = 0.0

    for left, right in zip(
        jax.tree_util.tree_leaves(
            before,
        ),
        jax.tree_util.tree_leaves(
            after,
        ),
    ):
        difference = np.asarray(
            right - left,
        )
        squared += float(
            np.sum(
                difference * difference,
            )
        )

    return squared ** 0.5


def _make_context_and_state():
    """Build a tiny shared classifier and its integrated policy context."""
    model = build_model(
        name="simple_cnn",
        num_classes=3,
    )
    state = create_train_state(
        rng=jax.random.PRNGKey(
            0,
        ),
        model=model,
        learning_rate=0.01,
        momentum=0.0,
        weight_decay=0.0,
        input_shape=(
            2,
            8,
            8,
            3,
        ),
    )
    meta_images = jnp.linspace(
        -1.0,
        1.0,
        2 * 8 * 8 * 3,
    ).reshape(
        (
            2,
            8,
            8,
            3,
        )
    )
    meta_labels = jnp.asarray(
        [
            1,
            2,
        ],
        dtype=jnp.int32,
    )
    context = create_metaaugment_context(
        rng=jax.random.PRNGKey(
            1,
        ),
        task_state=state,
        meta_dataset=[
            (
                meta_images,
                meta_labels,
            )
        ],
        input_shape=(
            2,
            8,
            8,
            3,
        ),
        dataset="cifar10",
        num_classes=3,
        policy_learning_rate=0.01,
        policy_momentum=0.0,
        policy_weight_decay=0.0,
        inner_learning_rate=0.01,
        learn_inner_learning_rate=True,
        cutout_size=0,
        sampler_history_epochs=2,
        translate_const=2.0,
    )

    return state, context


def test_transformation_embedding_matches_reference_layout() -> None:
    """Verify operation pairs occupy alternating slots in the 28-D embedding."""
    embedding = transformation_embedding(
        op1=jnp.asarray(
            [
                2,
            ]
        ),
        op2=jnp.asarray(
            [
                13,
            ]
        ),
        magnitude1=jnp.asarray(
            [
                4.0,
            ]
        ),
        magnitude2=jnp.asarray(
            [
                7.0,
            ]
        ),
    )

    assert embedding.shape == (
        1,
        NUM_OPS * 2,
    )
    assert float(
        embedding[
            0,
            4,
        ]
    ) == 5.0
    assert float(
        embedding[
            0,
            27,
        ]
    ) == 11.0
    assert int(
        jnp.count_nonzero(
            embedding,
        )
    ) == 2


def test_sampler_and_policy_weight_normalization() -> None:
    """Verify sampling dimensions and normalized controller weights."""
    sampled = sample_transformations(
        key=jax.random.PRNGKey(
            2,
        ),
        sampler_probs=initial_sampler_probs(),
        batch_size=4,
        num_transforms_per_sample=2,
    )
    assert all(
        value.shape == (
            8,
        )
        for value in sampled
    )
    weights = normalized_policy_weights(
        jnp.asarray(
            [
                1.0,
                2.0,
                3.0,
            ]
        )
    )
    np.testing.assert_allclose(
        np.asarray(
            jnp.sum(
                weights,
            )
        ),
        1.0,
        rtol=1.0e-6,
    )


def test_cutout_uses_exact_configured_side_length() -> None:
    """Ensure an interior even-sized Cutout mask is not one pixel too wide."""
    images = jnp.zeros(
        (
            1,
            32,
            32,
            3,
        ),
        dtype=jnp.float32,
    )
    output = cutout(
        images=images,
        key=jax.random.PRNGKey(
            2,
        ),
        size=16,
    )
    changed = np.any(
        np.asarray(
            output,
        ) != 0.0,
        axis=-1,
    )

    assert int(
        np.count_nonzero(
            changed,
        )
    ) == 16 * 16


def test_all_operation_outputs_match_reference_regression() -> None:
    """Lock deterministic operation outputs to the imported reference."""
    image = jnp.linspace(
        0.0,
        1.0,
        8 * 8 * 3,
        dtype=jnp.float32,
    ).reshape(
        (
            8,
            8,
            3,
        )
    )
    actual = []

    for operation_id in range(
        NUM_OPS,
    ):
        output = apply_op(
            image=image,
            op_id=jnp.asarray(
                operation_id,
            ),
            magnitude=jnp.asarray(
                6.25,
            ),
            key=jax.random.PRNGKey(
                100 + operation_id,
            ),
            translate_const=3.0,
        )
        actual.append(
            float(
                jnp.sum(
                    output,
                )
            )
        )

    expected = np.asarray(
        [
            96.000008,
            96.0,
            96.0,
            94.870605,
            50.764397,
            95.895386,
            95.895393,
            130.379272,
            96.0,
            96.0,
            96.0,
            98.143982,
            78.848175,
            96.0,
        ]
    )
    np.testing.assert_allclose(
        np.asarray(
            actual,
        ),
        expected,
        rtol=1.0e-6,
        atol=1.0e-5,
    )


def test_unified_epoch_loop_updates_task_policy_and_sampler() -> None:
    """Verify the shared epoch loop executes the complete bilevel update."""
    state, context = _make_context_and_state()
    original_task_params = state.params
    original_policy_params = context.method_state.policy_state.params
    images = jnp.linspace(
        -1.5,
        1.5,
        2 * 8 * 8 * 3,
    ).reshape(
        (
            2,
            8,
            8,
            3,
        )
    )
    labels = jnp.asarray(
        [
            0,
            1,
        ],
        dtype=jnp.int32,
    )
    new_state, _, loss, accuracy, metrics = train_one_epoch(
        state=state,
        rng=jax.random.PRNGKey(
            3,
        ),
        train_ds=[
            (
                images,
                labels,
            )
        ],
        mixer_fn=None,
        method="metaaugment",
        num_classes=3,
        max_train_steps=-1,
        validation_aware_strategy=context,
    )

    assert np.isfinite(
        loss,
    )
    assert 0.0 <= accuracy <= 1.0
    assert int(
        new_state.step,
    ) == 1
    assert int(
        context.method_state.policy_state.step,
    ) == 1
    assert _tree_delta(
        original_task_params,
        new_state.params,
    ) > 0.0
    assert _tree_delta(
        original_policy_params,
        context.method_state.policy_state.params,
    ) > 0.0
    assert int(
        context.method_state.sampler_history_count,
    ) == 1
    assert "metaaugment_policy_loss" in metrics
    assert "metaaugment_inner_lr" in metrics
    assert "metaaugment_sampler_entropy" in metrics
    np.testing.assert_allclose(
        np.asarray(
            jnp.sum(
                context.method_state.sampler_probs,
            )
        ),
        1.0,
        rtol=1.0e-6,
    )


def test_metaaugment_checkpoint_contains_shared_task_state() -> None:
    """Verify one checkpoint tree owns both shared and policy state."""
    state, context = _make_context_and_state()
    checkpoint = context.checkpoint_state(
        task_state=state,
    )

    assert checkpoint.task_state is state
    assert checkpoint.method_state is context.method_state


def test_metaaugment_checkpoint_round_trip(
    tmp_path,
) -> None:
    """Verify the shared Orbax path restores classifier and policy state."""
    state, context = _make_context_and_state()
    checkpoint = context.checkpoint_state(
        task_state=state,
    )
    try:
        save_best_checkpoint(
            state=checkpoint,
            checkpoint_dir=tmp_path,
        )
    except AttributeError as exc:
        if "enable_memories" not in str(
            exc,
        ):
            raise
        payload = serialization.to_bytes(
            checkpoint,
        )
        restored = serialization.from_bytes(
            checkpoint,
            payload,
        )
    else:
        restored = restore_checkpoint(
            state=checkpoint,
            checkpoint_path=str(
                tmp_path / "best",
            ),
        )

    assert int(
        restored.task_state.step,
    ) == int(
        state.step,
    )
    assert int(
        restored.method_state.policy_state.step,
    ) == int(
        context.method_state.policy_state.step,
    )
    np.testing.assert_allclose(
        np.asarray(
            restored.method_state.sampler_probs,
        ),
        np.asarray(
            context.method_state.sampler_probs,
        ),
    )


def test_unified_metaaugment_config_parses(
    monkeypatch,
) -> None:
    """Verify the flat AllTheMix config owns policy hyperparameters."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--config",
            "configs/cifar10/preact_resnet18/metaaugment.yaml",
        ],
    )
    args = parse_args()

    assert args.method == "metaaugment"
    assert args.validation_split == pytest.approx(
        0.1,
    )
    assert args.metaaugment_policy_learning_rate == pytest.approx(
        1.0e-3,
    )
    assert args.distributed is True
    assert args.sync_batch_stats is True
    assert args.cross_device_shuffle is False


def test_cifar100_metaaugment_config_uses_synchronized_global_batch_norm(
    monkeypatch,
) -> None:
    """Verify PMAP preserves the main protocol's global BN semantics."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--config",
            "configs/cifar100/preact_resnet18/metaaugment.yaml",
        ],
    )
    args = parse_args()

    assert args.dataset == "cifar100"
    assert args.method == "metaaugment"
    assert args.distributed is True
    assert args.sync_batch_stats is True
    assert args.cross_device_shuffle is False


def test_cars196_metaaugment_config_matches_fine_grained_protocol(
    monkeypatch,
) -> None:
    """Verify Cars MetaAugment shares the fine-grained task protocol."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--config",
            "configs/cars196/preact_resnet18/metaaugment.yaml",
        ],
    )
    args = parse_args()

    assert args.dataset == "cars196"
    assert args.method == "metaaugment"
    assert args.batch_size == 64
    assert args.epochs == 200
    assert args.validation_split == pytest.approx(0.1)
    assert args.final_test_checkpoint == "best"
    assert args.learning_rate == pytest.approx(0.05)
    assert args.weight_decay == pytest.approx(5.0e-4)
    assert args.lr_schedule == "cosine"
    assert args.aug_recipe == "fine_grained"
    assert args.metaaugment_cutout_size == 0
    assert args.metaaugment_learn_inner_learning_rate is True
    assert args.distributed is True
    assert args.sync_batch_stats is True
    assert args.cross_device_shuffle is False


def test_cub200_metaaugment_config_matches_fine_grained_protocol(
    monkeypatch,
) -> None:
    """Verify CUB MetaAugment shares the CUB classifier protocol."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--config",
            "configs/cub200/preact_resnet18/metaaugment.yaml",
        ],
    )
    args = parse_args()

    assert args.dataset == "caltech_birds2011"
    assert args.method == "metaaugment"
    assert args.batch_size == 64
    assert args.epochs == 200
    assert args.validation_split == pytest.approx(0.1)
    assert args.eval_on_test_each_epoch is False
    assert args.final_test is True
    assert args.final_test_checkpoint == "best"
    assert args.learning_rate == pytest.approx(0.05)
    assert args.weight_decay == pytest.approx(5.0e-4)
    assert args.lr_schedule == "cosine"
    assert args.aug_recipe == "cub"
    assert args.metaaugment_inner_learning_rate == pytest.approx(0.05)
    assert args.metaaugment_learn_inner_learning_rate is True
    assert args.metaaugment_sampler_update_epochs == 5
    assert args.metaaugment_sampler_history_epochs == 50
    assert args.metaaugment_cutout_size == 0
    assert args.distributed is True
    assert args.sync_batch_stats is True
    assert args.cross_device_shuffle is False


def test_cub200_stable_metaaugment_alias_matches_formal_config(
    monkeypatch,
) -> None:
    """Verify the legacy CUB stability path aliases the formal configuration."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--config",
            "configs/cub200/preact_resnet18/metaaugment_stable.yaml",
        ],
    )
    args = parse_args()

    assert args.dataset == "caltech_birds2011"
    assert args.method == "metaaugment"
    assert args.batch_size == 64
    assert args.epochs == 200
    assert args.validation_split == pytest.approx(0.1)
    assert args.learning_rate == pytest.approx(0.05)
    assert args.weight_decay == pytest.approx(5.0e-4)
    assert args.lr_schedule == "cosine"
    assert args.aug_recipe == "cub"
    assert args.final_test_checkpoint == "best"
    assert args.metaaugment_policy_learning_rate == pytest.approx(1.0e-3)
    assert args.metaaugment_epsilon == pytest.approx(0.1)
    assert args.metaaugment_inner_learning_rate == pytest.approx(0.05)
    assert args.metaaugment_learn_inner_learning_rate is True
    assert args.metaaugment_sampler_update_epochs == 5
    assert args.metaaugment_sampler_history_epochs == 50
    assert args.metaaugment_cutout_size == 0
    assert args.distributed is True
    assert args.sync_batch_stats is True
    assert args.cross_device_shuffle is False


def test_metaaugment_distributed_mode_requires_sync_batch_stats(
    monkeypatch,
) -> None:
    """Verify PMAP cannot silently change global BatchNorm semantics."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--method",
            "metaaugment",
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


def test_metaaugment_accepts_synchronized_distributed_mode(
    monkeypatch,
) -> None:
    """Verify synchronized PMAP MetaAugment passes argument validation."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--method",
            "metaaugment",
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


def test_metaaugment_distributed_step_keeps_replicas_synchronized() -> None:
    """Exercise the global bilevel update on two forced CPU devices."""
    script = textwrap.dedent(
        """
        import jax
        import jax.numpy as jnp
        import numpy as np
        import flax.linen as nn
        from flax import jax_utils

        from allthemix.competitors.metaaugment.runtime import (
            create_metaaugment_context,
            normalized_policy_weights,
        )
        from allthemix.methods.selector import get_mixer
        from allthemix.networks.batch_norm import batch_norm
        from allthemix.training.engine.parallel.parallel_loop import (
            parallel_train_one_epoch,
        )
        from allthemix.training.engine.single.train import create_train_state

        class TinyBatchNormClassifier(nn.Module):
            '''Small classifier that exercises synchronized running stats.'''

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
                logits = nn.Dense(3, name="head")(features)

                if return_features:
                    return logits, features

                return logits

        assert jax.local_device_count() == 2
        normalize_weights = jax.pmap(
            lambda values: normalized_policy_weights(
                values,
                axis_name="batch",
            ),
            axis_name="batch",
        )
        normalized_weights = normalize_weights(
            np.asarray(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ],
                dtype=np.float32,
            )
        )
        assert np.isclose(np.asarray(normalized_weights).sum(), 1.0)
        batch_size = 4
        model = TinyBatchNormClassifier()
        state = create_train_state(
            rng=jax.random.PRNGKey(0),
            model=model,
            learning_rate=0.01,
            momentum=0.0,
            weight_decay=0.0,
            input_shape=(batch_size, 8, 8, 3),
        )
        images = np.linspace(
            -1.0,
            1.0,
            batch_size * 8 * 8 * 3,
            dtype=np.float32,
        ).reshape(batch_size, 8, 8, 3)
        labels = np.arange(batch_size, dtype=np.int32) % 3
        context = create_metaaugment_context(
            rng=jax.random.PRNGKey(1),
            task_state=state,
            meta_dataset=[(images[::-1].copy(), labels[::-1].copy())],
            input_shape=(batch_size, 8, 8, 3),
            dataset="cifar10",
            num_classes=3,
            policy_learning_rate=0.01,
            policy_momentum=0.0,
            policy_weight_decay=0.0,
            inner_learning_rate=0.01,
            learn_inner_learning_rate=False,
            cutout_size=0,
            sampler_history_epochs=2,
            translate_const=2.0,
            distributed=True,
            sync_batch_stats=True,
        )
        state = jax_utils.replicate(state)
        context.replicate_method_state()
        state, _, loss, _, _ = parallel_train_one_epoch(
            state=state,
            rngs=jax.random.split(jax.random.PRNGKey(2), 2),
            train_ds=[(images, labels)],
            mixer_fn=get_mixer("baseline", 3),
            method="metaaugment",
            num_classes=3,
            max_train_steps=-1,
            sync_batch_stats=True,
            validation_aware_strategy=context,
        )
        task_spread = max(
            float(np.max(np.abs(np.asarray(leaf) - np.asarray(leaf)[0])))
            for leaf in jax.tree_util.tree_leaves(state.params)
        )
        policy_spread = max(
            float(np.max(np.abs(np.asarray(leaf) - np.asarray(leaf)[0])))
            for leaf in jax.tree_util.tree_leaves(
                context.method_state.policy_state.params
            )
        )
        batch_stats_spread = max(
            float(np.max(np.abs(np.asarray(leaf) - np.asarray(leaf)[0])))
            for leaf in jax.tree_util.tree_leaves(state.batch_stats)
        )
        checkpoint = context.checkpoint_state(jax_utils.unreplicate(state))
        pair_count = float(
            np.asarray(checkpoint.method_state.sampler_history)[0, ..., 1].sum()
        )
        assert np.isfinite(loss)
        assert task_spread == 0.0
        assert policy_spread == 0.0
        assert batch_stats_spread == 0.0
        assert int(checkpoint.method_state.epoch) == 1
        assert pair_count == batch_size
        """
    )
    environment = os.environ.copy()
    existing_flags = environment.get(
        "XLA_FLAGS",
        "",
    )
    environment["XLA_FLAGS"] = (
        f"{existing_flags} --xla_force_host_platform_device_count=2"
    ).strip()

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


def test_meta_validation_pipeline_reuses_stratified_split(
    monkeypatch,
) -> None:
    """Verify policy batches come from the shared held-out split and repeat."""
    images = np.zeros(
        (
            20,
            32,
            32,
            3,
        ),
        dtype=np.uint8,
    )
    labels = np.asarray(
        [
            index % 2
            for index in range(
                20,
            )
        ],
        dtype=np.int64,
    )
    raw_dataset = tf.data.Dataset.from_tensor_slices(
        {
            "image": images,
            "label": labels,
        }
    )
    monkeypatch.setattr(
        "allthemix.data.pipeline.load_train_dataset",
        lambda **_: raw_dataset,
    )
    meta_dataset = build_meta_validation_pipeline(
        name="cifar10",
        data_dir="unused",
        batch_size=2,
        validation_split=0.1,
        shuffle_buffer_size=20,
        seed=0,
    )
    iterator = iter(
        meta_dataset,
    )
    first_images, first_labels = next(
        iterator,
    )
    second_images, second_labels = next(
        iterator,
    )

    assert first_images.shape == (
        2,
        32,
        32,
        3,
    )
    assert first_labels.shape == (
        2,
    )
    assert second_images.shape == first_images.shape
    assert second_labels.shape == first_labels.shape
    assert set(
        np.asarray(
            first_labels,
        ).tolist()
    ) == {
        0,
        1,
    }


def test_meta_validation_pipeline_can_emit_one_finite_partial_batch(
    monkeypatch,
) -> None:
    """Allow IF-AugNet to consume each held-out example at most once."""
    images = np.zeros(
        (20, 32, 32, 3),
        dtype=np.uint8,
    )
    labels = np.asarray(
        [index % 2 for index in range(20)],
        dtype=np.int64,
    )
    raw_dataset = tf.data.Dataset.from_tensor_slices(
        {
            "image": images,
            "label": labels,
        }
    )
    monkeypatch.setattr(
        "allthemix.data.pipeline.load_train_dataset",
        lambda **_: raw_dataset,
    )

    meta_dataset = build_meta_validation_pipeline(
        name="cifar10",
        data_dir="unused",
        batch_size=3,
        validation_split=0.1,
        shuffle_buffer_size=20,
        seed=0,
        repeat=False,
        drop_remainder=False,
    )
    batches = list(
        meta_dataset.as_numpy_iterator(),
    )

    assert len(batches) == 1
    assert batches[0][0].shape == (2, 32, 32, 3)
    assert set(batches[0][1].tolist()) == {0, 1}
