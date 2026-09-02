from __future__ import annotations

import hashlib
import os
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
nn = pytest.importorskip("flax.linen")

from allthemix.methods.selector import get_mixer
from allthemix.training.engine.parallel.parallel_train import parallel_train_step
from allthemix.training.engine.single.train import create_train_state
from allthemix.utils.parallel import create_device_rngs, replicate_state, shard_array
from salutary_da.gradient_alignment_strategy import (
    GradientAlignmentBatchStrategy,
    _replicated_direction_geometry,
    _replicated_optimizer_step,
    _summarize_decision,
    make_parallel_mixup_probe,
)
from salutary_da.policies.per_row_continuous import (
    PerRowContinuousPolicyConfig,
    decide_and_summarize_per_row_continuous_device,
    decide_per_row_continuous_device,
)

_CPU4_CHILD_ENV = "ALLTHEMIX_CPU4_GA_TEST_CHILD"


def _assert_array_tree_exact(left, right) -> None:
    """Require matching pytree structure, shapes, dtypes, and array values."""

    assert jax.tree_util.tree_structure(left) == jax.tree_util.tree_structure(right)
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = np.asarray(left_leaf)
        right_array = np.asarray(right_leaf)
        assert left_array.shape == right_array.shape
        assert left_array.dtype == right_array.dtype
        np.testing.assert_array_equal(left_array, right_array)


@pytest.mark.parametrize(
    "config, nonfinite, use_weight_scores",
    [
        (PerRowContinuousPolicyConfig(mode="score_only"), False, False),
        (
            PerRowContinuousPolicyConfig(
                mode="soft_label",
                maximum_rows=1,
                soft_label_dose=0.025,
            ),
            False,
            False,
        ),
        (
            PerRowContinuousPolicyConfig(
                mode="soft_label",
                minimum_gain=1e6,
                fallback_enabled=True,
                fallback_soft_label_dose=0.01,
            ),
            False,
            False,
        ),
        (
            PerRowContinuousPolicyConfig(
                mode="reweight",
                maximum_weight_deviation=0.2,
                minimum_relative_ess=0.9,
            ),
            False,
            True,
        ),
        (
            PerRowContinuousPolicyConfig(
                mode="soft_label",
                fallback_enabled=True,
            ),
            True,
            False,
        ),
    ],
)
def test_fused_policy_matches_independent_decision_and_summary(
    config,
    nonfinite,
    use_weight_scores,
) -> None:
    """Fuse one PMAP without changing any decision or 22-field metric leaf."""

    devices = jax.local_device_count()
    local_rows = 2
    classes = 5
    rows = devices * local_rows
    raw = np.linspace(-0.2, 0.8, rows * classes, dtype=np.float32).reshape(
        devices,
        local_rows,
        classes,
    )
    if nonfinite:
        raw[0, 0, 1] = np.nan
    targets = np.full_like(raw, 1.0 / classes)
    learning_rate = jnp.full((devices,), 0.1, dtype=jnp.float32)
    weight_scores = (
        jnp.asarray(
            np.linspace(-1.0, 1.0, rows, dtype=np.float32).reshape(
                devices,
                local_rows,
            )
        )
        if use_weight_scores
        else None
    )
    legacy_decision = decide_per_row_continuous_device(
        jnp.asarray(raw),
        jnp.asarray(targets),
        learning_rate,
        config,
        sample_weight_scores=weight_scores,
    )
    legacy_summary = _summarize_decision(legacy_decision)
    fused_decision, fused_summary = (
        decide_and_summarize_per_row_continuous_device(
            jnp.asarray(raw),
            jnp.asarray(targets),
            learning_rate,
            config,
            sample_weight_scores=weight_scores,
        )
    )

    _assert_array_tree_exact(fused_decision, legacy_decision)
    assert list(fused_summary) == list(legacy_summary)
    assert len(fused_summary) == 22
    _assert_array_tree_exact(fused_summary, legacy_summary)


class _TinyHead(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(self, features):
        return nn.Dense(self.num_classes)(features)


class _TinyClassifier(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(
        self,
        images,
        *,
        training: bool,
        return_features: bool = False,
        sync_batch_stats: bool = False,
    ):
        features = jnp.mean(images, axis=(1, 2))
        features = nn.BatchNorm(
            use_running_average=not training,
            axis_name="batch" if sync_batch_stats else None,
        )(features)
        logits = _TinyHead(self.num_classes, name="head")(features)
        return (logits, features) if return_features else logits


def _assert_tree_equal(left, right) -> None:
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    assert len(left_leaves) == len(right_leaves)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


def _assert_tree_numerically_equal(left, right) -> None:
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    assert len(left_leaves) == len(right_leaves)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_allclose(
            np.asarray(left_leaf),
            np.asarray(right_leaf),
            rtol=2e-6,
            atol=1e-7,
        )


def _fixture():
    devices = jax.local_device_count()
    batch_size = devices * 2
    classes = 3
    model = _TinyClassifier(num_classes=classes)
    state = create_train_state(
        rng=jax.random.PRNGKey(1),
        model=model,
        learning_rate=0.01,
        momentum=0.9,
        weight_decay=1e-4,
        input_shape=(batch_size, 4, 4, 3),
    )
    images = np.linspace(
        0.0,
        1.0,
        batch_size * 4 * 4 * 3,
        dtype=np.float32,
    ).reshape(batch_size, 4, 4, 3)
    labels = np.arange(batch_size, dtype=np.int32) % classes
    return (
        replicate_state(state),
        state.params,
        images,
        labels,
        get_mixer("mixup", classes, mixup_alpha=0.2),
        create_device_rngs(jax.random.PRNGKey(2)),
        classes,
    )


def _strategy(
    state,
    template_params,
    images,
    labels,
    mixer,
    classes,
    *,
    policy,
    action_enabled,
    base_method="mixup",
    shuffled_control=False,
    validation_direction_mode="full",
    validation_batch_size=500,
    validation_batch_seed=0,
):
    return GradientAlignmentBatchStrategy(
        apply_fn=state.apply_fn,
        template_params=template_params,
        mixer_fn=mixer,
        num_classes=classes,
        validation_images=images,
        validation_labels=labels,
        validation_direction_mode=validation_direction_mode,
        validation_batch_size=validation_batch_size,
        validation_batch_seed=validation_batch_seed,
        learning_rate_fn=lambda step: jnp.full_like(
            step,
            0.01,
            dtype=jnp.float32,
        ),
        policy=policy,
        parameter_scope="full",
        sync_batch_stats=True,
        action_enabled=action_enabled,
        expected_validation_examples=images.shape[0],
        base_method=base_method,
        shuffled_control=shuffled_control,
        control_seed=17,
    )


def _full_schedule_sha256(example_count: int) -> str:
    """Hash the exact materialized-row order used by full validation."""

    rows = np.arange(example_count, dtype="<i8")
    return hashlib.sha256(rows.tobytes()).hexdigest()


def test_replicated_optimizer_step_requires_integer_replica_agreement() -> None:
    """Read a resume step once and fail closed on divergent replicas."""

    devices = jax.local_device_count()
    assert _replicated_optimizer_step(jnp.full((devices,), 17)) == 17
    with pytest.raises(ValueError, match="replicas disagree"):
        _replicated_optimizer_step(jnp.asarray([0, 1], dtype=jnp.int32))
    with pytest.raises(ValueError, match="integers"):
        _replicated_optimizer_step(jnp.ones((devices,), dtype=jnp.float32))


def test_replicated_direction_geometry_uses_one_equal_replica() -> None:
    """Measure exact and orthogonal directions without counting replicas."""

    devices = jax.local_device_count()
    first = {"value": jnp.tile(jnp.asarray([[1.0, 0.0]]), (devices, 1))}
    same = {"value": jnp.tile(jnp.asarray([[1.0, 0.0]]), (devices, 1))}
    orthogonal = {
        "value": jnp.tile(jnp.asarray([[0.0, 1.0]]), (devices, 1))
    }
    cosine, relative_l2 = _replicated_direction_geometry(first, same)
    np.testing.assert_allclose(cosine, 1.0, rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(relative_l2, 0.0, rtol=0.0, atol=1e-7)
    cosine, relative_l2 = _replicated_direction_geometry(first, orthogonal)
    np.testing.assert_allclose(cosine, 0.0, rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(
        relative_l2,
        np.sqrt(2.0),
        rtol=1e-6,
        atol=1e-7,
    )


def test_complete_vdev_strategy_rejects_non_four_device_process(
    monkeypatch,
) -> None:
    """Reject the production 5000-example protocol outside four devices."""

    monkeypatch.setattr(jax, "local_device_count", lambda: 1)
    with pytest.raises(ValueError, match="exactly four local devices"):
        GradientAlignmentBatchStrategy(
            apply_fn=lambda *_args, **_kwargs: None,
            template_params={},
            mixer_fn=lambda *_args, **_kwargs: None,
            num_classes=100,
            validation_images=np.zeros((5_000, 1), dtype=np.float32),
            validation_labels=np.arange(5_000, dtype=np.int32) % 100,
            learning_rate_fn=lambda step: step,
            policy=PerRowContinuousPolicyConfig(mode="score_only"),
            expected_validation_examples=5_000,
        )


def test_default_and_explicit_full_direction_paths_are_identical() -> None:
    """Keep the reviewed full-pool path unchanged behind its explicit mode."""

    state, template_params, images, labels, mixer, rngs, classes = _fixture()
    sharded_images = shard_array(images)
    sharded_labels = shard_array(labels)
    policy = PerRowContinuousPolicyConfig(mode="score_only")
    shared = {
        "apply_fn": state.apply_fn,
        "template_params": template_params,
        "mixer_fn": mixer,
        "num_classes": classes,
        "validation_images": images,
        "validation_labels": labels,
        "learning_rate_fn": lambda step: jnp.full_like(
            step,
            0.01,
            dtype=jnp.float32,
        ),
        "policy": policy,
        "parameter_scope": "full",
        "expected_validation_examples": images.shape[0],
    }
    default_strategy = GradientAlignmentBatchStrategy(**shared)
    explicit_strategy = GradientAlignmentBatchStrategy(
        **shared,
        validation_direction_mode="full",
    )

    default_result = default_strategy.train_step(
        state,
        sharded_images,
        sharded_labels,
        rngs,
    )
    explicit_result = explicit_strategy.train_step(
        state,
        sharded_images,
        sharded_labels,
        rngs,
    )

    _assert_tree_equal(default_result, explicit_result)
    assert default_strategy.execution_summary() == (
        explicit_strategy.execution_summary()
    )
    summary = default_strategy.execution_summary()
    assert summary["direction_refreshes"] == 1
    assert summary["validation_gradient_evaluations"] == 1
    assert summary["validation_exact_reanchors"] == 0
    assert summary["direction_validation_example_visits"] == images.shape[0]


def test_disabled_strategy_is_exact_standard_mixup_no_op() -> None:
    state, template_params, images, labels, mixer, rngs, classes = _fixture()
    sharded_images = shard_array(images)
    sharded_labels = shard_array(labels)
    expected = parallel_train_step(
        state,
        rngs,
        sharded_images,
        sharded_labels,
        {},
        mixer,
        "mixup",
        classes,
        0.5,
        -1.0,
        False,
        False,
        False,
        True,
        False,
    )
    strategy = _strategy(
        state,
        template_params,
        images,
        labels,
        mixer,
        classes,
        policy=PerRowContinuousPolicyConfig(mode="score_only"),
        action_enabled=False,
    )
    actual_state, actual_loss, actual_accuracy, metrics = strategy.train_step(
        state,
        sharded_images,
        sharded_labels,
        rngs,
    )

    _assert_tree_equal(actual_state, expected[0])
    np.testing.assert_array_equal(np.asarray(actual_loss), np.asarray(expected[1]))
    np.testing.assert_array_equal(
        np.asarray(actual_accuracy),
        np.asarray(expected[2]),
    )
    assert metrics == {}
    assert strategy.execution_summary() == {
        "action_enabled": False,
        "parameter_scope": "full",
        "base_method": "mixup",
        "shuffled_control": False,
        "score_start_optimizer_step": 0,
        "score_stop_optimizer_step": None,
        "action_start_optimizer_step": 0,
        "action_stop_optimizer_step": None,
        "validation_direction_mode": "full",
        "validation_pool_examples": images.shape[0],
        "validation_examples_per_gradient_evaluation": images.shape[0],
        "validation_direction_cycle_length": 1,
        "validation_reanchor_interval": None,
        "validation_batch_seed": None,
        "validation_initial_optimizer_step": None,
        "validation_batch_schedule_sha256": _full_schedule_sha256(
            images.shape[0]
        ),
        "train_steps": 1,
        "scored_steps": 0,
        "action_active_steps": 0,
        "direction_refreshes": 0,
        "validation_gradient_evaluations": 0,
        "validation_exact_reanchors": 0,
        "validation_anchor_drift_comparisons": 0,
        "validation_anchor_stale_to_exact_cosine_mean": None,
        "validation_anchor_stale_to_exact_cosine_min": None,
        "validation_anchor_stale_to_exact_relative_l2_mean": None,
        "validation_anchor_stale_to_exact_relative_l2_max": None,
        "direction_validation_example_visits": 0,
    }


def test_origin_action_warmup_is_exact_standard_update_without_ga() -> None:
    """Keep pre-score and post-action origin updates exact to standard ERM."""

    state, template_params, images, labels, _mixer, rngs, classes = _fixture()
    baseline_mixer = get_mixer("baseline", classes)
    sharded_images = shard_array(images)
    sharded_labels = shard_array(labels)
    expected = parallel_train_step(
        state,
        rngs,
        sharded_images,
        sharded_labels,
        {},
        baseline_mixer,
        "baseline",
        classes,
        0.5,
        -1.0,
        False,
        False,
        False,
        True,
        False,
    )
    strategy = GradientAlignmentBatchStrategy(
        apply_fn=state.apply_fn,
        template_params=template_params,
        mixer_fn=baseline_mixer,
        num_classes=classes,
        validation_images=images,
        validation_labels=labels,
        learning_rate_fn=lambda step: jnp.full_like(
            step,
            0.01,
            dtype=jnp.float32,
        ),
        policy=PerRowContinuousPolicyConfig(
            mode="soft_label",
            maximum_rows=images.shape[0],
            soft_label_dose=0.01,
        ),
        parameter_scope="classifier_head",
        sync_batch_stats=True,
        action_enabled=True,
        expected_validation_examples=images.shape[0],
        base_method="baseline",
        score_start_optimizer_step=1,
        score_stop_optimizer_step=3,
        action_start_optimizer_step=2,
        action_stop_optimizer_step=3,
    )

    actual_state, actual_loss, actual_accuracy, metrics = strategy.train_step(
        state,
        sharded_images,
        sharded_labels,
        rngs,
    )

    _assert_tree_equal(actual_state, expected[0])
    np.testing.assert_array_equal(np.asarray(actual_loss), np.asarray(expected[1]))
    np.testing.assert_array_equal(
        np.asarray(actual_accuracy),
        np.asarray(expected[2]),
    )
    assert metrics == {}

    class PositiveOffTargetScorer:
        def __init__(self, direction_scorer):
            self.direction_scorer = direction_scorer

        def distributed_validation_direction_replicated(
            self,
            state,
            batch,
            *,
            verify_count_on_host,
        ):
            return self.direction_scorer.distributed_validation_direction_replicated(
                state,
                batch,
                verify_count_on_host=verify_count_on_host,
            )

        def score_hard_labels_device_replicated(
            self,
            _params,
            _batch_stats,
            batch,
            _direction,
        ):
            return jnp.broadcast_to(
                jnp.arange(1, classes + 1, dtype=jnp.float32),
                batch.soft_targets.shape,
            )

    strategy._scorer = PositiveOffTargetScorer(strategy._scorer)
    scored_state, *_ = strategy.train_step(
        actual_state,
        sharded_images,
        sharded_labels,
        rngs,
    )
    action_state, _, _, action_metrics = strategy.train_step(
        scored_state,
        sharded_images,
        sharded_labels,
        rngs,
    )
    assert float(np.asarray(action_metrics["salda_applied_fraction"])[0]) > 0.0
    assert float(np.asarray(action_metrics["salda_dose_mean"])[0]) > 0.0
    expected_post = parallel_train_step(
        action_state,
        rngs,
        sharded_images,
        sharded_labels,
        {},
        baseline_mixer,
        "baseline",
        classes,
        0.5,
        -1.0,
        False,
        False,
        False,
        True,
        False,
    )
    post_state, post_loss, post_accuracy, post_metrics = strategy.train_step(
        action_state,
        sharded_images,
        sharded_labels,
        rngs,
    )
    _assert_tree_equal(post_state, expected_post[0])
    np.testing.assert_array_equal(np.asarray(post_loss), np.asarray(expected_post[1]))
    np.testing.assert_array_equal(
        np.asarray(post_accuracy),
        np.asarray(expected_post[2]),
    )
    assert post_metrics == {}
    summary = strategy.execution_summary()
    assert summary["train_steps"] == 4
    assert summary["scored_steps"] == 2
    assert summary["action_active_steps"] == 1
    assert summary["direction_refreshes"] == 2


def test_four_device_multistep_noop_preserves_recipe_rng_metrics_and_state() -> None:
    if jax.local_device_count() != 4:
        pytest.skip(
            "distributed parity requires "
            "XLA_FLAGS=--xla_force_host_platform_device_count=4"
        )
    devices = 4
    local_rows = 2
    batch_size = devices * local_rows
    classes = batch_size
    model = _TinyClassifier(num_classes=classes)
    host_state = create_train_state(
        rng=jax.random.PRNGKey(101),
        model=model,
        learning_rate=0.01,
        momentum=0.9,
        weight_decay=1.0e-4,
        input_shape=(batch_size, 4, 4, 3),
    )
    standard_state = replicate_state(host_state)
    strategy_state = replicate_state(host_state)
    images = np.linspace(
        -1.0,
        1.0,
        batch_size * 4 * 4 * 3,
        dtype=np.float32,
    ).reshape(batch_size, 4, 4, 3)
    labels = np.arange(batch_size, dtype=np.int32)
    sharded_images = shard_array(images)
    sharded_labels = shard_array(labels)
    mixer = get_mixer("mixup", classes, mixup_alpha=0.2)
    probe = make_parallel_mixup_probe(mixer, classes)
    strategy = _strategy(
        strategy_state,
        host_state.params,
        images,
        labels,
        mixer,
        classes,
        policy=PerRowContinuousPolicyConfig(mode="score_only"),
        action_enabled=False,
    )
    main_rng = jax.random.PRNGKey(202)
    recipe_history = []
    for step in range(4):
        main_rng, step_rng = jax.random.split(main_rng)
        rngs = create_device_rngs(step_rng)
        recipe = probe(rngs, sharded_images, sharded_labels)
        repeated_recipe = probe(rngs, sharded_images, sharded_labels)
        for observed, repeated in zip(recipe, repeated_recipe, strict=True):
            np.testing.assert_array_equal(np.asarray(observed), np.asarray(repeated))
        mixed, labels_a, labels_b, lambdas, soft_targets, dropout_keys = recipe
        np.testing.assert_array_equal(np.asarray(labels_a), np.asarray(sharded_labels))
        assert np.array_equal(
            np.sort(np.asarray(labels_b), axis=1),
            np.sort(np.asarray(labels_a), axis=1),
        )
        np.testing.assert_allclose(
            np.sum(np.asarray(soft_targets), axis=-1),
            1.0,
            rtol=0.0,
            atol=1.0e-6,
        )
        recipe_history.append(
            tuple(
                np.asarray(value).copy()
                for value in (
                    mixed,
                    labels_a,
                    labels_b,
                    lambdas,
                    soft_targets,
                    dropout_keys,
                )
            )
        )
        expected = parallel_train_step(
            standard_state,
            rngs,
            sharded_images,
            sharded_labels,
            {},
            mixer,
            "mixup",
            classes,
            0.5,
            -1.0,
            False,
            False,
            False,
            True,
            False,
        )
        actual = strategy.train_step(
            strategy_state,
            sharded_images,
            sharded_labels,
            rngs,
        )
        standard_state = expected[0]
        strategy_state = actual[0]
        _assert_tree_equal(strategy_state, standard_state)
        np.testing.assert_array_equal(np.asarray(actual[1]), np.asarray(expected[1]))
        np.testing.assert_array_equal(np.asarray(actual[2]), np.asarray(expected[2]))
        assert actual[3] == {}
    assert len(recipe_history) == 4
    assert not np.array_equal(recipe_history[0][-1], recipe_history[1][-1])
    assert strategy.execution_summary() == {
        "action_enabled": False,
        "parameter_scope": "full",
        "base_method": "mixup",
        "shuffled_control": False,
        "score_start_optimizer_step": 0,
        "score_stop_optimizer_step": None,
        "action_start_optimizer_step": 0,
        "action_stop_optimizer_step": None,
        "validation_direction_mode": "full",
        "validation_pool_examples": images.shape[0],
        "validation_examples_per_gradient_evaluation": images.shape[0],
        "validation_direction_cycle_length": 1,
        "validation_reanchor_interval": None,
        "validation_batch_seed": None,
        "validation_initial_optimizer_step": None,
        "validation_batch_schedule_sha256": _full_schedule_sha256(
            images.shape[0]
        ),
        "train_steps": 4,
        "scored_steps": 0,
        "action_active_steps": 0,
        "direction_refreshes": 0,
        "validation_gradient_evaluations": 0,
        "validation_exact_reanchors": 0,
        "validation_anchor_drift_comparisons": 0,
        "validation_anchor_stale_to_exact_cosine_mean": None,
        "validation_anchor_stale_to_exact_cosine_min": None,
        "validation_anchor_stale_to_exact_relative_l2_mean": None,
        "validation_anchor_stale_to_exact_relative_l2_max": None,
        "direction_validation_example_visits": 0,
    }


def test_validation_batch_aggregate_reanchors_and_wraps_from_step_zero() -> None:
    """Refresh balanced components cyclically and reanchor at one shared state."""

    if os.environ.get(_CPU4_CHILD_ENV) != "1":
        pytest.skip(
            "covered by the isolated four-CPU-device regression test"
        )
    assert jax.default_backend() == "cpu"
    assert jax.local_device_count() == 4
    images = np.arange(5_000, dtype=np.float32).reshape(5_000, 1)
    labels = np.repeat(np.arange(100, dtype=np.int32), 50)
    template_params = {
        "head": {
            "Dense_0": {
                "kernel": jnp.zeros((1, 100), dtype=jnp.float32),
                "bias": jnp.zeros((100,), dtype=jnp.float32),
            }
        }
    }
    strategy = GradientAlignmentBatchStrategy(
        apply_fn=lambda *_args, **_kwargs: None,
        template_params=template_params,
        mixer_fn=lambda *_args, **_kwargs: None,
        num_classes=100,
        validation_images=images,
        validation_labels=labels,
        validation_direction_mode="batch_aggregate",
        validation_batch_size=500,
        validation_batch_seed=19,
        validation_reanchor_interval=50,
        learning_rate_fn=lambda step: jnp.asarray(step, dtype=jnp.float32),
        policy=PerRowContinuousPolicyConfig(mode="score_only"),
        parameter_scope="classifier_head",
        expected_validation_examples=5_000,
        audit_mode=True,
    )

    class FakeScorer:
        """Return a replicated direction encoding batch rows and model step."""

        def __init__(self) -> None:
            """Initialize the ordered batch-call record."""

            self.batches = []

        def distributed_validation_direction_replicated(
            self,
            state,
            batch,
            *,
            verify_count_on_host,
        ):
            """Encode the component mean and replicated optimizer step."""

            assert verify_count_on_host is True
            self.batches.append(batch)
            value = jnp.mean(batch.images) + state.step.astype(jnp.float32)
            return {
                "kernel": jnp.broadcast_to(value[:, None, None], (4, 1, 100)),
                "bias": jnp.broadcast_to(value[:, None], (4, 100)),
            }

    fake_scorer = FakeScorer()
    strategy._scorer = fake_scorer
    state = SimpleNamespace(step=jnp.zeros((4,), dtype=jnp.int32))
    anchored = strategy._validation_direction_for_state(state)
    expected_anchor = np.mean(images)
    np.testing.assert_allclose(
        np.asarray(anchored["bias"]),
        expected_anchor,
        rtol=0.0,
        atol=2e-4,
    )
    assert all(
        observed is expected
        for observed, expected in zip(
            fake_scorer.batches,
            strategy._validation_batch_cycle.batches,
            strict=True,
        )
    )

    for step in range(1, 11):
        strategy._train_steps = step
        state = SimpleNamespace(step=jnp.full((4,), step, dtype=jnp.int32))
        refreshed = strategy._validation_direction_for_state(state)
        assert fake_scorer.batches[-1] is (
            strategy._validation_batch_cycle.batches[step % 10]
        )
        expected_increment = sum(range(1, step + 1)) / 10.0
        np.testing.assert_allclose(
            np.asarray(refreshed["bias"]),
            expected_anchor + expected_increment,
            rtol=0.0,
            atol=2e-4,
        )

    strategy._train_steps = 11
    discontinuous = SimpleNamespace(
        step=jnp.full((4,), 12, dtype=jnp.int32)
    )
    with pytest.raises(ValueError, match="optimizer step is discontinuous"):
        strategy._validation_direction_for_state(discontinuous)

    strategy._train_steps = 50
    state = SimpleNamespace(step=jnp.full((4,), 50, dtype=jnp.int32))
    reanchored = strategy._validation_direction_for_state(state)
    np.testing.assert_allclose(
        np.asarray(reanchored["bias"]),
        expected_anchor + 50.0,
        rtol=0.0,
        atol=2e-4,
    )
    summary = strategy.execution_summary()
    assert summary["validation_direction_mode"] == "batch_aggregate"
    assert summary["validation_pool_examples"] == 5_000
    assert summary["validation_examples_per_gradient_evaluation"] == 500
    assert summary["validation_direction_cycle_length"] == 10
    assert summary["validation_reanchor_interval"] == 50
    assert summary["validation_batch_seed"] == 19
    assert summary["validation_initial_optimizer_step"] == 0
    assert summary["validation_gradient_evaluations"] == 30
    assert summary["validation_exact_reanchors"] == 2
    assert summary["validation_anchor_drift_comparisons"] == 1
    assert np.isfinite(
        summary["validation_anchor_stale_to_exact_cosine_mean"]
    )
    assert np.isfinite(
        summary["validation_anchor_stale_to_exact_relative_l2_max"]
    )
    assert summary["direction_validation_example_visits"] == 15_000
    assert len(summary["validation_batch_schedule_sha256"]) == 64


def test_score_only_strategy_preserves_standard_mixup_update() -> None:
    state, template_params, images, labels, mixer, rngs, classes = _fixture()
    sharded_images = shard_array(images)
    sharded_labels = shard_array(labels)
    expected = parallel_train_step(
        state,
        rngs,
        sharded_images,
        sharded_labels,
        {},
        mixer,
        "mixup",
        classes,
        0.5,
        -1.0,
        False,
        False,
        False,
        True,
        False,
    )
    strategy = _strategy(
        state,
        template_params,
        images,
        labels,
        mixer,
        classes,
        policy=PerRowContinuousPolicyConfig(mode="score_only"),
        action_enabled=True,
    )
    actual_state, actual_loss, actual_accuracy, metrics = strategy.train_step(
        state,
        sharded_images,
        sharded_labels,
        rngs,
    )

    _assert_tree_numerically_equal(actual_state, expected[0])
    np.testing.assert_allclose(
        np.asarray(actual_loss),
        np.asarray(expected[1]),
        rtol=2e-6,
        atol=1e-7,
    )
    np.testing.assert_array_equal(
        np.asarray(actual_accuracy),
        np.asarray(expected[2]),
    )
    assert np.all(np.asarray(metrics["salda_scores_valid"]) == 1.0)


def test_score_only_origin_strategy_preserves_standard_baseline_update() -> None:
    state, template_params, images, labels, _mixer, rngs, classes = _fixture()
    baseline_mixer = get_mixer("baseline", classes)
    sharded_images = shard_array(images)
    sharded_labels = shard_array(labels)
    expected = parallel_train_step(
        state,
        rngs,
        sharded_images,
        sharded_labels,
        {},
        baseline_mixer,
        "baseline",
        classes,
        0.5,
        -1.0,
        False,
        False,
        False,
        True,
        False,
    )
    strategy = _strategy(
        state,
        template_params,
        images,
        labels,
        baseline_mixer,
        classes,
        policy=PerRowContinuousPolicyConfig(mode="score_only"),
        action_enabled=True,
        base_method="baseline",
    )
    actual_state, actual_loss, actual_accuracy, _metrics = strategy.train_step(
        state,
        sharded_images,
        sharded_labels,
        rngs,
    )
    _assert_tree_equal(actual_state, expected[0])
    np.testing.assert_array_equal(np.asarray(actual_loss), np.asarray(expected[1]))
    np.testing.assert_array_equal(
        np.asarray(actual_accuracy),
        np.asarray(expected[2]),
    )


def test_shuffled_reweight_control_preserves_global_weight_contract() -> None:
    state, template_params, images, labels, mixer, rngs, classes = _fixture()
    strategy = _strategy(
        state,
        template_params,
        images,
        labels,
        mixer,
        classes,
        policy=PerRowContinuousPolicyConfig(
            mode="reweight",
            maximum_weight_deviation=0.2,
            minimum_relative_ess=0.9,
        ),
        action_enabled=True,
        shuffled_control=True,
    )
    new_state, loss, _accuracy, metrics = strategy.train_step(
        state,
        shard_array(images),
        shard_array(labels),
        rngs,
    )
    np.testing.assert_allclose(np.asarray(metrics["salda_weight_mean"]), 1.0, atol=2e-6)
    assert np.all(np.asarray(metrics["salda_weight_min"]) >= 0.8 - 1e-6)
    assert np.all(np.asarray(metrics["salda_weight_max"]) <= 1.2 + 1e-6)
    assert np.all(np.asarray(metrics["salda_weight_relative_ess"]) >= 0.9 - 1e-6)
    assert np.all(np.isfinite(np.asarray(loss)))
    assert int(np.asarray(new_state.step)[0]) == 1


def test_soft_label_fallback_applies_one_continuous_action() -> None:
    state, template_params, images, labels, mixer, rngs, classes = _fixture()
    strategy = _strategy(
        state,
        template_params,
        images,
        labels,
        mixer,
        classes,
        policy=PerRowContinuousPolicyConfig(
            mode="soft_label",
            minimum_gain=1e6,
            fallback_enabled=True,
            fallback_soft_label_dose=0.01,
        ),
        action_enabled=True,
    )
    new_state, loss, accuracy, metrics = strategy.train_step(
        state,
        shard_array(images),
        shard_array(labels),
        rngs,
    )

    assert np.all(np.asarray(metrics["salda_batch_action_coverage"]) == 1.0)
    assert np.all(np.asarray(metrics["salda_fallback_fraction"]) == 1.0)
    assert np.all(np.isfinite(np.asarray(loss)))
    assert np.all(np.isfinite(np.asarray(accuracy)))
    assert int(np.asarray(new_state.step)[0]) == 1
