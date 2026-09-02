from __future__ import annotations

import os
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import jax_utils
from flax.core import freeze

from salutary_da.scorers.gradient_alignment import (
    ClassifierHeadGradientAlignmentScorer,
    FullParameterGradientAlignmentScorer,
    prepare_stratified_validation_batch_cycle,
    prepare_validation_batch,
    relative_hard_label_gains_from_tangent,
)

_CPU4_CHILD_ENV = "ALLTHEMIX_CPU4_GA_TEST_CHILD"
_CPU4_ONLY = pytest.mark.skipif(
    os.environ.get(_CPU4_CHILD_ENV) != "1",
    reason="covered by the isolated four-CPU-device regression test",
)


def _require_four_cpu_devices() -> int:
    """Fail closed unless the isolated test process exposes four CPU devices."""

    assert os.environ.get(_CPU4_CHILD_ENV) == "1"
    assert jax.default_backend() == "cpu"
    assert jax.local_device_count() == 4
    return 4


def _assert_tree_allclose(left, right, *, rtol=2e-6, atol=2e-7) -> None:
    """Compare equal PyTree structures leaf by leaf."""

    assert jax.tree_util.tree_structure(left) == jax.tree_util.tree_structure(right)
    for left_leaf, right_leaf in zip(
        jax.tree_util.tree_leaves(left),
        jax.tree_util.tree_leaves(right),
        strict=True,
    ):
        np.testing.assert_allclose(
            np.asarray(left_leaf),
            np.asarray(right_leaf),
            rtol=rtol,
            atol=atol,
        )


def _assert_all_replicas_allclose(
    replicated,
    expected,
    *,
    rtol=2e-6,
    atol=2e-7,
) -> None:
    """Compare every PMAP replica with one external reference PyTree."""

    for replica in range(jax.local_device_count()):
        observed = jax.tree_util.tree_map(
            lambda value, replica_index=replica: value[replica_index],
            replicated,
        )
        _assert_tree_allclose(observed, expected, rtol=rtol, atol=atol)


def test_exhaustive_100_label_scores_have_registered_shape_and_are_finite() -> None:
    """Materialize every hard-label score for a sharded 100-class batch."""

    devices = jax.local_device_count()
    tangent = jnp.linspace(
        -2.0,
        3.0,
        devices * 2 * 100,
        dtype=jnp.float32,
    ).reshape(devices, 2, 100)
    targets = jnp.full_like(tangent, 0.01)
    gains = relative_hard_label_gains_from_tangent(tangent, targets)

    assert gains.shape == (devices, 2, 100)
    assert np.all(np.isfinite(np.asarray(gains)))
    expected = jnp.sum(targets * tangent, axis=-1, keepdims=True) - tangent
    np.testing.assert_array_equal(np.asarray(gains), np.asarray(expected))


@_CPU4_ONLY
def test_prepare_validation_batch_shards_all_5000_examples_once() -> None:
    """Materialize complete Vdev with no chunk, padding, or mask axis."""

    devices = _require_four_cpu_devices()
    images = np.arange(5_000, dtype=np.float32).reshape(5_000, 1)
    labels = np.arange(5_000, dtype=np.int32) % 100
    batch = prepare_validation_batch(
        images,
        labels,
        num_devices=devices,
    )

    assert batch.images.shape == (devices, 5_000 // devices, 1)
    assert batch.labels.shape == (devices, 5_000 // devices)
    assert batch.example_count == 5_000
    assert batch.local_batch_size == 5_000 // devices
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(batch.images)).reshape(images.shape),
        images,
    )
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(batch.labels)).reshape(labels.shape),
        labels,
    )


@_CPU4_ONLY
def test_prepare_validation_batch_shards_all_stl10_4000_examples_once() -> None:
    """Shard the registered STL-10 Vdev as exactly 1000 rows per device."""

    devices = _require_four_cpu_devices()
    images = np.arange(4_000 * 3, dtype=np.float32).reshape(4_000, 1, 1, 3)
    labels = np.repeat(np.arange(10, dtype=np.int32), 400)
    batch = prepare_validation_batch(images, labels, num_devices=devices)

    assert batch.images.shape == (devices, 1_000, 1, 1, 3)
    assert batch.labels.shape == (devices, 1_000)
    assert batch.example_count == 4_000
    assert batch.local_batch_size == 1_000
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(batch.images)).reshape(images.shape),
        images,
    )
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(batch.labels)).reshape(labels.shape),
        labels,
    )
    assert set(batch.__dataclass_fields__) == {
        "images",
        "labels",
        "example_count",
        "local_batch_size",
        "num_devices",
    }


def test_prepare_validation_batch_rejects_padding_requirement() -> None:
    """Fail instead of reintroducing padding for an uneven device split."""

    with pytest.raises(ValueError, match="divide evenly across devices"):
        prepare_validation_batch(
            np.zeros((3, 1), dtype=np.float32),
            np.zeros((3,), dtype=np.int32),
            num_devices=2,
        )

    with pytest.raises(ValueError, match="one-dimensional"):
        prepare_validation_batch(
            np.zeros((4, 1), dtype=np.float32),
            np.zeros((4, 1), dtype=np.int32),
            num_devices=jax.local_device_count(),
        )

    with pytest.raises(ValueError, match="num_devices must be positive"):
        prepare_stratified_validation_batch_cycle(
            np.zeros((200, 1), dtype=np.float32),
            np.repeat(np.arange(2, dtype=np.int32), 100),
            num_classes=2,
            global_batch_size=100,
            seed=0,
            num_devices=0,
        )


@_CPU4_ONLY
def test_stratified_validation_batch_cycle_matches_external_gradient() -> None:
    """Cover complete Vdev in balanced batches and match one batch gradient."""

    devices = _require_four_cpu_devices()
    class_count = 100
    validation_count = 5_000
    feature_count = 2
    validation_images = np.linspace(
        -0.9,
        0.8,
        validation_count * feature_count,
        dtype=np.float32,
    ).reshape(validation_count, feature_count)
    validation_labels = np.repeat(
        np.arange(class_count, dtype=np.int32),
        validation_count // class_count,
    )
    cycle = prepare_stratified_validation_batch_cycle(
        validation_images,
        validation_labels,
        num_classes=class_count,
        global_batch_size=500,
        seed=17,
        num_devices=devices,
    )
    repeated = prepare_stratified_validation_batch_cycle(
        validation_images,
        validation_labels,
        num_classes=class_count,
        global_batch_size=500,
        seed=17,
        num_devices=devices,
    )
    changed_seed = prepare_stratified_validation_batch_cycle(
        validation_images,
        validation_labels,
        num_classes=class_count,
        global_batch_size=500,
        seed=18,
        num_devices=devices,
    )

    assert cycle.example_count == 5_000
    assert cycle.batch_size == 500
    assert cycle.local_batch_size == 125
    assert cycle.cycle_length == 10
    assert cycle.examples_per_class_per_batch == 5
    assert len(cycle.schedule_sha256) == 64
    assert cycle.index_batches.flags.writeable is False
    np.testing.assert_array_equal(cycle.index_batches, repeated.index_batches)
    assert cycle.schedule_sha256 == repeated.schedule_sha256
    assert not np.array_equal(cycle.index_batches, changed_seed.index_batches)
    assert cycle.schedule_sha256 != changed_seed.schedule_sha256
    np.testing.assert_array_equal(
        np.sort(cycle.index_batches.reshape(-1)),
        np.arange(validation_count),
    )
    for batch, batch_indices in zip(
        cycle.batches,
        cycle.index_batches,
        strict=True,
    ):
        assert batch.images.shape == (devices, 125, feature_count)
        assert batch.labels.shape == (devices, 125)
        np.testing.assert_array_equal(
            np.bincount(
                np.asarray(jax.device_get(batch.labels)).reshape(-1),
                minlength=class_count,
            ),
            np.full((class_count,), 5),
        )
        np.testing.assert_array_equal(
            np.asarray(jax.device_get(batch.images)).reshape(500, feature_count),
            validation_images[batch_indices],
        )

    params = {
        "kernel": jnp.linspace(
            -0.2,
            0.3,
            feature_count * class_count,
            dtype=jnp.float32,
        ).reshape(feature_count, class_count),
        "bias": jnp.linspace(-0.1, 0.1, class_count, dtype=jnp.float32),
    }
    batch_stats = {"offset": jnp.zeros((class_count,), dtype=jnp.float32)}

    def apply_fn(variables, images, *, training):
        """Apply the independent affine validation model."""

        assert training is False
        return (
            images @ variables["params"]["kernel"]
            + variables["params"]["bias"]
            + variables["batch_stats"]["offset"]
        )

    scorer = FullParameterGradientAlignmentScorer(
        apply_fn=apply_fn,
        template_params=params,
    )
    state = SimpleNamespace(
        params=jax_utils.replicate(params),
        batch_stats=jax_utils.replicate(batch_stats),
    )

    def external_component_loss(value, component_images, component_labels):
        """Return an independent global-mean loss for one fixed component."""

        logits = apply_fn(
            {"params": value, "batch_stats": batch_stats},
            component_images,
            training=False,
        )
        targets = jax.nn.one_hot(component_labels, class_count)
        return -jnp.mean(
            jnp.sum(targets * jax.nn.log_softmax(logits), axis=-1)
        )

    external_component_gradient = jax.grad(external_component_loss)
    component_directions = []
    for component_batch, component_indices in zip(
        cycle.batches,
        cycle.index_batches,
        strict=True,
    ):
        replicated_direction = (
            scorer.distributed_validation_direction_replicated(
                state,
                component_batch,
            )
        )

        external_direction = external_component_gradient(
            params,
            jnp.asarray(validation_images[component_indices]),
            jnp.asarray(validation_labels[component_indices]),
        )
        _assert_tree_allclose(
            jax_utils.unreplicate(replicated_direction),
            external_direction,
        )
        _assert_all_replicas_allclose(
            replicated_direction,
            external_direction,
        )
        component_directions.append(replicated_direction)

    aggregate_direction = jax.tree_util.tree_map(
        lambda *leaves: sum(leaves[1:], leaves[0]) / len(leaves),
        *component_directions,
    )
    full_batch = prepare_validation_batch(
        validation_images,
        validation_labels,
        num_devices=devices,
    )
    full_direction = scorer.distributed_validation_direction_replicated(
        state,
        full_batch,
    )
    _assert_tree_allclose(
        aggregate_direction,
        full_direction,
        rtol=2e-5,
        atol=2e-7,
    )
    _assert_all_replicas_allclose(
        aggregate_direction,
        jax_utils.unreplicate(full_direction),
        rtol=2e-5,
        atol=2e-7,
    )


@_CPU4_ONLY
def test_full_ga_matches_external_validation_gradient_and_training_jvp() -> None:
    """Check the vanilla full-Vdev gradient and production training-mode JVP."""

    devices = _require_four_cpu_devices()
    local_rows = 2
    feature_count = 3
    class_count = 4
    params = {
        "kernel": jnp.linspace(
            -0.4,
            0.5,
            feature_count * class_count,
            dtype=jnp.float32,
        ).reshape(feature_count, class_count),
        "bias": jnp.linspace(-0.1, 0.2, class_count, dtype=jnp.float32),
    }
    batch_stats = {"offset": jnp.zeros((class_count,), dtype=jnp.float32)}

    def apply_fn(
        variables,
        images,
        *,
        training,
        mutable=None,
        rngs=None,
        sync_batch_stats=False,
    ):
        logits = (
            images @ variables["params"]["kernel"]
            + variables["params"]["bias"]
            + variables["batch_stats"]["offset"]
        )
        if training:
            assert mutable == ["batch_stats"]
            assert rngs is not None and set(rngs) == {"dropout"}
            assert sync_batch_stats is True
            return logits, {"batch_stats": variables["batch_stats"]}
        assert mutable is None
        assert rngs is None
        assert sync_batch_stats is False
        return logits

    validation_count = 5_000
    validation_images = np.linspace(
        -0.8,
        0.7,
        validation_count * feature_count,
        dtype=np.float32,
    ).reshape(validation_count, feature_count)
    validation_labels = np.arange(validation_count, dtype=np.int32) % class_count
    batch = prepare_validation_batch(
        validation_images,
        validation_labels,
        num_devices=devices,
    )
    scorer = FullParameterGradientAlignmentScorer(
        apply_fn=apply_fn,
        template_params=params,
    )
    state = SimpleNamespace(
        params=jax_utils.replicate(params),
        batch_stats=jax_utils.replicate(batch_stats),
    )
    params_before = jax.tree_util.tree_map(
        lambda value: np.asarray(value).copy(),
        state.params,
    )
    batch_stats_before = jax.tree_util.tree_map(
        lambda value: np.asarray(value).copy(),
        state.batch_stats,
    )
    replicated_direction = scorer.distributed_validation_direction_replicated(
        state,
        batch,
    )
    direction = jax_utils.unreplicate(replicated_direction)

    def external_validation_loss(value):
        logits = apply_fn(
            {"params": value, "batch_stats": batch_stats},
            jnp.asarray(validation_images),
            training=False,
        )
        targets = jax.nn.one_hot(jnp.asarray(validation_labels), class_count)
        return -jnp.mean(jnp.sum(targets * jax.nn.log_softmax(logits), axis=-1))

    external_direction = jax.grad(external_validation_loss)(params)
    assert batch.example_count == validation_count
    assert batch.local_batch_size == validation_count // devices
    _assert_tree_allclose(direction, external_direction)
    _assert_all_replicas_allclose(replicated_direction, external_direction)
    _assert_tree_allclose(state.params, params_before, rtol=0.0, atol=0.0)
    _assert_tree_allclose(
        state.batch_stats,
        batch_stats_before,
        rtol=0.0,
        atol=0.0,
    )

    images = jnp.linspace(
        -0.5,
        0.9,
        devices * local_rows * feature_count,
        dtype=jnp.float32,
    ).reshape(devices, local_rows, feature_count)
    targets = jnp.full(
        (devices, local_rows, class_count),
        1.0 / class_count,
        dtype=jnp.float32,
    )
    dropout_keys = jax.random.split(jax.random.PRNGKey(17), devices)
    batch = SimpleNamespace(
        images=images,
        soft_targets=targets,
        dropout_keys=dropout_keys,
    )
    observed, observed_utility = (
        scorer.score_labels_and_sample_weights_device_replicated(
            state.params,
            state.batch_stats,
            batch,
            replicated_direction,
        )
    )
    logits = []
    tangents = []
    for device in range(devices):
        def training_logits(value, *, device_index=device):
            result, _ = apply_fn(
                {"params": value, "batch_stats": batch_stats},
                images[device_index],
                training=True,
                mutable=["batch_stats"],
                rngs={"dropout": dropout_keys[device_index]},
                sync_batch_stats=True,
            )
            return result

        primal, tangent = jax.jvp(
            training_logits,
            (params,),
            (external_direction,),
        )
        logits.append(primal)
        tangents.append(tangent)
    logits = jnp.stack(logits)
    tangents = jnp.stack(tangents)
    expected = relative_hard_label_gains_from_tangent(tangents, targets)
    expected_utility = jnp.sum(
        (jax.nn.softmax(logits, axis=-1) - targets) * tangents,
        axis=-1,
    )
    np.testing.assert_allclose(observed, expected, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(
        observed_utility,
        expected_utility,
        rtol=2e-6,
        atol=2e-7,
    )


@_CPU4_ONLY
def test_classifier_head_ga_matches_external_head_and_full_component() -> None:
    """Check the affine head gradient, full-gradient component, and scores."""

    devices = _require_four_cpu_devices()
    local_rows = 2
    input_width = 3
    feature_width = 5
    class_count = 4
    params = freeze(
        {
            "backbone": {
                "projection": jnp.linspace(
                    -0.5,
                    0.4,
                    input_width * feature_width,
                    dtype=jnp.float32,
                ).reshape(input_width, feature_width)
            },
            "head": {
                "Dense_0": {
                    "kernel": jnp.linspace(
                        -0.3,
                        0.6,
                        feature_width * class_count,
                        dtype=jnp.float32,
                    ).reshape(feature_width, class_count),
                    "bias": jnp.linspace(-0.1, 0.1, class_count),
                }
            },
        }
    )
    batch_stats = freeze({"unused": jnp.asarray(0.0, dtype=jnp.float32)})

    def apply_fn(
        variables,
        images,
        *,
        training,
        return_features=False,
        mutable=None,
        rngs=None,
        sync_batch_stats=False,
    ):
        features = jnp.tanh(
            images @ variables["params"]["backbone"]["projection"]
        )
        dense = variables["params"]["head"]["Dense_0"]
        logits = features @ dense["kernel"] + dense["bias"]
        output = (logits, features) if return_features else logits
        if training:
            assert mutable == ["batch_stats"]
            assert rngs is not None and set(rngs) == {"dropout"}
            assert sync_batch_stats is True
            return output, {"batch_stats": variables["batch_stats"]}
        assert mutable is None
        assert rngs is None
        assert sync_batch_stats is False
        return output

    validation_count = 5_000
    validation_images = np.linspace(
        -0.7,
        0.8,
        validation_count * input_width,
        dtype=np.float32,
    ).reshape(validation_count, input_width)
    validation_labels = np.arange(validation_count, dtype=np.int32) % class_count
    batch = prepare_validation_batch(
        validation_images,
        validation_labels,
        num_devices=devices,
    )
    state = SimpleNamespace(
        params=jax_utils.replicate(params),
        batch_stats=jax_utils.replicate(batch_stats),
    )
    params_before = jax.tree_util.tree_map(
        lambda value: np.asarray(value).copy(),
        state.params,
    )
    batch_stats_before = jax.tree_util.tree_map(
        lambda value: np.asarray(value).copy(),
        state.batch_stats,
    )
    head_scorer = ClassifierHeadGradientAlignmentScorer(
        apply_fn=apply_fn,
        template_params=params,
    )
    head_replicated = head_scorer.distributed_validation_direction_replicated(
        state,
        batch,
    )
    head_direction = jax_utils.unreplicate(head_replicated)
    dense = params["head"]["Dense_0"]
    _, validation_features = apply_fn(
        {"params": params, "batch_stats": batch_stats},
        jnp.asarray(validation_images),
        training=False,
        return_features=True,
    )

    def external_head_loss(value):
        logits = validation_features @ value["kernel"] + value["bias"]
        targets = jax.nn.one_hot(jnp.asarray(validation_labels), class_count)
        return -jnp.mean(jnp.sum(targets * jax.nn.log_softmax(logits), axis=-1))

    external_head = jax.grad(external_head_loss)(dense)
    _assert_tree_allclose(head_direction, external_head)
    _assert_all_replicas_allclose(head_replicated, external_head)
    _assert_tree_allclose(state.params, params_before, rtol=0.0, atol=0.0)
    _assert_tree_allclose(
        state.batch_stats,
        batch_stats_before,
        rtol=0.0,
        atol=0.0,
    )

    full_scorer = FullParameterGradientAlignmentScorer(
        apply_fn=apply_fn,
        template_params=params,
    )
    full_direction = jax_utils.unreplicate(
        full_scorer.distributed_validation_direction_replicated(state, batch)
    )
    _assert_tree_allclose(full_direction["head"]["Dense_0"], head_direction)

    cycle = prepare_stratified_validation_batch_cycle(
        validation_images,
        validation_labels,
        num_classes=class_count,
        global_batch_size=500,
        seed=31,
        num_devices=devices,
    )

    def external_component_head_loss(
        value,
        component_features,
        component_labels,
    ):
        """Return the independent affine-head loss for one component."""

        logits = component_features @ value["kernel"] + value["bias"]
        targets = jax.nn.one_hot(component_labels, class_count)
        return -jnp.mean(
            jnp.sum(targets * jax.nn.log_softmax(logits), axis=-1)
        )

    external_component_head_gradient = jax.grad(
        external_component_head_loss
    )
    head_components = []
    for component_batch, component_indices in zip(
        cycle.batches,
        cycle.index_batches,
        strict=True,
    ):
        component_direction = (
            head_scorer.distributed_validation_direction_replicated(
                state,
                component_batch,
            )
        )
        component_features = validation_features[component_indices]
        component_labels = validation_labels[component_indices]

        external_component = external_component_head_gradient(
            dense,
            component_features,
            jnp.asarray(component_labels),
        )
        _assert_tree_allclose(
            jax_utils.unreplicate(component_direction),
            external_component,
        )
        _assert_all_replicas_allclose(
            component_direction,
            external_component,
        )
        head_components.append(component_direction)

    aggregate_head = jax.tree_util.tree_map(
        lambda *leaves: sum(leaves[1:], leaves[0]) / len(leaves),
        *head_components,
    )
    _assert_tree_allclose(
        aggregate_head,
        head_replicated,
        rtol=2e-5,
        atol=2e-7,
    )
    _assert_all_replicas_allclose(
        aggregate_head,
        head_direction,
        rtol=2e-5,
        atol=2e-7,
    )

    images = jnp.linspace(
        -0.6,
        0.9,
        devices * local_rows * input_width,
        dtype=jnp.float32,
    ).reshape(devices, local_rows, input_width)
    targets = jnp.full(
        (devices, local_rows, class_count),
        1.0 / class_count,
        dtype=jnp.float32,
    )
    dropout_keys = jax.random.split(jax.random.PRNGKey(29), devices)
    batch = SimpleNamespace(
        images=images,
        soft_targets=targets,
        dropout_keys=dropout_keys,
    )
    observed, observed_utility = (
        head_scorer.score_labels_and_sample_weights_device_replicated(
            state.params,
            state.batch_stats,
            batch,
            head_replicated,
        )
    )
    aggregate_observed, aggregate_utility = (
        head_scorer.score_labels_and_sample_weights_device_replicated(
            state.params,
            state.batch_stats,
            batch,
            aggregate_head,
        )
    )
    np.testing.assert_allclose(
        aggregate_observed,
        observed,
        rtol=2e-5,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        aggregate_utility,
        observed_utility,
        rtol=2e-5,
        atol=2e-7,
    )
    expected_logits = []
    expected_tangents = []
    for device in range(devices):
        (logits, features), _ = apply_fn(
            {"params": params, "batch_stats": batch_stats},
            images[device],
            training=True,
            return_features=True,
            mutable=["batch_stats"],
            rngs={"dropout": dropout_keys[device]},
            sync_batch_stats=True,
        )
        expected_logits.append(logits)
        expected_tangents.append(
            features @ external_head["kernel"] + external_head["bias"]
        )
    expected_logits = jnp.stack(expected_logits)
    expected_tangents = jnp.stack(expected_tangents)
    expected = relative_hard_label_gains_from_tangent(
        expected_tangents,
        targets,
    )
    expected_utility = jnp.sum(
        (jax.nn.softmax(expected_logits, axis=-1) - targets)
        * expected_tangents,
        axis=-1,
    )
    np.testing.assert_allclose(observed, expected, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(
        observed_utility,
        expected_utility,
        rtol=2e-6,
        atol=2e-7,
    )


def test_classifier_head_ga_rejects_noncanonical_head_layout() -> None:
    """Fail closed when the model has no exact head/Dense_0 affine layer."""

    params = {"head": {"Dense_1": {"kernel": jnp.ones((2, 3))}}}
    with pytest.raises(ValueError, match="Dense_0"):
        ClassifierHeadGradientAlignmentScorer(
            apply_fn=lambda *_args, **_kwargs: None,
            template_params=params,
        )
