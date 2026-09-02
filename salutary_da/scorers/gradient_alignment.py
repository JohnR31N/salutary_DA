"""Device-resident gradient-alignment scoring for instantaneous GA."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import jax_utils


# #### GA HARD-LABEL GAIN PROJECTION: START ####
def relative_hard_label_gains_from_tangent(
    logit_tangent: np.ndarray | jax.Array,
    soft_targets: np.ndarray | jax.Array,
) -> jax.Array:
    """Return every hard-label GA score relative to the current target.

    For validation-gradient direction ``u`` and training-logit Jacobian ``J``,
    replacing target ``t`` by class ``c`` has first-order gain
    ``(t - one_hot(c)) @ J u``.
    """

    tangent = jnp.asarray(logit_tangent)
    targets = jnp.asarray(soft_targets)
    if tangent.shape != targets.shape or tangent.ndim not in (2, 3):
        raise ValueError(
            "logit_tangent and soft_targets must share shape [N, C] or [D, L, C]"
        )
    soft_projection = jnp.sum(targets * tangent, axis=-1, keepdims=True)
    return soft_projection - tangent
# #### GA HARD-LABEL GAIN PROJECTION: END ####


@dataclass(frozen=True)
class PreparedValidationBatch:
    """The complete Vdev split sharded once across all local devices."""

    images: jax.Array
    labels: jax.Array
    example_count: int
    local_batch_size: int
    num_devices: int


@dataclass(frozen=True)
class PreparedValidationBatchCycle:
    """A class-balanced no-replacement cycle over the complete Vdev pool."""

    batches: tuple[PreparedValidationBatch, ...]
    index_batches: np.ndarray
    example_count: int
    batch_size: int
    local_batch_size: int
    cycle_length: int
    num_devices: int
    num_classes: int
    examples_per_class_per_batch: int
    seed: int
    schedule_sha256: str


def prepare_validation_batch(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    num_devices: int,
) -> PreparedValidationBatch:
    """Shard the complete validation split as one vanilla PMAP batch."""

    images = np.asarray(images)
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("validation labels must be a one-dimensional array")
    if images.shape[0] == 0 or images.shape[0] != labels.shape[0]:
        raise ValueError("validation images and labels must be non-empty and aligned")
    if isinstance(num_devices, bool) or not isinstance(
        num_devices,
        (int, np.integer),
    ):
        raise TypeError("num_devices must be an integer")
    if num_devices <= 0:
        raise ValueError("num_devices must be positive")
    if labels.shape[0] % num_devices:
        raise ValueError(
            "the complete validation split must divide evenly across devices"
        )
    devices = jax.local_devices()
    if len(devices) != num_devices:
        raise ValueError(
            f"requested {num_devices} devices but JAX exposes {len(devices)}"
        )
    local_batch_size = labels.shape[0] // num_devices

    def shard(value: np.ndarray) -> jax.Array:
        local = value.reshape(
            (num_devices, local_batch_size, *value.shape[1:])
        )
        return jax.device_put_sharded(
            [jnp.asarray(part) for part in local],
            devices,
        )

    return PreparedValidationBatch(
        images=shard(images),
        labels=shard(labels),
        example_count=int(labels.shape[0]),
        local_batch_size=int(local_batch_size),
        num_devices=int(num_devices),
    )


def prepare_stratified_validation_batch_cycle(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    num_classes: int,
    global_batch_size: int,
    seed: int,
    num_devices: int,
) -> PreparedValidationBatchCycle:
    """Prepare balanced Vdev mini-batches that cover the full pool once."""

    images = np.asarray(images)
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("validation labels must be a one-dimensional array")
    if images.shape[0] == 0 or images.shape[0] != labels.shape[0]:
        raise ValueError("validation images and labels must be non-empty and aligned")
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError("validation labels must have an integer dtype")
    if isinstance(num_classes, bool) or not isinstance(
        num_classes,
        (int, np.integer),
    ):
        raise TypeError("num_classes must be an integer")
    if num_classes < 2:
        raise ValueError("num_classes must be at least two")
    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError("validation labels must lie within the class range")
    class_counts = np.bincount(labels, minlength=num_classes)
    if class_counts.shape != (num_classes,) or np.any(
        class_counts != class_counts[0]
    ):
        raise ValueError("validation batch cycling requires equal class counts")
    if isinstance(global_batch_size, bool) or not isinstance(
        global_batch_size,
        (int, np.integer),
    ):
        raise TypeError("validation batch size must be an integer")
    if global_batch_size <= 0 or global_batch_size >= labels.shape[0]:
        raise ValueError(
            "validation batch size must be positive and smaller than Vdev"
        )
    if isinstance(num_devices, bool) or not isinstance(
        num_devices,
        (int, np.integer),
    ):
        raise TypeError("num_devices must be an integer")
    if num_devices <= 0:
        raise ValueError("num_devices must be positive")
    if global_batch_size % num_devices:
        raise ValueError("validation batch size must divide evenly across devices")
    if global_batch_size % num_classes:
        raise ValueError("validation batch size must divide evenly across classes")
    if labels.shape[0] % global_batch_size:
        raise ValueError("validation batch size must divide the complete Vdev pool")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("validation batch seed must be an integer")

    cycle_length = labels.shape[0] // global_batch_size
    examples_per_class = global_batch_size // num_classes
    if class_counts[0] != cycle_length * examples_per_class:
        raise AssertionError("balanced validation cycle arithmetic changed")

    rng = np.random.default_rng(int(seed))
    class_batches = []
    for class_index in range(num_classes):
        class_rows = np.flatnonzero(labels == class_index)
        class_rows = rng.permutation(class_rows)
        class_batches.append(
            class_rows.reshape(cycle_length, examples_per_class)
        )

    index_batches = []
    for cycle_index in range(cycle_length):
        batch_indices = np.concatenate(
            [rows[cycle_index] for rows in class_batches]
        )
        batch_indices = rng.permutation(batch_indices)
        index_batches.append(batch_indices)
    index_batches = np.stack(index_batches).astype(np.int64, copy=False)

    flattened = index_batches.reshape(-1)
    if not np.array_equal(np.sort(flattened), np.arange(labels.shape[0])):
        raise AssertionError("validation batch cycle must cover every row exactly once")
    expected_class_counts = np.full(
        (num_classes,),
        examples_per_class,
        dtype=np.int64,
    )
    for batch_indices in index_batches:
        if not np.array_equal(
            np.bincount(labels[batch_indices], minlength=num_classes),
            expected_class_counts,
        ):
            raise AssertionError("validation mini-batch lost class balance")

    index_batches.setflags(write=False)
    schedule_bytes = index_batches.astype("<i8", copy=False).tobytes()
    batches = tuple(
        prepare_validation_batch(
            images[batch_indices],
            labels[batch_indices],
            num_devices=num_devices,
        )
        for batch_indices in index_batches
    )
    return PreparedValidationBatchCycle(
        batches=batches,
        index_batches=index_batches,
        example_count=int(labels.shape[0]),
        batch_size=int(global_batch_size),
        local_batch_size=int(global_batch_size // num_devices),
        cycle_length=int(cycle_length),
        num_devices=int(num_devices),
        num_classes=int(num_classes),
        examples_per_class_per_batch=int(examples_per_class),
        seed=int(seed),
        schedule_sha256=hashlib.sha256(schedule_bytes).hexdigest(),
    )


def _validated_classifier_head(params):
    """Return the exact supported ``head/Dense_0/{kernel,bias}`` subtree."""

    if not isinstance(params, Mapping) or "head" not in params:
        raise ValueError("classifier-head GA requires params['head']")
    head = params["head"]
    if not isinstance(head, Mapping) or set(head) != {"Dense_0"}:
        raise ValueError(
            "classifier-head GA requires the exact params['head']['Dense_0'] layout"
        )
    dense = head["Dense_0"]
    if not isinstance(dense, Mapping) or set(dense) != {"bias", "kernel"}:
        raise ValueError(
            "classifier-head GA requires exactly head/Dense_0/{bias,kernel}"
        )
    kernel = dense["kernel"]
    bias = dense["bias"]
    if kernel.ndim != 2 or bias.ndim != 1 or kernel.shape[-1] != bias.shape[0]:
        raise ValueError("classifier-head GA received an invalid dense-layer shape")
    return dense


def _verify_example_count(total_count, expected: int) -> None:
    """Audit one replicated count without synchronizing production calls."""

    observed = int(jax.device_get(total_count[0]))
    if observed != expected:
        raise AssertionError(
            f"validation gradient used {observed} examples, expected {expected}"
        )


class FullParameterGradientAlignmentScorer:
    """Prepared-validation-batch direction and JVP over all parameters."""

    def __init__(self, *, apply_fn, template_params) -> None:
        self._apply_fn = apply_fn
        self._parameter_structure = jax.tree_util.tree_structure(template_params)
        self._parameter_count = sum(
            int(value.size) for value in jax.tree_util.tree_leaves(template_params)
        )

        # #### GA FULL-PARAMETER VALIDATION DIRECTION: START ####
        # Compute u_B = grad_theta L_B and average it across PMAP devices.
        # B is full Vdev in exact mode or one component in aggregate mode.
        @partial(jax.pmap, axis_name="batch")
        def validation_gradient(
            params,
            batch_stats,
            images,
            labels,
        ):
            def mean_loss(value):
                logits = apply_fn(
                    {"params": value, "batch_stats": batch_stats},
                    images,
                    training=False,
                )
                targets = jax.nn.one_hot(labels, logits.shape[-1])
                return jnp.mean(
                    optax.softmax_cross_entropy(logits, targets)
                )

            local_gradient = jax.grad(mean_loss)(params)
            direction = jax.lax.pmean(local_gradient, axis_name="batch")
            total_count = jax.lax.psum(
                jnp.asarray(labels.shape[0], dtype=jnp.int32),
                axis_name="batch",
            )
            return direction, total_count

        self._validation_gradient = validation_gradient
        # #### GA FULL-PARAMETER VALIDATION DIRECTION: END ####

        # #### GA FULL-PARAMETER JVP: START ####
        # Compute J_theta logits(x; theta) @ u_B in training mode. The first path
        # returns only the tangent; reweighting also needs the primal logits.
        @partial(jax.pmap, axis_name="batch")
        def training_tangent(params, direction, batch_stats, images, dropout_rng):
            def logits(value):
                result, _ = apply_fn(
                    {"params": value, "batch_stats": batch_stats},
                    images,
                    training=True,
                    mutable=["batch_stats"],
                    rngs={"dropout": dropout_rng},
                    sync_batch_stats=True,
                )
                return result

            return jax.jvp(logits, (params,), (direction,))[1]

        self._training_tangent = training_tangent

        @partial(jax.pmap, axis_name="batch")
        def training_logits_and_tangent(
            params,
            direction,
            batch_stats,
            images,
            dropout_rng,
        ):
            def logits(value):
                result, _ = apply_fn(
                    {"params": value, "batch_stats": batch_stats},
                    images,
                    training=True,
                    mutable=["batch_stats"],
                    rngs={"dropout": dropout_rng},
                    sync_batch_stats=True,
                )
                return result

            return jax.jvp(logits, (params,), (direction,))

        self._training_logits_and_tangent = training_logits_and_tangent
        # #### GA FULL-PARAMETER JVP: END ####

    def distributed_validation_direction_replicated(
        self,
        state,
        batch: PreparedValidationBatch,
        *,
        verify_count_on_host: bool = True,
    ):
        """Return the full-parameter mean gradient for the supplied batch."""

        direction, total_count = self._validation_gradient(
            state.params,
            state.batch_stats,
            batch.images,
            batch.labels,
        )
        if verify_count_on_host:
            _verify_example_count(total_count, batch.example_count)
        one_direction = jax_utils.unreplicate(direction)
        if jax.tree_util.tree_structure(one_direction) != self._parameter_structure:
            raise AssertionError("validation-gradient parameter structure changed")
        if sum(
            int(value.size) for value in jax.tree_util.tree_leaves(one_direction)
        ) != self._parameter_count:
            raise AssertionError("validation-gradient parameter count changed")
        return direction

    def score_hard_labels_device_replicated(
        self,
        params,
        batch_stats,
        batch,
        direction,
    ) -> jax.Array:
        """Return all hard-label scores without a host transfer."""

        tangent = self._training_tangent(
            params,
            direction,
            batch_stats,
            jnp.asarray(batch.images),
            jnp.asarray(batch.dropout_keys),
        )
        return relative_hard_label_gains_from_tangent(
            tangent,
            jnp.asarray(batch.soft_targets),
        )

    def score_labels_and_sample_weights_device_replicated(
        self,
        params,
        batch_stats,
        batch,
        direction,
    ) -> tuple[jax.Array, jax.Array]:
        """Return label-relative gains and current-target sample utility."""

        logits, tangent = self._training_logits_and_tangent(
            params,
            direction,
            batch_stats,
            jnp.asarray(batch.images),
            jnp.asarray(batch.dropout_keys),
        )
        targets = jnp.asarray(batch.soft_targets)
        gains = relative_hard_label_gains_from_tangent(tangent, targets)
        utility = jnp.sum(
            (jax.nn.softmax(logits, axis=-1) - targets) * tangent,
            axis=-1,
        )
        return gains, utility


class ClassifierHeadGradientAlignmentScorer:
    """Prepared-validation-batch direction for the final affine head."""

    def __init__(self, *, apply_fn, template_params) -> None:
        dense = _validated_classifier_head(template_params)
        kernel = np.asarray(dense["kernel"])
        bias = np.asarray(dense["bias"])
        if kernel.dtype != bias.dtype or not np.issubdtype(kernel.dtype, np.floating):
            raise ValueError("classifier-head GA requires one shared floating dtype")
        self._parameter_count = int(kernel.size + bias.size)

        # #### GA CLASSIFIER-HEAD VALIDATION DIRECTION: START ####
        # Compute u_B = grad_theta_head L_B for head/Dense_0/{kernel,bias}.
        # B is full Vdev in exact mode or one component in aggregate mode.
        @partial(jax.pmap, axis_name="batch")
        def validation_head_gradient(
            params,
            batch_stats,
            images,
            labels,
        ):
            current_dense = _validated_classifier_head(params)
            _, features = apply_fn(
                {"params": params, "batch_stats": batch_stats},
                images,
                training=False,
                return_features=True,
            )
            features = jax.lax.stop_gradient(features)

            def mean_loss(value):
                logits = features @ value["kernel"] + value["bias"]
                targets = jax.nn.one_hot(labels, logits.shape[-1])
                return jnp.mean(
                    optax.softmax_cross_entropy(logits, targets)
                )

            local_gradient = jax.grad(mean_loss)(current_dense)
            direction = jax.lax.pmean(local_gradient, axis_name="batch")
            total_count = jax.lax.psum(
                jnp.asarray(labels.shape[0], dtype=jnp.int32),
                axis_name="batch",
            )
            return direction, total_count

        self._validation_head_gradient = validation_head_gradient
        # #### GA CLASSIFIER-HEAD VALIDATION DIRECTION: END ####

        # #### GA CLASSIFIER-HEAD DIRECTIONAL DERIVATIVE: START ####
        # For the affine head, J_theta_head logits(x; theta) @ u_B has the closed
        # form features @ u_kernel + u_bias, so no generic jax.jvp is needed.
        @partial(jax.pmap, axis_name="batch")
        def training_tangent(params, direction, batch_stats, images, dropout_rng):
            (_, features), _ = apply_fn(
                {"params": params, "batch_stats": batch_stats},
                images,
                training=True,
                return_features=True,
                mutable=["batch_stats"],
                rngs={"dropout": dropout_rng},
                sync_batch_stats=True,
            )
            return features @ direction["kernel"] + direction["bias"]

        self._training_tangent = training_tangent

        @partial(jax.pmap, axis_name="batch")
        def training_logits_and_tangent(
            params,
            direction,
            batch_stats,
            images,
            dropout_rng,
        ):
            (logits, features), _ = apply_fn(
                {"params": params, "batch_stats": batch_stats},
                images,
                training=True,
                return_features=True,
                mutable=["batch_stats"],
                rngs={"dropout": dropout_rng},
                sync_batch_stats=True,
            )
            tangent = features @ direction["kernel"] + direction["bias"]
            return logits, tangent

        self._training_logits_and_tangent = training_logits_and_tangent
        # #### GA CLASSIFIER-HEAD DIRECTIONAL DERIVATIVE: END ####

    def distributed_validation_direction_replicated(
        self,
        state,
        batch: PreparedValidationBatch,
        *,
        verify_count_on_host: bool = True,
    ):
        """Return the classifier-head mean gradient for the supplied batch."""

        direction, total_count = self._validation_head_gradient(
            state.params,
            state.batch_stats,
            batch.images,
            batch.labels,
        )
        if verify_count_on_host:
            _verify_example_count(total_count, batch.example_count)
        if sum(
            int(value.size)
            for value in jax.tree_util.tree_leaves(jax_utils.unreplicate(direction))
        ) != self._parameter_count:
            raise AssertionError("classifier-head direction structure changed")
        return direction

    def score_hard_labels_device_replicated(
        self,
        params,
        batch_stats,
        batch,
        direction,
    ) -> jax.Array:
        """Return all classifier-head hard-label scores on device."""

        tangent = self._training_tangent(
            params,
            direction,
            batch_stats,
            jnp.asarray(batch.images),
            jnp.asarray(batch.dropout_keys),
        )
        return relative_hard_label_gains_from_tangent(
            tangent,
            jnp.asarray(batch.soft_targets),
        )

    def score_labels_and_sample_weights_device_replicated(
        self,
        params,
        batch_stats,
        batch,
        direction,
    ) -> tuple[jax.Array, jax.Array]:
        """Return classifier-head label gains and sample utility."""

        logits, tangent = self._training_logits_and_tangent(
            params,
            direction,
            batch_stats,
            jnp.asarray(batch.images),
            jnp.asarray(batch.dropout_keys),
        )
        targets = jnp.asarray(batch.soft_targets)
        gains = relative_hard_label_gains_from_tangent(tangent, targets)
        utility = jnp.sum(
            (jax.nn.softmax(logits, axis=-1) - targets) * tangent,
            axis=-1,
        )
        return gains, utility
