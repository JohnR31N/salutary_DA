from __future__ import annotations

# This integration preserves MetaAugment's bilevel update while replacing its
# standalone loader, task network, CLI, logger, and epoch loop with AllTheMix.
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import jax_utils, struct
from flax.training.train_state import TrainState

from allthemix.competitors.metaaugment.augmentations import (
    NUM_OPS,
    apply_metaaugment,
    initial_sampler_probs,
)
from allthemix.competitors.metaaugment.policy import MetaAugmentPolicy
from allthemix.data.utils.normalization import (
    get_normalization_stats as get_dataset_normalization_stats,
)

META_AUGMENT_METRIC_NAMES = [
    "metaaugment_policy_loss",
    "metaaugment_policy_weight_mean",
    "metaaugment_inner_lr",
    "metaaugment_sampler_entropy",
    "metaaugment_sampler_max_probability",
]


class PolicyTrainState(TrainState):
    """Optimizer state for the MetaAugment controller."""


@struct.dataclass
class MetaAugmentMethodState:
    """Checkpointable state owned only by the MetaAugment strategy."""

    policy_state: PolicyTrainState
    sampler_probs: jnp.ndarray
    sampler_history: jnp.ndarray
    sampler_history_count: jnp.ndarray
    epoch: jnp.ndarray


@struct.dataclass
class MetaAugmentCheckpointState:
    """Bundle shared classifier state with MetaAugment-specific state."""

    task_state: Any
    method_state: MetaAugmentMethodState


def get_normalization_stats(
    dataset: str,
    tiny_imagenet_normalization: str = "imagenet",
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return the exact channel statistics used by AllTheMix preprocessing."""
    return get_dataset_normalization_stats(
        dataset=dataset,
        tiny_imagenet_normalization=tiny_imagenet_normalization,
    )


def _softplus_inverse(
    value: float,
) -> float:
    """Map a positive initial inner rate into unconstrained parameter space."""
    return float(
        np.log(
            np.expm1(
                value,
            )
        )
    )


def _hard_cross_entropy_per_sample(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    num_classes: int,
) -> jnp.ndarray:
    """Compute one hard-label cross entropy value per sample."""
    targets = jax.nn.one_hot(
        labels,
        num_classes,
    )
    log_probabilities = jax.nn.log_softmax(
        logits,
        axis=-1,
    )

    return -jnp.sum(
        targets * log_probabilities,
        axis=-1,
    )


def normalized_policy_weights(
    raw_weights: jnp.ndarray,
    axis_name: str | None = None,
) -> jnp.ndarray:
    """Normalize controller outputs over the global augmented batch."""
    weight_sum = jnp.sum(
        raw_weights,
    )
    if axis_name is not None:
        weight_sum = jax.lax.psum(
            weight_sum,
            axis_name=axis_name,
        )

    return raw_weights / (
        weight_sum
        + 1.0e-8
    )


def _build_policy_optimizer(
    params,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
    nesterov: bool,
    learn_inner_learning_rate: bool,
) -> optax.GradientTransformation:
    """Build controller SGD while excluding the learned scalar from decay."""
    decay_mask = {
        "policy": jax.tree_util.tree_map(
            lambda _: True,
            params["policy"],
        ),
    }
    if learn_inner_learning_rate:
        decay_mask["inner_log_lr"] = False

    transforms = []
    if weight_decay > 0.0:
        transforms.append(
            optax.add_decayed_weights(
                weight_decay,
                mask=decay_mask,
            )
        )
    transforms.append(
        optax.sgd(
            learning_rate=learning_rate,
            momentum=momentum,
            nesterov=nesterov,
        )
    )

    return optax.chain(
        *transforms,
    )


def _build_train_step(
    policy: MetaAugmentPolicy,
    num_classes: int,
    normalization_mean: tuple[float, float, float],
    normalization_std: tuple[float, float, float],
    num_transforms_per_sample: int,
    cutout_size: int,
    translate_const: float,
    inner_learning_rate: float,
    learn_inner_learning_rate: bool,
    distributed: bool = False,
    sync_batch_stats: bool = False,
) -> Callable:
    """Build the differentiable policy-before-classifier update."""
    axis_name = "batch" if distributed else None
    mean = jnp.asarray(
        normalization_mean,
        dtype=jnp.float32,
    ).reshape(
        (
            1,
            1,
            1,
            3,
        )
    )
    std = jnp.asarray(
        normalization_std,
        dtype=jnp.float32,
    ).reshape(
        (
            1,
            1,
            1,
            3,
        )
    )

    def normalize(
        images: jnp.ndarray,
    ) -> jnp.ndarray:
        """Normalize raw policy images for the shared classifier."""
        return (
            images - mean
        ) / std

    def denormalize(
        images: jnp.ndarray,
    ) -> jnp.ndarray:
        """Recover [0, 1] pixels after AllTheMix base preprocessing."""
        return jnp.clip(
            images * std + mean,
            0.0,
            1.0,
        )

    def global_sum(
        value: jnp.ndarray,
    ) -> jnp.ndarray:
        """Sum one scalar over replicas when data parallelism is active."""
        if axis_name is None:
            return value

        return jax.lax.psum(
            value,
            axis_name=axis_name,
        )

    def global_mean(
        value: jnp.ndarray,
    ) -> jnp.ndarray:
        """Average one local scalar over all replicas."""
        if axis_name is None:
            return value

        return jax.lax.pmean(
            value,
            axis_name=axis_name,
        )

    def average_gradients(
        gradients,
    ):
        """Average replicated parameter gradients over all devices."""
        if axis_name is None:
            return gradients

        return jax.lax.pmean(
            gradients,
            axis_name=axis_name,
        )

    def step(
        task_state,
        policy_state: PolicyTrainState,
        sampler_probs: jnp.ndarray,
        images: jnp.ndarray,
        labels: jnp.ndarray,
        meta_images: jnp.ndarray,
        meta_labels: jnp.ndarray,
        rng: jax.Array,
    ):
        """Run one official-order MetaAugment bilevel update."""
        rng_augment, rng_inner, rng_task = jax.random.split(
            rng,
            3,
        )
        raw_images = denormalize(
            images,
        )
        augmented_images, augmented_labels, embeddings, pair_ids = apply_metaaugment(
            images=raw_images,
            labels=labels,
            key=rng_augment,
            sampler_probs=sampler_probs,
            num_transforms_per_sample=num_transforms_per_sample,
            cutout_size=cutout_size,
            translate_const=translate_const,
        )
        normalized_images = normalize(
            augmented_images,
        )

        def weighted_augmented_loss(
            task_params,
            policy_params,
            dropout_rng: jax.Array,
        ):
            """Compute the normalized sample-weighted task objective."""
            (
                logits,
                features,
            ), new_model_state = task_state.apply_fn(
                {
                    "params": task_params,
                    "batch_stats": task_state.batch_stats,
                },
                normalized_images,
                training=True,
                return_features=True,
                mutable=[
                    "batch_stats",
                ],
                rngs={
                    "dropout": dropout_rng,
                },
                sync_batch_stats=sync_batch_stats,
            )
            raw_weights = policy.apply(
                {
                    "params": policy_params,
                },
                jax.lax.stop_gradient(
                    features,
                ),
                embeddings,
            )
            losses = _hard_cross_entropy_per_sample(
                logits=logits,
                labels=augmented_labels,
                num_classes=num_classes,
            )
            weights = normalized_policy_weights(
                raw_weights,
                axis_name=axis_name,
            )
            local_loss = jnp.sum(  # Local contribution to global L_train.
                weights * losses,
            )
            loss = global_sum(
                local_loss,
            )

            return loss, (
                logits,
                raw_weights,
                new_model_state["batch_stats"],
            )

        def policy_loss_fn(
            meta_params,
        ) -> jnp.ndarray:
            """Differentiate validation loss through one pseudo task update."""
            if learn_inner_learning_rate:
                inner_rate = jax.nn.softplus(
                    meta_params["inner_log_lr"],
                )
            else:
                inner_rate = jnp.asarray(
                    inner_learning_rate,
                    dtype=jnp.float32,
                )

            def inner_task_loss(
                task_params,
            ) -> jnp.ndarray:
                """Build the policy-weighted objective for pseudo parameters."""
                loss, _ = weighted_augmented_loss(
                    task_params=task_params,
                    policy_params=meta_params["policy"],
                    dropout_rng=rng_inner,
                )

                return loss

            task_gradients = jax.grad(
                inner_task_loss,
            )(
                task_state.params,
            )
            task_gradients = average_gradients(
                task_gradients,
            )
            pseudo_params = optax.apply_updates(
                task_state.params,
                jax.tree_util.tree_map(
                    lambda gradient: -inner_rate * gradient,
                    task_gradients,
                ),
            )
            meta_logits = task_state.apply_fn(
                {
                    "params": pseudo_params,
                    "batch_stats": task_state.batch_stats,
                },
                meta_images,
                training=False,
            )
            meta_losses = _hard_cross_entropy_per_sample(
                logits=meta_logits,
                labels=meta_labels,
                num_classes=num_classes,
            )

            meta_loss_sum = global_sum(
                jnp.sum(
                    meta_losses,
                )
            )
            meta_count = global_sum(
                jnp.asarray(
                    meta_losses.size,
                    dtype=meta_losses.dtype,
                )
            )

            return meta_loss_sum / meta_count

        policy_loss, policy_gradients = jax.value_and_grad(
            policy_loss_fn,
        )(
            policy_state.params,
        )
        policy_gradients = average_gradients(
            policy_gradients,
        )
        new_policy_state = policy_state.apply_gradients(
            grads=policy_gradients,
        )

        def task_loss_fn(
            task_params,
        ):
            """Update the classifier with the newly updated policy."""
            return weighted_augmented_loss(
                task_params=task_params,
                policy_params=new_policy_state.params["policy"],
                dropout_rng=rng_task,
            )

        (
            task_loss,
            (
                logits,
                raw_weights,
                new_batch_stats,
            ),
        ), task_gradients = jax.value_and_grad(
            task_loss_fn,
            has_aux=True,
        )(
            task_state.params,
        )
        task_gradients = average_gradients(
            task_gradients,
        )
        new_task_state = task_state.apply_gradients(
            grads=task_gradients,
            batch_stats=new_batch_stats,
        )
        predictions = jnp.argmax(
            logits,
            axis=-1,
        )
        accuracy = global_mean(
            jnp.mean(
                predictions == augmented_labels,
            )
        )
        pair_sums = jnp.zeros(
            (
                NUM_OPS * NUM_OPS,
            ),
            dtype=jnp.float32,
        ).at[
            pair_ids
        ].add(
            raw_weights,
        )
        pair_counts = jnp.zeros(
            (
                NUM_OPS * NUM_OPS,
            ),
            dtype=jnp.float32,
        ).at[
            pair_ids
        ].add(
            1.0,
        )
        pair_sums = global_sum(
            pair_sums,
        )
        pair_counts = global_sum(
            pair_counts,
        )
        if learn_inner_learning_rate:
            current_inner_rate = jax.nn.softplus(
                new_policy_state.params["inner_log_lr"],
            )
        else:
            current_inner_rate = jnp.asarray(
                inner_learning_rate,
                dtype=jnp.float32,
            )
        metrics = {
            "metaaugment_policy_loss": global_mean(
                policy_loss,
            ),
            "metaaugment_policy_weight_mean": global_mean(
                jnp.mean(
                    raw_weights,
                ),
            ),
            "metaaugment_inner_lr": current_inner_rate,
        }

        return (
            new_task_state,
            new_policy_state,
            task_loss,
            accuracy,
            metrics,
            pair_sums.reshape(
                (
                    NUM_OPS,
                    NUM_OPS,
                )
            ),
            pair_counts.reshape(
                (
                    NUM_OPS,
                    NUM_OPS,
                )
            ),
        )

    if distributed:
        return jax.pmap(
            step,
            axis_name="batch",
        )

    return jax.jit(
        step,
    )


def _updated_sampler_probs(
    history: np.ndarray,
    history_count: int,
    epsilon: float,
) -> np.ndarray:
    """Update operation-pair sampling from recent mean policy weights."""
    valid_history = history[
        :history_count
    ]
    sums = np.sum(
        valid_history[..., 0],
        axis=0,
        dtype=np.float64,
    )
    counts = np.sum(
        valid_history[..., 1],
        axis=0,
        dtype=np.float64,
    )
    values = sums / np.maximum(
        counts,
        1.0,
    )
    values = np.where(
        counts > 0.0,
        values,
        1.0,
    )
    values = values / np.maximum(
        np.sum(
            values,
        ),
        1.0e-12,
    )
    uniform = np.full_like(
        values,
        1.0 / values.size,
    )

    return (
        (1.0 - epsilon) * values
        + epsilon * uniform
    ).astype(
        np.float32,
    )


@dataclass
class MetaAugmentContext:
    """Mutable host wrapper around checkpointable MetaAugment method state."""

    method_state: MetaAugmentMethodState
    meta_dataset: Any
    train_step_fn: Callable = field(
        repr=False,
    )
    epsilon: float = 0.1
    sampler_update_epochs: int = 1
    distributed: bool = False
    replicated: bool = field(
        default=False,
        init=False,
    )
    meta_iterator: Iterator | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def next_meta_batch(
        self,
    ) -> Any:
        """Return a validation meta-batch, cycling finite test iterables."""
        if self.meta_iterator is None:
            self.meta_iterator = iter(
                self.meta_dataset,
            )

        try:
            return next(
                self.meta_iterator,
            )
        except StopIteration:
            self.meta_iterator = iter(
                self.meta_dataset,
            )

            return next(
                self.meta_iterator,
            )

    def replicate_method_state(self) -> None:
        """Replicate policy and sampler state over local JAX devices."""
        if not self.distributed or self.replicated:
            return

        self.method_state = jax_utils.replicate(
            self.method_state,
        )
        self.replicated = True

    def _host_method_state(self) -> MetaAugmentMethodState:
        """Return an unreplicated method state for host-side operations."""
        if self.replicated:
            return jax_utils.unreplicate(
                self.method_state,
            )

        return self.method_state

    def _set_host_method_state(
        self,
        method_state: MetaAugmentMethodState,
    ) -> None:
        """Store host state and restore replication when required."""
        if self.replicated:
            self.method_state = jax_utils.replicate(
                method_state,
            )
        else:
            self.method_state = method_state

    def train_step(
        self,
        task_state,
        images: jnp.ndarray,
        labels: jnp.ndarray,
        meta_images: jnp.ndarray,
        meta_labels: jnp.ndarray,
        rng: jax.Array,
    ):
        """Run one step and retain the updated controller state."""
        (
            new_task_state,
            new_policy_state,
            loss,
            accuracy,
            metrics,
            pair_sums,
            pair_counts,
        ) = self.train_step_fn(
            task_state,
            self.method_state.policy_state,
            self.method_state.sampler_probs,
            images,
            labels,
            meta_images,
            meta_labels,
            rng,
        )
        self.method_state = self.method_state.replace(
            policy_state=new_policy_state,
        )

        return (
            new_task_state,
            loss,
            accuracy,
            metrics,
            pair_sums,
            pair_counts,
        )

    def finish_epoch(
        self,
        pair_sums: np.ndarray,
        pair_counts: np.ndarray,
    ) -> dict[str, float]:
        """Append sampler statistics and perform the configured epoch update."""
        host_method_state = self._host_method_state()
        history = np.asarray(
            jax.device_get(
                host_method_state.sampler_history,
            )
        ).copy()
        history_count = int(
            jax.device_get(
                host_method_state.sampler_history_count,
            )
        )
        entry = np.stack(
            [
                pair_sums,
                pair_counts,
            ],
            axis=-1,
        ).astype(
            np.float32,
        )
        if history_count < history.shape[0]:
            history[
                history_count
            ] = entry
            history_count += 1
        else:
            history[:-1] = history[1:]
            history[-1] = entry

        epoch = int(
            jax.device_get(
                host_method_state.epoch,
            )
        ) + 1
        sampler_probs = np.asarray(
            jax.device_get(
                host_method_state.sampler_probs,
            )
        )
        if epoch % self.sampler_update_epochs == 0:
            sampler_probs = _updated_sampler_probs(
                history=history,
                history_count=history_count,
                epsilon=self.epsilon,
            )

        self._set_host_method_state(
            host_method_state.replace(
                sampler_probs=jnp.asarray(
                    sampler_probs,
                ),
                sampler_history=jnp.asarray(
                    history,
                ),
                sampler_history_count=jnp.asarray(
                    history_count,
                    dtype=jnp.int32,
                ),
                epoch=jnp.asarray(
                    epoch,
                    dtype=jnp.int32,
                ),
            )
        )
        entropy = -np.sum(
            sampler_probs
            * np.log(
                sampler_probs + 1.0e-12,
            )
        )

        return {
            "metaaugment_sampler_entropy": float(
                entropy,
            ),
            "metaaugment_sampler_max_probability": float(
                np.max(
                    sampler_probs,
                )
            ),
        }

    def checkpoint_state(
        self,
        task_state,
    ) -> MetaAugmentCheckpointState:
        """Create one checkpoint tree for classifier and controller state."""
        return MetaAugmentCheckpointState(
            task_state=task_state,
            method_state=self._host_method_state(),
        )

    def restore_method_state(
        self,
        method_state: MetaAugmentMethodState,
    ) -> None:
        """Restore controller, sampler, and sampler history state."""
        self._set_host_method_state(
            method_state,
        )

    def save_sampler_probs(
        self,
        path: Path,
    ) -> None:
        """Persist the final learned operation-pair distribution."""
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        np.save(
            path,
            np.asarray(
                jax.device_get(
                    self._host_method_state().sampler_probs,
                )
            ),
        )


def create_metaaugment_context(
    rng: jax.Array,
    task_state,
    meta_dataset: Any,
    input_shape: tuple[int, int, int, int],
    dataset: str,
    num_classes: int,
    policy_learning_rate: float = 1.0e-3,
    policy_momentum: float = 0.9,
    policy_weight_decay: float = 5.0e-4,
    policy_nesterov: bool = False,
    inner_learning_rate: float = 0.1,
    learn_inner_learning_rate: bool = True,
    epsilon: float = 0.1,
    num_transforms_per_sample: int = 1,
    cutout_size: int = 16,
    sampler_update_epochs: int = 1,
    sampler_history_epochs: int = 50,
    translate_const: float = 10.0,
    tiny_imagenet_normalization: str = "imagenet",
    distributed: bool = False,
    sync_batch_stats: bool = False,
) -> MetaAugmentContext:
    """Initialize MetaAugment around an existing AllTheMix classifier state."""
    policy = MetaAugmentPolicy()
    dummy_shape = (
        1,
        *input_shape[1:],
    )
    dummy_images = jnp.ones(
        dummy_shape,
        dtype=jnp.float32,
    )
    _, dummy_features = task_state.apply_fn(
        {
            "params": task_state.params,
            "batch_stats": task_state.batch_stats,
        },
        dummy_images,
        training=False,
        return_features=True,
    )
    policy_params = policy.init(
        rng,
        dummy_features,
        jnp.ones(
            (
                1,
                NUM_OPS * 2,
            ),
            dtype=jnp.float32,
        ),
    )[
        "params"
    ]
    meta_params = {
        "policy": policy_params,
    }
    if learn_inner_learning_rate:
        meta_params["inner_log_lr"] = jnp.asarray(
            _softplus_inverse(
                inner_learning_rate,
            ),
            dtype=jnp.float32,
        )
    policy_optimizer = _build_policy_optimizer(
        params=meta_params,
        learning_rate=policy_learning_rate,
        momentum=policy_momentum,
        weight_decay=policy_weight_decay,
        nesterov=policy_nesterov,
        learn_inner_learning_rate=learn_inner_learning_rate,
    )
    policy_state = PolicyTrainState.create(
        apply_fn=policy.apply,
        params=meta_params,
        tx=policy_optimizer,
    )
    method_state = MetaAugmentMethodState(
        policy_state=policy_state,
        sampler_probs=initial_sampler_probs(),
        sampler_history=jnp.zeros(
            (
                sampler_history_epochs,
                NUM_OPS,
                NUM_OPS,
                2,
            ),
            dtype=jnp.float32,
        ),
        sampler_history_count=jnp.asarray(
            0,
            dtype=jnp.int32,
        ),
        epoch=jnp.asarray(
            0,
            dtype=jnp.int32,
        ),
    )
    normalization_mean, normalization_std = get_normalization_stats(
        dataset=dataset,
        tiny_imagenet_normalization=tiny_imagenet_normalization,
    )
    train_step_fn = _build_train_step(
        policy=policy,
        num_classes=num_classes,
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
        num_transforms_per_sample=num_transforms_per_sample,
        cutout_size=cutout_size,
        translate_const=translate_const,
        inner_learning_rate=inner_learning_rate,
        learn_inner_learning_rate=learn_inner_learning_rate,
        distributed=distributed,
        sync_batch_stats=sync_batch_stats,
    )

    return MetaAugmentContext(
        method_state=method_state,
        meta_dataset=meta_dataset,
        train_step_fn=train_step_fn,
        epsilon=epsilon,
        sampler_update_epochs=sampler_update_epochs,
        distributed=distributed,
    )
