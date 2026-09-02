from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from allthemix.methods.utils.validation import normalize_method_name
from allthemix.training.engine.batch_utils import unpack_batch
from allthemix.training.engine.single.eval import eval_step
from allthemix.training.engine.single.single_utils import (
    append_step_metrics,
    to_jax_aux_info,
)
from allthemix.training.engine.single.train import train_step
from allthemix.training.strategy import BatchTrainingStrategy, ValidationAwareStrategy
from allthemix.training.utils.metric_aggregation import aggregate_epoch_metric_lists

AuxInfo = dict[str, jnp.ndarray]


def train_one_epoch(
    state,
    rng: jax.Array,
    train_ds,
    mixer_fn,
    method: str,
    num_classes: int,
    max_train_steps: int,
    sumix_gamma: float = 0.5,
    sumix_semantic_scale: float = -1.0,
    return_sumix_metrics: bool = False,
    return_mix_metrics: bool = False,
    validation_aware_strategy: ValidationAwareStrategy | None = None,
    batch_training_strategy: BatchTrainingStrategy | None = None,
    return_batch_count: bool = False,
):
    """Run one training epoch on a single device.

    When ``return_batch_count`` is enabled, append the number of updates
    actually executed.  The default return shape remains unchanged.
    """
    losses = []
    accuracies = []
    extra_metric_lists: dict[str, list[float]] = {}
    method_name = normalize_method_name(
        method,
    )
    strategy_pair_sums = None
    strategy_pair_counts = None

    if (
        validation_aware_strategy is not None
        and batch_training_strategy is not None
    ):
        raise ValueError(
            "Only one custom training strategy may own a batch update."
        )

    if method_name == "metaaugment" and validation_aware_strategy is None:
        raise ValueError(
            "MetaAugment training requires a validation-aware strategy."
        )

    for step, batch in enumerate(train_ds):
        if max_train_steps > 0 and step >= max_train_steps:
            break

        images, labels, aux_info = unpack_batch(batch)

        images = jnp.asarray(images)
        labels = jnp.asarray(labels)
        aux_info = to_jax_aux_info(aux_info)

        rng, step_rng = jax.random.split(rng)

        if batch_training_strategy is not None:
            state, loss, accuracy, step_extra_metrics = (
                batch_training_strategy.train_step(
                    task_state=state,
                    images=images,
                    labels=labels,
                    rng=step_rng,
                )
            )
            append_step_metrics(
                metric_lists=extra_metric_lists,
                metrics=step_extra_metrics,
            )

        elif validation_aware_strategy is not None:
            meta_batch = validation_aware_strategy.next_meta_batch()
            meta_images, meta_labels, _ = unpack_batch(
                meta_batch,
            )
            meta_images = jnp.asarray(
                meta_images,
            )
            meta_labels = jnp.asarray(
                meta_labels,
            )
            (
                state,
                loss,
                accuracy,
                step_extra_metrics,
                step_pair_sums,
                step_pair_counts,
            ) = validation_aware_strategy.train_step(
                task_state=state,
                images=images,
                labels=labels,
                meta_images=meta_images,
                meta_labels=meta_labels,
                rng=step_rng,
            )
            append_step_metrics(
                metric_lists=extra_metric_lists,
                metrics=step_extra_metrics,
            )
            host_pair_sums = np.asarray(
                jax.device_get(
                    step_pair_sums,
                )
            )
            host_pair_counts = np.asarray(
                jax.device_get(
                    step_pair_counts,
                )
            )
            if strategy_pair_sums is None:
                strategy_pair_sums = np.zeros_like(
                    host_pair_sums,
                    dtype=np.float64,
                )
                strategy_pair_counts = np.zeros_like(
                    host_pair_counts,
                    dtype=np.float64,
                )
            strategy_pair_sums += host_pair_sums
            strategy_pair_counts += host_pair_counts

        elif return_sumix_metrics or return_mix_metrics:
            state, loss, accuracy, step_extra_metrics = train_step(
                state=state,
                rng=step_rng,
                images=images,
                labels=labels,
                aux_info=aux_info,
                mixer_fn=mixer_fn,
                method=method,
                num_classes=num_classes,
                sumix_gamma=sumix_gamma,
                sumix_semantic_scale=sumix_semantic_scale,
                return_sumix_metrics=return_sumix_metrics,
                return_mix_metrics=return_mix_metrics,
            )

            append_step_metrics(
                metric_lists=extra_metric_lists,
                metrics=step_extra_metrics,
            )

        else:
            state, loss, accuracy = train_step(
                state=state,
                rng=step_rng,
                images=images,
                labels=labels,
                aux_info=aux_info,
                mixer_fn=mixer_fn,
                method=method,
                num_classes=num_classes,
                sumix_gamma=sumix_gamma,
                sumix_semantic_scale=sumix_semantic_scale,
            )

        losses.append(float(loss))
        accuracies.append(float(accuracy))

    mean_loss = float(np.mean(losses))  # Average step losses over the epoch.
    mean_accuracy = float(np.mean(accuracies))  # Average step accuracies over the epoch.
    mean_extra_metrics = aggregate_epoch_metric_lists(
        extra_metric_lists,
    )

    if validation_aware_strategy is not None:
        if strategy_pair_sums is None or strategy_pair_counts is None:
            raise ValueError(
                "Validation-aware training dataset produced no batches."
            )
        mean_extra_metrics.update(
            validation_aware_strategy.finish_epoch(
                pair_sums=strategy_pair_sums,
                pair_counts=strategy_pair_counts,
            )
        )

    result = (state, rng, mean_loss, mean_accuracy, mean_extra_metrics)
    if return_batch_count:
        return (*result, len(losses))
    return result


def evaluate(
    state,
    test_ds,
    num_classes: int,
    max_eval_steps: int,
    return_counts: bool = False,
):
    """Evaluate a model with every example receiving equal weight.

    When ``return_counts`` is enabled, append the processed batch and example
    counts while preserving the default five-metric return tuple.
    """
    total_loss = 0.0
    total_top1_correct = 0.0
    total_top5_correct = 0.0
    total_count = 0
    processed_batches = 0

    for step, batch in enumerate(test_ds):
        if max_eval_steps > 0 and step >= max_eval_steps:
            break

        images, labels, _ = unpack_batch(batch)

        images = jnp.asarray(images)
        labels = jnp.asarray(labels)
        batch_count = int(labels.shape[0])

        loss, top1_acc, top5_acc, _top1_error, _top5_error = eval_step(
            state=state,
            images=images,
            labels=labels,
            num_classes=num_classes,
        )

        # Convert per-batch means back to sums before aggregating uneven batches.
        total_loss += float(loss) * batch_count
        total_top1_correct += float(top1_acc) * batch_count
        total_top5_correct += float(top5_acc) * batch_count
        total_count += batch_count
        processed_batches += 1

    if total_count == 0:
        raise ValueError(
            "Evaluation dataset produced no samples."
        )

    mean_loss = float(total_loss / total_count)
    mean_top1_accuracy = float(total_top1_correct / total_count)
    mean_top5_accuracy = float(total_top5_correct / total_count)
    mean_top1_error = float(1.0 - mean_top1_accuracy)
    mean_top5_error = float(1.0 - mean_top5_accuracy)

    result = (
        mean_loss,
        mean_top1_accuracy,
        mean_top5_accuracy,
        mean_top1_error,
        mean_top5_error,
    )
    if return_counts:
        return (*result, processed_batches, total_count)
    return result
