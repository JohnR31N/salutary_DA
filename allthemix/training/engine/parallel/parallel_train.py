from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import optax

from allthemix.methods.catchupmix import (
    CatchupMixContext,
    make_catchup_mix_feature_hook,
)
from allthemix.methods.selector import MixerFn
from allthemix.methods.utils.validation import normalize_method_name
from allthemix.training.engine.parallel.parallel_utils import (
    make_cross_device_pairs,
)
from allthemix.training.engine.state import TrainStateWithBatchStats
from allthemix.training.losses.loss_selector import compute_train_loss_and_targets
from allthemix.training.losses.sumix_loss import sumix_loss
from allthemix.training.utils.mix_metrics import (
    compute_mix_debug_metrics,
    unpack_mix_debug_inputs,
)


@partial(
    jax.pmap,
    axis_name="batch",
    static_broadcasted_argnums=(
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
    ),
)
def parallel_train_step(
    state: TrainStateWithBatchStats,
    rng: jax.Array,
    images: jnp.ndarray,
    labels: jnp.ndarray,
    aux_info: dict[str, jnp.ndarray],
    mixer_fn: MixerFn,
    method: str,
    num_classes: int,
    sumix_gamma: float = 0.5,
    sumix_semantic_scale: float = -1.0,
    return_sumix_metrics: bool = False,
    cross_device_shuffle: bool = False,
    cross_device_no_repeat: bool = False,
    sync_batch_stats: bool = False,
    return_mix_metrics: bool = False,
) -> tuple[TrainStateWithBatchStats, jnp.ndarray, jnp.ndarray]:
    """Run one PMAP training step across local devices."""
    method_name = normalize_method_name(method)
    sumix_metrics = {}

    mix_rng, dropout_rng = jax.random.split(
        rng,
        2,
    )

    cross_device_supported_methods = (
        "mixup",
        "cutmix",
        "saliencymix",
        "fmix",
        "resizemix",
    )

    if cross_device_shuffle and method_name in cross_device_supported_methods:
        paired_images, paired_labels, paired_perm, paired_aux = make_cross_device_pairs(
            rng=mix_rng,
            images=images,
            labels=labels,
            aux_info=aux_info,
            no_repeat=cross_device_no_repeat,
        )
        aux_info = dict(
            aux_info,
        )
        aux_info["paired_images"] = paired_images
        aux_info["paired_labels"] = paired_labels
        aux_info["paired_perm"] = paired_perm
        aux_info.update(
            paired_aux,
        )

    mixer_output = mixer_fn(
        rng=mix_rng,
        images=images,
        labels=labels,
        aux_info=aux_info,
    )

    mixed_images = mixer_output.images
    mix_metrics = {}

    if return_mix_metrics:
        _, labels_a, labels_b, lam, perm = unpack_mix_debug_inputs(
            mixer_output,
        )
        mix_metrics = compute_mix_debug_metrics(
            images=images,
            mixed_images=mixed_images,
            labels_a=labels_a,
            labels_b=labels_b,
            lam=lam,
            perm=perm,
        )

    if method_name in (
        "catchupmix",
        "catchup_mix",
        "catch_up_mix",
    ):
        catchup_context = CatchupMixContext(
            layer=mixer_output.layer,
            lam=mixer_output.lam,
            perm=mixer_output.perm,
        )
        catchup_feature_hook = make_catchup_mix_feature_hook(
            catchup_context,
        )

        def loss_fn(
            params,
        ):
            """Compute loss and auxiliary outputs for gradient evaluation."""
            variables = {
                "params": params,
                "batch_stats": state.batch_stats,
            }

            logits, new_model_state = state.apply_fn(
                variables,
                mixed_images,
                training=True,
                mutable=[
                    "batch_stats",
                ],
                rngs={
                    "dropout": dropout_rng,
                },
                feature_hook=catchup_feature_hook,
                sync_batch_stats=sync_batch_stats,
            )

            loss, target_labels = compute_train_loss_and_targets(
                method=method,
                logits=logits,
                mixer_output=mixer_output,
                num_classes=num_classes,
            )

            return loss, (
                logits,
                target_labels,
                new_model_state,
            )

        (
            loss,
            aux,
        ), grads = jax.value_and_grad(  # Differentiate loss with model params.
            loss_fn,
            has_aux=True,
        )(
            state.params,
        )

        logits, target_labels, new_model_state = aux

        grads = jax.lax.pmean(  # Average gradients across devices.
            grads,
            axis_name="batch",
        )

        new_batch_stats = new_model_state[
            "batch_stats"
        ]  # Keep the BN stats produced by this forward pass.

        new_state = state.apply_gradients(
            grads=grads,
            batch_stats=new_batch_stats,
        )

    elif method_name == "cutmix_sumix":
        dropout_original_rng, dropout_mix_rng = jax.random.split(
            dropout_rng,
            2,
        )

        def loss_fn(
            params,
            sumix_params,
        ):
            """Compute loss and auxiliary outputs for gradient evaluation."""
            variables = {
                "params": params,
                "batch_stats": state.batch_stats,
            }

            (
                logits_original,
                features_original,
            ), original_model_state = state.apply_fn(
                variables,
                images,
                training=True,
                return_features=True,
                mutable=[
                    "batch_stats",
                ],
                rngs={
                    "dropout": dropout_original_rng,
                },
                sync_batch_stats=sync_batch_stats,
            )

            original_batch_stats = jax.tree_util.tree_map(
                jax.lax.stop_gradient,
                original_model_state.get(
                    "batch_stats",
                    {},
                ),
            )

            mixed_variables = {
                "params": params,
                "batch_stats": original_batch_stats,
            }

            (
                logits_mix,
                features_mix,
            ), new_model_state = state.apply_fn(
                mixed_variables,
                mixed_images,
                training=True,
                return_features=True,
                mutable=[
                    "batch_stats",
                ],
                rngs={
                    "dropout": dropout_mix_rng,
                },
                sync_batch_stats=sync_batch_stats,
            )

            uncertainty_original = state.sumix_apply_fn(
                {
                    "params": sumix_params,
                },
                features_original,
                training=True,
            )

            uncertainty_mixed = state.sumix_apply_fn(
                {
                    "params": sumix_params,
                },
                features_mix,
                training=True,
            )

            loss, metrics = sumix_loss(
                logits_original=logits_original,
                logits_mixed=logits_mix,
                uncertainty_original=uncertainty_original,
                uncertainty_mixed=uncertainty_mixed,
                labels_a=mixer_output.labels_a,
                labels_b=mixer_output.labels_b,
                area_lam=mixer_output.lam,
                perm=mixer_output.perm,
                num_classes=num_classes,
                gamma=sumix_gamma,
                semantic_scale=sumix_semantic_scale,
            )

            target_labels = mixer_output.labels_a

            return loss, (
                logits_mix,
                target_labels,
                new_model_state,
                metrics,
            )

        (
            loss,
            aux,
        ), (
            model_grads,
            sumix_grads,
        ) = jax.value_and_grad(  # Differentiate loss with model and SUMix params.
            loss_fn,
            argnums=(
                0,
                1,
            ),
            has_aux=True,
        )(
            state.params,
            state.sumix_params,
        )

        logits, target_labels, new_model_state, sumix_metrics = aux

        model_grads = jax.lax.pmean(  # Average model gradients across devices.
            model_grads,
            axis_name="batch",
        )

        sumix_grads = jax.lax.pmean(  # Average SUMix gradients across devices.
            sumix_grads,
            axis_name="batch",
        )

        sumix_metrics = jax.tree_util.tree_map(
            lambda value: jax.lax.pmean(
                value,
                axis_name="batch",
            ),
            sumix_metrics,
        )

        new_batch_stats = new_model_state[
            "batch_stats"
        ]  # Keep the BN stats produced by this forward pass.

        sumix_updates, new_sumix_opt_state = state.sumix_tx.update(
            sumix_grads,
            state.sumix_opt_state,
            state.sumix_params,
        )

        new_sumix_params = optax.apply_updates(  # Apply SUMix optimizer updates.
            state.sumix_params,
            sumix_updates,
        )

        new_state = state.apply_gradients(
            grads=model_grads,
            batch_stats=new_batch_stats,
            sumix_params=new_sumix_params,
            sumix_opt_state=new_sumix_opt_state,
        )

    else:

        def loss_fn(
            params,
        ):
            """Compute loss and auxiliary outputs for gradient evaluation."""
            variables = {
                "params": params,
                "batch_stats": state.batch_stats,
            }

            logits, new_model_state = state.apply_fn(
                variables,
                mixed_images,
                training=True,
                mutable=[
                    "batch_stats",
                ],
                rngs={
                    "dropout": dropout_rng,
                },
                sync_batch_stats=sync_batch_stats,
            )

            loss, target_labels = compute_train_loss_and_targets(
                method=method,
                logits=logits,
                mixer_output=mixer_output,
                num_classes=num_classes,
            )

            return loss, (
                logits,
                target_labels,
                new_model_state,
            )

        (
            loss,
            aux,
        ), grads = jax.value_and_grad(  # Differentiate loss with model params.
            loss_fn,
            has_aux=True,
        )(
            state.params,
        )

        logits, target_labels, new_model_state = aux

        grads = jax.lax.pmean(  # Average gradients across devices.
            grads,
            axis_name="batch",
        )

        new_batch_stats = new_model_state[
            "batch_stats"
        ]  # Keep the BN stats produced by this forward pass.

        new_state = state.apply_gradients(
            grads=grads,
            batch_stats=new_batch_stats,
        )

    predictions = jnp.argmax(  # Convert logits to predicted class ids.
        logits,
        axis=-1,
    )

    accuracy = jnp.mean(  # Average exact-match correctness per device.
        predictions == target_labels,
    )

    loss = jax.lax.pmean(  # Average scalar loss across devices.
        loss,
        axis_name="batch",
    )

    accuracy = jax.lax.pmean(  # Average scalar accuracy across devices.
        accuracy,
        axis_name="batch",
    )

    if return_sumix_metrics or return_mix_metrics:
        extra_metrics = {
            **mix_metrics,
            **sumix_metrics,
        }
        extra_metrics = jax.tree_util.tree_map(
            lambda value: jax.lax.pmean(
                value,
                axis_name="batch",
            ),
            extra_metrics,
        )

        return new_state, loss, accuracy, extra_metrics

    return new_state, loss, accuracy
