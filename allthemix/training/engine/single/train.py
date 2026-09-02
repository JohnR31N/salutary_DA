from __future__ import annotations

from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from allthemix.methods.catchupmix import (
    CatchupMixContext,
    make_catchup_mix_feature_hook,
)
from allthemix.methods.selector import MixerFn
from allthemix.methods.utils.validation import normalize_method_name
from allthemix.networks.heads.sumix_head import SUMixUncertaintyHead
from allthemix.training.engine.state import TrainStateWithBatchStats
from allthemix.training.losses.loss_selector import compute_train_loss_and_targets
from allthemix.training.losses.sumix_loss import sumix_loss
from allthemix.training.utils.mix_metrics import (
    compute_mix_debug_metrics,
    unpack_mix_debug_inputs,
)


def _build_optimizer(
    learning_rate: float | optax.Schedule,
    momentum: float,
    weight_decay: float,
    nesterov: bool = False,
) -> optax.GradientTransformation:
    """Build the SGD optimizer with optional decoupled-style weight decay."""
    optimizer = optax.sgd(
        learning_rate=learning_rate,
        momentum=momentum,
        nesterov=nesterov,
    )

    if weight_decay > 0:
        optimizer = optax.chain(
            optax.add_decayed_weights(
                weight_decay,
            ),
            optimizer,
        )

    return optimizer


def create_train_state(
    rng: jax.Array,
    model: nn.Module,
    learning_rate: float | optax.Schedule,
    momentum: float,
    weight_decay: float,
    input_shape: tuple[int, int, int, int],
    nesterov: bool = False,
) -> TrainStateWithBatchStats:
    """Initialize model, optimizer, batch stats, and SUMix auxiliary state."""
    rng_model, rng_sumix = jax.random.split(
        rng,
        2,
    )

    dummy_images = jnp.ones(
        input_shape,
    )

    variables = model.init(
        rng_model,
        dummy_images,
        training=True,
    )

    params = variables["params"]
    batch_stats = variables.get(
        "batch_stats",
        {},
    )

    optimizer = _build_optimizer(
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=nesterov,
    )

    dummy_logits, dummy_features = model.apply(
        {
            "params": params,
            "batch_stats": batch_stats,
        },
        dummy_images,
        training=False,
        return_features=True,
    )

    num_classes = dummy_logits.shape[-1]

    sumix_head = SUMixUncertaintyHead(
        num_classes=num_classes,
        hidden_dim=128,
    )

    sumix_variables = sumix_head.init(
        rng_sumix,
        dummy_features,
        training=True,
    )

    sumix_params = sumix_variables["params"]

    sumix_optimizer = _build_optimizer(
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=nesterov,
    )

    sumix_opt_state = sumix_optimizer.init(
        sumix_params,
    )

    state = TrainStateWithBatchStats.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        batch_stats=batch_stats,
        sumix_apply_fn=sumix_head.apply,
        sumix_params=sumix_params,
        sumix_tx=sumix_optimizer,
        sumix_opt_state=sumix_opt_state,
    )

    return state


@partial(
    jax.jit,
    static_argnames=(
        "mixer_fn",
        "method",
        "num_classes",
        "sumix_gamma",
        "sumix_semantic_scale",
        "return_sumix_metrics",
        "return_mix_metrics",
    ),
)
def train_step(
    state: TrainStateWithBatchStats,
    rng: jax.Array,
    images: jnp.ndarray,
    labels: jnp.ndarray,
    mixer_fn: MixerFn,
    method: str,
    num_classes: int,
    aux_info: dict[str, jnp.ndarray] | None = None,
    sumix_gamma: float = 0.5,
    sumix_semantic_scale: float = -1.0,
    return_sumix_metrics: bool = False,
    return_mix_metrics: bool = False,
) -> tuple[TrainStateWithBatchStats, jnp.ndarray, jnp.ndarray]:
    """Run one JIT-compiled single-device training step."""
    method_name = normalize_method_name(method)
    sumix_metrics = {}

    if aux_info is None:
        aux_info = {}

    mix_rng, dropout_rng = jax.random.split(
        rng,
        2,
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

        new_state = state.apply_gradients(
            grads=grads,
            batch_stats=new_model_state["batch_stats"],
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
            batch_stats=new_model_state["batch_stats"],
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

        new_state = state.apply_gradients(
            grads=grads,
            batch_stats=new_model_state["batch_stats"],
        )

    predictions = jnp.argmax(  # Convert logits to predicted class ids.
        logits,
        axis=-1,
    )

    accuracy = jnp.mean(  # Average exact-match correctness over the batch.
        predictions == target_labels,
    )

    if return_sumix_metrics or return_mix_metrics:
        return new_state, loss, accuracy, {
            **mix_metrics,
            **sumix_metrics,
        }

    return new_state, loss, accuracy
