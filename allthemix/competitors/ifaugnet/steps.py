from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import jax_utils
from flax.core import FrozenDict
from flax.training.train_state import TrainState

from allthemix.competitors.ifaugnet.influence import (
    compute_s_test,
    influence_up_loss,
    s_test_residual_norm,
)


class AugmentTrainState(TrainState):
    """Optimizer state for the coupled IF-AugNet E/G model."""


class DiscriminatorTrainState(TrainState):
    """Optimizer state for one IF-AugNet discriminator."""


def _tree_global_norm(
    tree: Any,
) -> jnp.ndarray:
    """Compute a pytree L2 norm without relying on Optax versioned APIs."""
    squared_norms = [
        jnp.sum(
            jnp.square(
                leaf,
            )
        )
        for leaf in jax.tree_util.tree_leaves(
            tree,
        )
    ]

    if not squared_norms:
        return jnp.asarray(
            0.0,
            dtype=jnp.float32,
        )

    return jnp.sqrt(
        jnp.sum(
            jnp.stack(
                squared_norms,
            )
        )
    )


def inherit_pretrained_decoder(
    fresh_state: AugmentTrainState,
    pretrained_state: AugmentTrainState,
) -> AugmentTrainState:
    """Copy only pretrained G parameters into a freshly initialized E/G state."""
    if "decoder" not in fresh_state.params:
        raise ValueError("Fresh IF-AugNet parameters do not contain 'decoder'.")
    if "decoder" not in pretrained_state.params:
        raise ValueError("Pretrained IF-AugNet parameters do not contain 'decoder'.")

    if isinstance(
        fresh_state.params,
        FrozenDict,
    ):
        params = fresh_state.params.copy(
            {
                "decoder": pretrained_state.params["decoder"],
            }
        )
    else:
        params = {
            **fresh_state.params,
            "decoder": pretrained_state.params["decoder"],
        }

    return fresh_state.replace(
        params=params,
    )


def create_augment_state(
    rng: jax.Array,
    model: Any,
    input_shape: tuple[int, int, int, int],
    learning_rate: float | optax.Schedule,
    beta1: float = 0.9,
    beta2: float = 0.999,
    gradient_clip_norm: float = 1.0,
    zero_nonfinite_grads: bool = True,
) -> AugmentTrainState:
    """Initialize E/G and its Adam optimizer."""
    params_rng, dropout_rng = jax.random.split(
        rng,
    )
    variables = model.init(
        {
            "params": params_rng,
            "dropout": dropout_rng,
        },
        jnp.ones(
            input_shape,
            dtype=jnp.float32,
        ),
        training=True,
    )
    transforms = []

    if zero_nonfinite_grads:
        transforms.append(
            optax.zero_nans(),
        )

    if gradient_clip_norm > 0.0:
        transforms.append(
            optax.clip_by_global_norm(
                gradient_clip_norm,
            )
        )

    transforms.append(
        optax.adam(
            learning_rate=learning_rate,
            b1=beta1,
            b2=beta2,
        )
    )

    return AugmentTrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=optax.chain(
            *transforms,
        ),
    )


def create_discriminator_state(
    rng: jax.Array,
    model: Any,
    input_shape: tuple[int, ...],
    learning_rate: float = 2.0e-4,
    beta1: float = 0.5,
    beta2: float = 0.999,
) -> DiscriminatorTrainState:
    """Initialize one discriminator and its Adam optimizer."""
    variables = model.init(
        rng,
        jnp.ones(
            input_shape,
            dtype=jnp.float32,
        ),
    )

    return DiscriminatorTrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=optax.adam(
            learning_rate=learning_rate,
            b1=beta1,
            b2=beta2,
        ),
    )


def normalization_arrays(
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Convert channel statistics into broadcastable JAX arrays."""
    channels = len(
        mean,
    )

    return (
        jnp.asarray(
            mean,
            dtype=jnp.float32,
        ).reshape(
            (1, 1, 1, channels),
        ),
        jnp.asarray(
            std,
            dtype=jnp.float32,
        ).reshape(
            (1, 1, 1, channels),
        ),
    )


def normalize_images(
    images: jnp.ndarray,
    mean: jnp.ndarray,
    std: jnp.ndarray,
) -> jnp.ndarray:
    """Normalize raw [0, 1] pixels with dataset channel statistics."""
    return (images - mean) / std


def denormalize_images(
    images: jnp.ndarray,
    mean: jnp.ndarray,
    std: jnp.ndarray,
) -> jnp.ndarray:
    """Recover clipped [0, 1] pixels from normalized model inputs."""
    return jnp.clip(
        images * std + mean,
        0.0,
        1.0,
    )


def _find_linear_params(
    params: Mapping[str, Any],
) -> Mapping[str, jnp.ndarray]:
    """Find the unique dense classifier leaf in an AllTheMix head tree."""
    candidates = []

    def visit(tree):
        """Collect mappings with a rank-two kernel and optional bias."""
        if not isinstance(
            tree,
            Mapping,
        ):
            return

        kernel = tree.get(
            "kernel",
        )

        if kernel is not None and getattr(
            kernel,
            "ndim",
            None,
        ) == 2:
            candidates.append(
                tree,
            )

        for value in tree.values():
            visit(
                value,
            )

    visit(
        params,
    )

    if len(candidates) != 1:
        raise ValueError(
            "IF-AugNet requires exactly one linear classifier leaf under "
            f"params['head']; found {len(candidates)}."
        )

    return candidates[0]


def get_classifier_head_params(
    params: Mapping[str, Any],
) -> Mapping[str, jnp.ndarray]:
    """Return the shared classifier's final dense-layer parameters."""
    if "head" not in params:
        raise ValueError(
            "IF-AugNet expects AllTheMix ImageClassifier params to contain "
            "a 'head' subtree."
        )

    return _find_linear_params(
        params["head"],
    )


def extract_classifier_features(
    classifier_state,
    images: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return frozen features and logits from the shared classifier."""
    logits, features = classifier_state.apply_fn(
        {
            "params": classifier_state.params,
            "batch_stats": classifier_state.batch_stats,
        },
        images,
        training=False,
        return_features=True,
    )

    return features, logits


def infer_feature_dim(
    classifier_state,
    input_shape: tuple[int, int, int, int],
) -> int:
    """Infer the final backbone feature width from a classifier state."""
    features, _ = extract_classifier_features(
        classifier_state=classifier_state,
        images=jnp.ones(
            input_shape,
            dtype=jnp.float32,
        ),
    )

    return int(
        features.shape[-1],
    )


def tree_average(
    trees: list[Any],
) -> Any:
    """Average a non-empty list of identically structured pytrees."""
    if not trees:
        raise ValueError(
            "Cannot average an empty list of pytrees."
        )

    inverse_count = 1.0 / len(
        trees,
    )

    return jax.tree_util.tree_map(
        lambda *values: sum(values) * inverse_count,
        *trees,
    )


def _global_mean(
    values: jnp.ndarray,
    axis_name: str | None,
) -> jnp.ndarray:
    """Average values over the full data-parallel batch."""
    local_mean = jnp.mean(
        values,
    )

    if axis_name is None:
        return local_mean

    return jax.lax.pmean(
        local_mean,
        axis_name=axis_name,
    )


def _average_gradients(
    gradients,
    axis_name: str | None,
):
    """Average replicated optimizer gradients when PMAP is active."""
    if axis_name is None:
        return gradients

    return jax.lax.pmean(
        gradients,
        axis_name=axis_name,
    )


def _ragan_discriminator_loss(
    real_logits: jnp.ndarray,
    fake_logits: jnp.ndarray,
    axis_name: str | None = None,
) -> jnp.ndarray:
    """Compute relativistic-average discriminator loss."""
    real_relative = real_logits - _global_mean(
        fake_logits,
        axis_name=axis_name,
    )
    fake_relative = fake_logits - _global_mean(
        real_logits,
        axis_name=axis_name,
    )

    return _global_mean(
        jax.nn.softplus(
            -real_relative,
        ),
        axis_name=axis_name,
    ) + _global_mean(
        jax.nn.softplus(
            fake_relative,
        ),
        axis_name=axis_name,
    )


def _ragan_generator_loss(
    real_logits: jnp.ndarray,
    fake_logits: jnp.ndarray,
    axis_name: str | None = None,
) -> jnp.ndarray:
    """Compute relativistic-average generator loss."""
    real_relative = real_logits - _global_mean(
        fake_logits,
        axis_name=axis_name,
    )
    fake_relative = fake_logits - _global_mean(
        real_logits,
        axis_name=axis_name,
    )

    return _global_mean(
        jax.nn.softplus(
            real_relative,
        ),
        axis_name=axis_name,
    ) + _global_mean(
        jax.nn.softplus(
            -fake_relative,
        ),
        axis_name=axis_name,
    )


def _augnet_pretrain_step(
    augment_state: AugmentTrainState,
    image_discriminator_state: DiscriminatorTrainState,
    feature_discriminator_state: DiscriminatorTrainState,
    classifier_state,
    raw_images: jnp.ndarray,
    real_images: jnp.ndarray,
    discriminator_tau: jnp.ndarray,
    generator_tau: jnp.ndarray,
    mean: jnp.ndarray,
    std: jnp.ndarray,
    image_loss_weight: float = 1.0,
    feature_loss_weight: float = 1.0,
    identity_l2_weight: float = 0.0,
    axis_name: str | None = None,
) -> tuple[
    AugmentTrainState,
    DiscriminatorTrainState,
    DiscriminatorTrainState,
    dict[str, jnp.ndarray],
]:
    """Run one RaGAN update that pretrains G from random bounded codes."""
    fake_images = augment_state.apply_fn(
        {
            "params": augment_state.params,
        },
        raw_images,
        training=False,
        tau_override=discriminator_tau,
    )
    fake_images_for_discriminator = jax.lax.stop_gradient(
        fake_images,
    )
    real_features, _ = extract_classifier_features(
        classifier_state=classifier_state,
        images=normalize_images(
            images=real_images,
            mean=mean,
            std=std,
        ),
    )
    fake_features, _ = extract_classifier_features(
        classifier_state=classifier_state,
        images=normalize_images(
            images=fake_images_for_discriminator,
            mean=mean,
            std=std,
        ),
    )
    real_features_for_discriminator = jax.lax.stop_gradient(
        real_features,
    )
    fake_features_for_discriminator = jax.lax.stop_gradient(
        fake_features,
    )

    def discriminator_loss_fn(image_params, feature_params):
        """Train image and feature critics against detached G samples."""
        image_real_logits = image_discriminator_state.apply_fn(
            {
                "params": image_params,
            },
            real_images,
        )
        image_fake_logits = image_discriminator_state.apply_fn(
            {
                "params": image_params,
            },
            fake_images_for_discriminator,
        )
        feature_real_logits = feature_discriminator_state.apply_fn(
            {
                "params": feature_params,
            },
            real_features_for_discriminator,
        )
        feature_fake_logits = feature_discriminator_state.apply_fn(
            {
                "params": feature_params,
            },
            fake_features_for_discriminator,
        )
        image_loss = _ragan_discriminator_loss(
            real_logits=image_real_logits,
            fake_logits=image_fake_logits,
            axis_name=axis_name,
        )
        feature_loss = _ragan_discriminator_loss(
            real_logits=feature_real_logits,
            fake_logits=feature_fake_logits,
            axis_name=axis_name,
        )
        loss = (
            image_loss_weight * image_loss
            + feature_loss_weight * feature_loss
        )

        return loss, {
            "d_loss": loss,
            "d_image_loss": image_loss,
            "d_feature_loss": feature_loss,
            "d_real_logit": _global_mean(
                image_real_logits,
                axis_name=axis_name,
            ),
            "d_fake_logit": _global_mean(
                image_fake_logits,
                axis_name=axis_name,
            ),
        }

    (
        _,
        discriminator_metrics,
    ), (
        image_gradients,
        feature_gradients,
    ) = jax.value_and_grad(
        discriminator_loss_fn,
        argnums=(0, 1),
        has_aux=True,
    )(
        image_discriminator_state.params,
        feature_discriminator_state.params,
    )
    image_gradients = _average_gradients(
        image_gradients,
        axis_name=axis_name,
    )
    feature_gradients = _average_gradients(
        feature_gradients,
        axis_name=axis_name,
    )
    image_discriminator_state = image_discriminator_state.apply_gradients(
        grads=image_gradients,
    )
    feature_discriminator_state = feature_discriminator_state.apply_gradients(
        grads=feature_gradients,
    )

    def generator_loss_fn(augment_params):
        """Train G to fool both updated critics while staying near input."""
        generated_images, aux = augment_state.apply_fn(
            {
                "params": augment_params,
            },
            raw_images,
            training=False,
            return_aux=True,
            tau_override=generator_tau,
        )
        generated_features, _ = extract_classifier_features(
            classifier_state=classifier_state,
            images=normalize_images(
                images=generated_images,
                mean=mean,
                std=std,
            ),
        )
        image_real_logits = image_discriminator_state.apply_fn(
            {
                "params": image_discriminator_state.params,
            },
            jax.lax.stop_gradient(
                real_images,
            ),
        )
        image_fake_logits = image_discriminator_state.apply_fn(
            {
                "params": image_discriminator_state.params,
            },
            generated_images,
        )
        feature_real_logits = feature_discriminator_state.apply_fn(
            {
                "params": feature_discriminator_state.params,
            },
            jax.lax.stop_gradient(
                real_features,
            ),
        )
        feature_fake_logits = feature_discriminator_state.apply_fn(
            {
                "params": feature_discriminator_state.params,
            },
            generated_features,
        )
        image_loss = _ragan_generator_loss(
            real_logits=image_real_logits,
            fake_logits=image_fake_logits,
            axis_name=axis_name,
        )
        feature_loss = _ragan_generator_loss(
            real_logits=feature_real_logits,
            fake_logits=feature_fake_logits,
            axis_name=axis_name,
        )
        identity_l2 = _global_mean(  # ||G(x, tau) - x||_2^2.
            jnp.square(
                generated_images - raw_images,
            ),
            axis_name=axis_name,
        )
        loss = (
            image_loss_weight * image_loss
            + feature_loss_weight * feature_loss
            + identity_l2_weight * identity_l2
        )

        return loss, {
            "g_loss": loss,
            "g_image_loss": image_loss,
            "g_feature_loss": feature_loss,
            "pretrain_identity_l2": identity_l2,
            "pretrain_tau_abs_mean": _global_mean(
                jnp.abs(
                    aux["tau_pre_dropout"],
                ),
                axis_name=axis_name,
            ),
            "pretrain_tau_saturation_fraction": _global_mean(
                (
                    jnp.abs(
                        aux["tau_pre_dropout"],
                    )
                    >= 0.95
                ).astype(
                    jnp.float32,
                ),
                axis_name=axis_name,
            ),
        }

    (
        _,
        generator_metrics,
    ), augment_gradients = jax.value_and_grad(
        generator_loss_fn,
        has_aux=True,
    )(
        augment_state.params,
    )
    augment_gradients = _average_gradients(
        augment_gradients,
        axis_name=axis_name,
    )
    generator_metrics = {
        **generator_metrics,
        "pretrain_encoder_grad_norm": _tree_global_norm(
            augment_gradients["encoder"],
        ),
        "pretrain_decoder_grad_norm": _tree_global_norm(
            augment_gradients["decoder"],
        ),
    }
    augment_state = augment_state.apply_gradients(
        grads=augment_gradients,
    )

    return (
        augment_state,
        image_discriminator_state,
        feature_discriminator_state,
        {
            **discriminator_metrics,
            **generator_metrics,
        },
    )


@jax.jit
def augnet_pretrain_step(
    augment_state: AugmentTrainState,
    image_discriminator_state: DiscriminatorTrainState,
    feature_discriminator_state: DiscriminatorTrainState,
    classifier_state,
    raw_images: jnp.ndarray,
    real_images: jnp.ndarray,
    discriminator_tau: jnp.ndarray,
    generator_tau: jnp.ndarray,
    mean: jnp.ndarray,
    std: jnp.ndarray,
    image_loss_weight: float = 1.0,
    feature_loss_weight: float = 1.0,
    identity_l2_weight: float = 0.0,
):
    """Run one single-device adversarial pretraining update."""
    return _augnet_pretrain_step(
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
        image_loss_weight=image_loss_weight,
        feature_loss_weight=feature_loss_weight,
        identity_l2_weight=identity_l2_weight,
    )


@partial(
    jax.pmap,
    axis_name="batch",
    in_axes=(0, 0, 0, 0, 0, 0, 0, 0, None, None, None, None, None),
)
def parallel_augnet_pretrain_step(
    augment_state: AugmentTrainState,
    image_discriminator_state: DiscriminatorTrainState,
    feature_discriminator_state: DiscriminatorTrainState,
    classifier_state,
    raw_images: jnp.ndarray,
    real_images: jnp.ndarray,
    discriminator_tau: jnp.ndarray,
    generator_tau: jnp.ndarray,
    mean: jnp.ndarray,
    std: jnp.ndarray,
    image_loss_weight: float = 1.0,
    feature_loss_weight: float = 1.0,
    identity_l2_weight: float = 0.0,
):
    """Run globally synchronized adversarial updates across devices."""
    return _augnet_pretrain_step(
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
        image_loss_weight=image_loss_weight,
        feature_loss_weight=feature_loss_weight,
        identity_l2_weight=identity_l2_weight,
        axis_name="batch",
    )


def _gather_feature_batch(
    features: jnp.ndarray,
    labels: jnp.ndarray,
    axis_name: str | None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Assemble one global feature minibatch on every replica."""
    if axis_name is None:
        return features, labels

    gathered_features = jax.lax.all_gather(
        features,
        axis_name=axis_name,
    )
    gathered_labels = jax.lax.all_gather(
        labels,
        axis_name=axis_name,
    )

    return (
        gathered_features.reshape(
            (-1, *features.shape[1:]),
        ),
        gathered_labels.reshape(
            (-1, *labels.shape[1:]),
        ),
    )


def _extract_s_test_feature_batch(
    classifier_state,
    train_images: jnp.ndarray,
    train_labels: jnp.ndarray,
    validation_images: jnp.ndarray,
    validation_labels: jnp.ndarray,
    axis_name: str | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Extract one global feature batch for the fixed iHVP system."""
    train_features, _ = extract_classifier_features(
        classifier_state=classifier_state,
        images=train_images,
    )
    validation_features, _ = extract_classifier_features(
        classifier_state=classifier_state,
        images=validation_images,
    )
    train_features, train_labels = _gather_feature_batch(
        features=train_features,
        labels=train_labels,
        axis_name=axis_name,
    )
    validation_features, validation_labels = _gather_feature_batch(
        features=validation_features,
        labels=validation_labels,
        axis_name=axis_name,
    )

    return (
        train_features,
        train_labels,
        validation_features,
        validation_labels,
    )


@jax.jit
def extract_s_test_feature_batch(
    classifier_state,
    train_images: jnp.ndarray,
    train_labels: jnp.ndarray,
    validation_images: jnp.ndarray,
    validation_labels: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Extract one single-device feature batch for a later global solve."""
    return _extract_s_test_feature_batch(
        classifier_state=classifier_state,
        train_images=train_images,
        train_labels=train_labels,
        validation_images=validation_images,
        validation_labels=validation_labels,
    )


@partial(
    jax.pmap,
    axis_name="batch",
    in_axes=(0, 0, 0, 0, 0),
)
def parallel_extract_s_test_feature_batch(
    classifier_state,
    train_images: jnp.ndarray,
    train_labels: jnp.ndarray,
    validation_images: jnp.ndarray,
    validation_labels: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Extract the same global feature batch on every data-parallel replica."""
    return _extract_s_test_feature_batch(
        classifier_state=classifier_state,
        train_images=train_images,
        train_labels=train_labels,
        validation_images=validation_images,
        validation_labels=validation_labels,
        axis_name="batch",
    )


@partial(
    jax.jit,
    static_argnames=(
        "cg_iters",
    ),
)
def compute_feature_s_test(
    classifier_params: Mapping[str, jnp.ndarray],
    train_features: jnp.ndarray,
    train_labels: jnp.ndarray,
    validation_features: jnp.ndarray,
    validation_labels: jnp.ndarray,
    damping: float = 1.0e-2,
    cg_iters: int = 50,
) -> dict[str, jnp.ndarray]:
    """Solve one iHVP system over all precomputed feature batches."""
    return compute_s_test(
        classifier_params=classifier_params,
        train_features=train_features,
        train_labels=train_labels,
        validation_features=validation_features,
        validation_labels=validation_labels,
        damping=damping,
        cg_iters=cg_iters,
    )


@jax.jit
def compute_feature_s_test_residual(
    classifier_params: Mapping[str, jnp.ndarray],
    train_features: jnp.ndarray,
    train_labels: jnp.ndarray,
    validation_features: jnp.ndarray,
    validation_labels: jnp.ndarray,
    s_test: dict[str, jnp.ndarray],
    damping: float = 1.0e-2,
) -> jnp.ndarray:
    """Measure the residual of the aggregate feature-level iHVP solve."""
    return s_test_residual_norm(
        classifier_params=classifier_params,
        train_features=train_features,
        train_labels=train_labels,
        validation_features=validation_features,
        validation_labels=validation_labels,
        s_test=s_test,
        damping=damping,
    )


def _compute_batch_s_test(
    classifier_state,
    train_images: jnp.ndarray,
    train_labels: jnp.ndarray,
    validation_images: jnp.ndarray,
    validation_labels: jnp.ndarray,
    damping: float = 1.0e-2,
    cg_iters: int = 50,
    axis_name: str | None = None,
) -> dict[str, jnp.ndarray]:
    """Compute a minibatch iHVP over the shared classifier's final layer."""
    (
        train_features,
        train_labels,
        validation_features,
        validation_labels,
    ) = _extract_s_test_feature_batch(
        classifier_state=classifier_state,
        train_images=train_images,
        train_labels=train_labels,
        validation_images=validation_images,
        validation_labels=validation_labels,
        axis_name=axis_name,
    )

    return compute_s_test(
        classifier_params=get_classifier_head_params(
            classifier_state.params,
        ),
        train_features=train_features,
        train_labels=train_labels,
        validation_features=validation_features,
        validation_labels=validation_labels,
        damping=damping,
        cg_iters=cg_iters,
    )


@partial(
    jax.jit,
    static_argnames=(
        "cg_iters",
    ),
)
def compute_batch_s_test(
    classifier_state,
    train_images: jnp.ndarray,
    train_labels: jnp.ndarray,
    validation_images: jnp.ndarray,
    validation_labels: jnp.ndarray,
    damping: float = 1.0e-2,
    cg_iters: int = 50,
) -> dict[str, jnp.ndarray]:
    """Compute one single-device minibatch iHVP estimate."""
    return _compute_batch_s_test(
        classifier_state=classifier_state,
        train_images=train_images,
        train_labels=train_labels,
        validation_images=validation_images,
        validation_labels=validation_labels,
        damping=damping,
        cg_iters=cg_iters,
    )


@partial(
    jax.pmap,
    axis_name="batch",
    in_axes=(0, 0, 0, 0, 0, None, None),
    static_broadcasted_argnums=(6,),
)
def parallel_compute_batch_s_test(
    classifier_state,
    train_images: jnp.ndarray,
    train_labels: jnp.ndarray,
    validation_images: jnp.ndarray,
    validation_labels: jnp.ndarray,
    damping: float = 1.0e-2,
    cg_iters: int = 50,
) -> dict[str, jnp.ndarray]:
    """Compute the same global-batch iHVP estimate on every replica."""
    return _compute_batch_s_test(
        classifier_state=classifier_state,
        train_images=train_images,
        train_labels=train_labels,
        validation_images=validation_images,
        validation_labels=validation_labels,
        damping=damping,
        cg_iters=cg_iters,
        axis_name="batch",
    )


def _compute_batch_s_test_residual(
    classifier_state,
    train_images: jnp.ndarray,
    train_labels: jnp.ndarray,
    validation_images: jnp.ndarray,
    validation_labels: jnp.ndarray,
    s_test: dict[str, jnp.ndarray],
    damping: float = 1.0e-2,
    axis_name: str | None = None,
) -> jnp.ndarray:
    """Compute the relative residual for one minibatch iHVP estimate."""
    (
        train_features,
        train_labels,
        validation_features,
        validation_labels,
    ) = _extract_s_test_feature_batch(
        classifier_state=classifier_state,
        train_images=train_images,
        train_labels=train_labels,
        validation_images=validation_images,
        validation_labels=validation_labels,
        axis_name=axis_name,
    )

    return s_test_residual_norm(
        classifier_params=get_classifier_head_params(
            classifier_state.params,
        ),
        train_features=train_features,
        train_labels=train_labels,
        validation_features=validation_features,
        validation_labels=validation_labels,
        s_test=s_test,
        damping=damping,
    )


@jax.jit
def compute_batch_s_test_residual(
    classifier_state,
    train_images: jnp.ndarray,
    train_labels: jnp.ndarray,
    validation_images: jnp.ndarray,
    validation_labels: jnp.ndarray,
    s_test: dict[str, jnp.ndarray],
    damping: float = 1.0e-2,
) -> jnp.ndarray:
    """Compute one single-device iHVP residual."""
    return _compute_batch_s_test_residual(
        classifier_state=classifier_state,
        train_images=train_images,
        train_labels=train_labels,
        validation_images=validation_images,
        validation_labels=validation_labels,
        s_test=s_test,
        damping=damping,
    )


@partial(
    jax.pmap,
    axis_name="batch",
    in_axes=(0, 0, 0, 0, 0, 0, None),
)
def parallel_compute_batch_s_test_residual(
    classifier_state,
    train_images: jnp.ndarray,
    train_labels: jnp.ndarray,
    validation_images: jnp.ndarray,
    validation_labels: jnp.ndarray,
    s_test: dict[str, jnp.ndarray],
    damping: float = 1.0e-2,
) -> jnp.ndarray:
    """Measure the residual of the shared global-batch iHVP estimate."""
    return _compute_batch_s_test_residual(
        classifier_state=classifier_state,
        train_images=train_images,
        train_labels=train_labels,
        validation_images=validation_images,
        validation_labels=validation_labels,
        s_test=s_test,
        damping=damping,
        axis_name="batch",
    )


def _augnet_influence_train_step(
    augment_state: AugmentTrainState,
    classifier_state,
    raw_images: jnp.ndarray,
    labels: jnp.ndarray,
    s_test: dict[str, jnp.ndarray],
    rng: jax.Array,
    mean: jnp.ndarray,
    std: jnp.ndarray,
    identity_l2_weight: float = 0.0,
    influence_clip_value: float = 0.0,
    label_preservation_weight: float = 0.0,
    axis_name: str | None = None,
) -> tuple[AugmentTrainState, dict[str, jnp.ndarray]]:
    """Update E/G using replacement influence on validation loss."""
    labels = labels.astype(
        jnp.int32,
    )
    classifier_params = get_classifier_head_params(
        classifier_state.params,
    )

    def loss_fn(augment_params):
        """Compute influence, identity, and label-preservation objectives."""
        augmented_images, aux = augment_state.apply_fn(
            {
                "params": augment_params,
            },
            raw_images,
            training=True,
            return_aux=True,
            rngs={
                "dropout": rng,
            },
        )
        augmented_features, augmented_logits = extract_classifier_features(
            classifier_state=classifier_state,
            images=normalize_images(
                images=augmented_images,
                mean=mean,
                std=std,
            ),
        )
        original_features, original_logits = extract_classifier_features(
            classifier_state=classifier_state,
            images=normalize_images(
                images=raw_images,
                mean=mean,
                std=std,
            ),
        )
        augmented_influence = influence_up_loss(
            features=augmented_features,
            labels=labels,
            classifier_params=classifier_params,
            s_test=s_test,
        )
        original_influence = influence_up_loss(
            features=original_features,
            labels=labels,
            classifier_params=classifier_params,
            s_test=s_test,
        )
        replacement_influence = (  # I_aug = I_up(G(E(x))) - I_up(x).
            augmented_influence - original_influence
        )
        clip_value = jnp.asarray(
            influence_clip_value,
            dtype=replacement_influence.dtype,
        )
        clipped_influence = jnp.clip(
            replacement_influence,
            -clip_value,
            clip_value,
        )
        clipped_influence = jnp.where(
            clip_value > 0.0,
            clipped_influence,
            replacement_influence,
        )
        raw_influence_loss = _global_mean(
            replacement_influence,
            axis_name=axis_name,
        )
        replacement_influence_std = jnp.sqrt(
            _global_mean(
                jnp.square(
                    replacement_influence - raw_influence_loss,
                ),
                axis_name=axis_name,
            )
        )
        influence_loss = _global_mean(
            clipped_influence,
            axis_name=axis_name,
        )
        label_preservation_loss = _global_mean(
            optax.softmax_cross_entropy_with_integer_labels(
                augmented_logits,
                labels,
            ),
            axis_name=axis_name,
        )
        identity_l2 = _global_mean(
            jnp.square(
                augmented_images - raw_images,
            ),
            axis_name=axis_name,
        )
        loss = (
            influence_loss
            + identity_l2_weight * identity_l2
            + label_preservation_weight * label_preservation_loss
        )
        augmented_accuracy = _global_mean(
            (
                jnp.argmax(
                    augmented_logits,
                    axis=-1,
                )
                == labels
            ).astype(
                jnp.float32,
            ),
            axis_name=axis_name,
        )
        original_accuracy = _global_mean(
            (
                jnp.argmax(
                    original_logits,
                    axis=-1,
                )
                == labels
            ).astype(
                jnp.float32,
            ),
            axis_name=axis_name,
        )
        accuracy_retention = jnp.where(
            original_accuracy > 1.0e-6,
            augmented_accuracy / original_accuracy,
            1.0,
        )

        return loss, {
            "loss": loss,
            "i_aug_loss": influence_loss,
            "raw_i_aug_loss": raw_influence_loss,
            "replacement_influence_std": replacement_influence_std,
            "label_preservation_loss": label_preservation_loss,
            "augmented_influence": _global_mean(
                augmented_influence,
                axis_name=axis_name,
            ),
            "original_influence": _global_mean(
                original_influence,
                axis_name=axis_name,
            ),
            "estimated_val_loss_reduction": -influence_loss,
            "identity_l2": identity_l2,
            "accuracy_on_augmented": augmented_accuracy,
            "accuracy_on_original": original_accuracy,
            "accuracy_retention": accuracy_retention,
            "tau_abs_mean": _global_mean(
                jnp.abs(
                    aux["tau"],
                ),
                axis_name=axis_name,
            ),
            "tau_pre_dropout_abs_mean": _global_mean(
                jnp.abs(
                    aux["tau_pre_dropout"],
                ),
                axis_name=axis_name,
            ),
            "tau_saturation_fraction": _global_mean(
                (
                    jnp.abs(
                        aux["tau_pre_dropout"],
                    )
                    >= 0.95
                ).astype(
                    jnp.float32,
                ),
                axis_name=axis_name,
            ),
            "spatial_oob_fraction": _global_mean(
                aux["spatial_oob_fraction"],
                axis_name=axis_name,
            ),
            "appearance_out_of_range_fraction": _global_mean(
                aux.get(
                    "appearance_out_of_range_fraction",
                    jnp.asarray(0.0, dtype=raw_images.dtype),
                ),
                axis_name=axis_name,
            ),
            "augmented_out_of_range_fraction": _global_mean(
                aux["augmented_out_of_range_fraction"],
                axis_name=axis_name,
            ),
            "augmented_l1": _global_mean(
                aux["augmented_l1"],
                axis_name=axis_name,
            ),
        }

    (
        _,
        metrics,
    ), gradients = jax.value_and_grad(
        loss_fn,
        has_aux=True,
    )(
        augment_state.params,
    )
    gradients = _average_gradients(
        gradients,
        axis_name=axis_name,
    )
    metrics = {
        **metrics,
        "gradient_global_norm": _tree_global_norm(
            gradients,
        ),
    }
    augment_state = augment_state.apply_gradients(
        grads=gradients,
    )

    return augment_state, metrics


@jax.jit
def augnet_influence_train_step(
    augment_state: AugmentTrainState,
    classifier_state,
    raw_images: jnp.ndarray,
    labels: jnp.ndarray,
    s_test: dict[str, jnp.ndarray],
    rng: jax.Array,
    mean: jnp.ndarray,
    std: jnp.ndarray,
    identity_l2_weight: float = 0.0,
    influence_clip_value: float = 0.0,
    label_preservation_weight: float = 0.0,
):
    """Run one single-device influence update."""
    return _augnet_influence_train_step(
        augment_state=augment_state,
        classifier_state=classifier_state,
        raw_images=raw_images,
        labels=labels,
        s_test=s_test,
        rng=rng,
        mean=mean,
        std=std,
        identity_l2_weight=identity_l2_weight,
        influence_clip_value=influence_clip_value,
        label_preservation_weight=label_preservation_weight,
    )


@partial(
    jax.pmap,
    axis_name="batch",
    in_axes=(0, 0, 0, 0, 0, 0, None, None, None, None, None),
)
def parallel_augnet_influence_train_step(
    augment_state: AugmentTrainState,
    classifier_state,
    raw_images: jnp.ndarray,
    labels: jnp.ndarray,
    s_test: dict[str, jnp.ndarray],
    rng: jax.Array,
    mean: jnp.ndarray,
    std: jnp.ndarray,
    identity_l2_weight: float = 0.0,
    influence_clip_value: float = 0.0,
    label_preservation_weight: float = 0.0,
):
    """Run one globally synchronized influence update."""
    return _augnet_influence_train_step(
        augment_state=augment_state,
        classifier_state=classifier_state,
        raw_images=raw_images,
        labels=labels,
        s_test=s_test,
        rng=rng,
        mean=mean,
        std=std,
        identity_l2_weight=identity_l2_weight,
        influence_clip_value=influence_clip_value,
        label_preservation_weight=label_preservation_weight,
        axis_name="batch",
    )


def _classifier_retrain_step(
    task_state,
    augment_state: AugmentTrainState,
    images: jnp.ndarray,
    labels: jnp.ndarray,
    rng: jax.Array,
    mean: jnp.ndarray,
    std: jnp.ndarray,
    learned_aug_probability: float,
    axis_name: str | None = None,
    sync_batch_stats: bool = False,
) -> tuple[Any, jnp.ndarray, jnp.ndarray, dict[str, jnp.ndarray]]:
    """Train a fresh classifier on base views optionally transformed by E/G."""
    augment_rng, selection_rng, dropout_rng = jax.random.split(
        rng,
        3,
    )
    base_raw_images = denormalize_images(
        images=images,
        mean=mean,
        std=std,
    )
    augmented_raw_images, augment_aux = augment_state.apply_fn(
        {
            "params": augment_state.params,
        },
        base_raw_images,
        training=True,
        return_aux=True,
        rngs={
            "dropout": augment_rng,
        },
    )
    probability = jnp.clip(
        jnp.asarray(
            learned_aug_probability,
            dtype=images.dtype,
        ),
        0.0,
        1.0,
    )
    mask_shape = (
        images.shape[0],
        *(
            (1,)
            * (images.ndim - 1)
        ),
    )
    use_augmented = jax.random.bernoulli(
        selection_rng,
        p=probability,
        shape=mask_shape,
    )
    selected_raw_images = jnp.where(
        use_augmented,
        augmented_raw_images,
        base_raw_images,
    )
    selected_images = normalize_images(
        images=selected_raw_images,
        mean=mean,
        std=std,
    )

    def loss_fn(params):
        """Compute hard-label loss for the fresh classifier update."""
        logits, updates = task_state.apply_fn(
            {
                "params": params,
                "batch_stats": task_state.batch_stats,
            },
            selected_images,
            training=True,
            mutable=[
                "batch_stats",
            ],
            rngs={
                "dropout": dropout_rng,
            },
            sync_batch_stats=sync_batch_stats,
        )
        loss = _global_mean(
            optax.softmax_cross_entropy_with_integer_labels(
                logits,
                labels.astype(
                    jnp.int32,
                ),
            ),
            axis_name=axis_name,
        )

        return loss, (
            logits,
            updates["batch_stats"],
        )

    (
        loss,
        (
            logits,
            batch_stats,
        ),
    ), gradients = jax.value_and_grad(
        loss_fn,
        has_aux=True,
    )(
        task_state.params,
    )
    gradients = _average_gradients(
        gradients,
        axis_name=axis_name,
    )
    task_state = task_state.apply_gradients(
        grads=gradients,
        batch_stats=batch_stats,
    )
    accuracy = _global_mean(
        jnp.argmax(
            logits,
            axis=-1,
        )
        == labels,
        axis_name=axis_name,
    )

    return task_state, loss, accuracy, {
        "ifaugnet_learned_aug_fraction": _global_mean(
            use_augmented.astype(
                jnp.float32,
            ),
            axis_name=axis_name,
        ),
        "ifaugnet_learned_aug_probability": probability,
        "ifaugnet_aug_l1": _global_mean(
            jnp.abs(
                augmented_raw_images - base_raw_images,
            ),
            axis_name=axis_name,
        ),
        "ifaugnet_spatial_oob_fraction": _global_mean(
            augment_aux["spatial_oob_fraction"],
            axis_name=axis_name,
        ),
        "ifaugnet_appearance_out_of_range_fraction": _global_mean(
            augment_aux.get(
                "appearance_out_of_range_fraction",
                jnp.asarray(0.0, dtype=images.dtype),
            ),
            axis_name=axis_name,
        ),
        "ifaugnet_augmented_out_of_range_fraction": _global_mean(
            augment_aux["augmented_out_of_range_fraction"],
            axis_name=axis_name,
        ),
    }


@jax.jit
def classifier_retrain_step(
    task_state,
    augment_state: AugmentTrainState,
    images: jnp.ndarray,
    labels: jnp.ndarray,
    rng: jax.Array,
    mean: jnp.ndarray,
    std: jnp.ndarray,
    learned_aug_probability: float,
):
    """Run one single-device fresh-classifier update."""
    return _classifier_retrain_step(
        task_state=task_state,
        augment_state=augment_state,
        images=images,
        labels=labels,
        rng=rng,
        mean=mean,
        std=std,
        learned_aug_probability=learned_aug_probability,
    )


@partial(
    jax.pmap,
    axis_name="batch",
    in_axes=(0, 0, 0, 0, 0, None, None, None),
)
def parallel_classifier_retrain_step(
    task_state,
    augment_state: AugmentTrainState,
    images: jnp.ndarray,
    labels: jnp.ndarray,
    rng: jax.Array,
    mean: jnp.ndarray,
    std: jnp.ndarray,
    learned_aug_probability: float,
):
    """Retrain with global gradients and synchronized BatchNorm."""
    return _classifier_retrain_step(
        task_state=task_state,
        augment_state=augment_state,
        images=images,
        labels=labels,
        rng=rng,
        mean=mean,
        std=std,
        learned_aug_probability=learned_aug_probability,
        axis_name="batch",
        sync_batch_stats=True,
    )


class AugNetRetrainStrategy:
    """Shared-loop strategy for fresh-classifier retraining with frozen E/G."""

    def __init__(
        self,
        augment_state: AugmentTrainState,
        mean: tuple[float, ...],
        std: tuple[float, ...],
        learned_aug_probability: float,
        distributed: bool = False,
        sync_batch_stats: bool = False,
    ) -> None:
        """Store frozen method state and normalization constants."""
        if distributed and not sync_batch_stats:
            raise ValueError(
                "Distributed IF-AugNet retraining requires synchronized "
                "BatchNorm statistics."
            )
        self.distributed = distributed
        self.augment_state = (
            jax_utils.replicate(
                augment_state,
            )
            if distributed
            else augment_state
        )
        self.mean, self.std = normalization_arrays(
            mean=mean,
            std=std,
        )
        self.learned_aug_probability = learned_aug_probability

    def train_step(
        self,
        task_state,
        images: jnp.ndarray,
        labels: jnp.ndarray,
        rng: jax.Array,
    ) -> tuple[Any, jnp.ndarray, jnp.ndarray, dict[str, jnp.ndarray]]:
        """Run one fresh-classifier update through the shared epoch loop."""
        step_fn = (
            parallel_classifier_retrain_step
            if self.distributed
            else classifier_retrain_step
        )

        return step_fn(
            task_state,
            self.augment_state,
            images,
            labels,
            rng,
            self.mean,
            self.std,
            self.learned_aug_probability,
        )
