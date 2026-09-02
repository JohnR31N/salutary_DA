from __future__ import annotations

import jax
import jax.numpy as jnp

from allthemix.training.losses.cross_entropy import hard_cross_entropy_per_sample


def _l2_normalize(
    x: jnp.ndarray,
    axis: int = -1,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """Normalize vectors by their L2 norm."""
    norm = jnp.sqrt(  # Compute sqrt(sum(x^2) + eps) for stable normalization.
        jnp.sum(
            jnp.square(
                x,
            ),
            axis=axis,
            keepdims=True,
        )
        + eps
    )

    return x / norm  # Scale each vector to unit L2 length.


def _as_batch_lambda(
    lam: jnp.ndarray,
    batch_size: int,
) -> jnp.ndarray:
    """Convert scalar or column lambda into a batch vector."""
    lam = jnp.asarray(
        lam,
        dtype=jnp.float32,
    )

    if lam.ndim == 0:
        lam = jnp.full(
            (
                batch_size,
            ),
            lam,
            dtype=jnp.float32,
        )

    if lam.ndim == 2 and lam.shape[-1] == 1:
        lam = lam.squeeze(
            -1,
        )

    return lam


def _gather_class_values(
    values: jnp.ndarray,
    labels: jnp.ndarray,
) -> jnp.ndarray:
    """Gather the value assigned to each sample's class label."""
    labels = labels.astype(
        jnp.int32,
    )

    gathered = jnp.take_along_axis(
        values,
        labels[:, None],
        axis=-1,
    )

    return gathered.squeeze(
        -1,
    )


def compute_sumix_lambda(
    logits_original: jnp.ndarray,
    logits_mixed: jnp.ndarray,
    uncertainty_original: jnp.ndarray,
    uncertainty_mixed: jnp.ndarray,
    labels_a: jnp.ndarray,
    labels_b: jnp.ndarray,
    area_lam: jnp.ndarray,
    perm: jnp.ndarray,
    semantic_scale: float = -1.0,
    eps: float = 1e-12,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute official-style SUMix adaptive lambda.

    This implements the SUMix IN-loss idea:

        semantic_original = softmax(logits_original)
        semantic_mixed = softmax(logits_mixed)

        alpha_a = || softmax(semantic_mixed - semantic_a) ||_2 * batch_size
        alpha_b = || softmax(semantic_mixed - semantic_b) ||_2 * batch_size

        beta_a = uncertainty_a + uncertainty_mixed
        beta_b = uncertainty_b + uncertainty_mixed

        INa = exp(-alpha_a)
        INb = exp(-alpha_b)

        lam_a = area_lam * INa[label_a]
        lam_b = (1 - area_lam) * INb[label_b]
        lam_a = lam_a / (lam_a + lam_b)
        lam_b = 1 - lam_a
    """

    batch_size = logits_mixed.shape[0]

    area_lam = _as_batch_lambda(
        lam=area_lam,
        batch_size=batch_size,
    )

    semantic_original = jax.nn.softmax(  # Convert original logits to class semantics.
        jax.lax.stop_gradient(
            logits_original,
        ),
        axis=-1,
    )

    semantic_mixed = jax.nn.softmax(  # Convert mixed logits to class semantics.
        jax.lax.stop_gradient(
            logits_mixed,
        ),
        axis=-1,
    )

    semantic_a = semantic_original
    semantic_b = semantic_original[perm]

    uncertainty_a = uncertainty_original
    uncertainty_b = uncertainty_original[perm]

    batch_indices = jnp.arange(
        batch_size,
    )

    semantic_a = semantic_a.at[
        batch_indices,
        labels_b.astype(
            jnp.int32,
        ),
    ].set(
        0.0,
    )

    semantic_b = semantic_b.at[
        batch_indices,
        labels_a.astype(
            jnp.int32,
        ),
    ].set(
        0.0,
    )

    alpha_a = _l2_normalize(  # Semantic distance from mixed sample to label-a source.
        jax.nn.softmax(
            semantic_mixed - semantic_a,
            axis=-1,
        ),
        axis=-1,
    )

    alpha_b = _l2_normalize(  # Semantic distance from mixed sample to label-b source.
        jax.nn.softmax(
            semantic_mixed - semantic_b,
            axis=-1,
        ),
        axis=-1,
    )

    if semantic_scale <= 0:
        semantic_scale = float(
            batch_size,
        )

    alpha_a = alpha_a * float(  # Scale semantic distance; official uses batch size.
        semantic_scale,
    )

    alpha_b = alpha_b * float(  # Scale semantic distance; official uses batch size.
        semantic_scale,
    )

    beta_a = uncertainty_a + uncertainty_mixed  # Combine source-a and mixed uncertainty.
    beta_b = uncertainty_b + uncertainty_mixed  # Combine source-b and mixed uncertainty.

    ina_feature = jnp.exp(  # Convert SUMix IN distance into source-a confidence.
        -(
            beta_a
            + alpha_a
        )
    )

    inb_feature = jnp.exp(  # Convert SUMix IN distance into source-b confidence.
        -(
            beta_b
            + alpha_b
        )
    )

    alpha_a_label = _gather_class_values(
        values=alpha_a,
        labels=labels_a,
    )

    alpha_b_label = _gather_class_values(
        values=alpha_b,
        labels=labels_b,
    )

    log_lam_a = (  # Combine area prior with source-a semantic confidence in log space.
        jnp.log(
            area_lam
            + eps,
        )
        - alpha_a_label
    )

    log_lam_b = (  # Combine area prior with source-b semantic confidence in log space.
        jnp.log(
            1.0
            - area_lam
            + eps,
        )
        - alpha_b_label
    )

    log_lam_pair = jnp.stack(
        [
            log_lam_a,
            log_lam_b,
        ],
        axis=-1,
    )

    lam_pair = jax.nn.softmax(  # Normalize the two adaptive lambda weights.
        log_lam_pair,
        axis=-1,
    )

    lam_a = lam_pair[
        :,
        0,
    ]

    lam_b = lam_pair[
        :,
        1,
    ]

    lam_a = jnp.clip(
        lam_a,
        1e-6,
        1.0 - 1e-6,
    )

    lam_b = jnp.clip(
        lam_b,
        1e-6,
        1.0 - 1e-6,
    )

    lam_sum = lam_a + lam_b  # Re-normalize after clipping.

    lam_a = lam_a / lam_sum  # Final adaptive weight for label_a.
    lam_b = lam_b / lam_sum  # Final adaptive weight for label_b.

    return lam_a, lam_b, ina_feature, inb_feature


def sumix_loss(
    logits_original: jnp.ndarray,
    logits_mixed: jnp.ndarray,
    uncertainty_original: jnp.ndarray,
    uncertainty_mixed: jnp.ndarray,
    labels_a: jnp.ndarray,
    labels_b: jnp.ndarray,
    area_lam: jnp.ndarray,
    perm: jnp.ndarray,
    num_classes: int,
    gamma: float = 0.5,
    semantic_scale: float = -1.0,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """
    Official-style SUMix loss.

    Returns:
        loss:
            Scalar training loss.

        metrics:
            Diagnostic metrics for debugging lambda and uncertainty.
    """

    batch_size = logits_mixed.shape[0]

    area_lam_batch = _as_batch_lambda(
        lam=area_lam,
        batch_size=batch_size,
    )

    lam_a, lam_b, ina_feature, inb_feature = compute_sumix_lambda(
        logits_original=logits_original,
        logits_mixed=logits_mixed,
        uncertainty_original=uncertainty_original,
        uncertainty_mixed=uncertainty_mixed,
        labels_a=labels_a,
        labels_b=labels_b,
        area_lam=area_lam_batch,
        perm=perm,
        semantic_scale=semantic_scale,
    )

    if semantic_scale <= 0:
        semantic_scale = float(
            batch_size,
        )

    ce_a = hard_cross_entropy_per_sample(
        logits=logits_mixed,
        labels=labels_a,
        num_classes=num_classes,
    )

    ce_b = hard_cross_entropy_per_sample(
        logits=logits_mixed,
        labels=labels_b,
        num_classes=num_classes,
    )

    ce_a_mean = jnp.mean(  # Match official criterion(..., avg_factor=batch_size).
        ce_a,
    )

    ce_b_mean = jnp.mean(  # Match official criterion(..., avg_factor=batch_size).
        ce_b,
    )

    classification_loss = (  # Official batch-mean CE weighted by adaptive lambdas.
        ce_a_mean * lam_a
        + ce_b_mean * lam_b
    )

    ina_loss = hard_cross_entropy_per_sample(
        logits=ina_feature,
        labels=labels_a,
        num_classes=num_classes,
    )

    inb_loss = hard_cross_entropy_per_sample(
        logits=inb_feature,
        labels=labels_b,
        num_classes=num_classes,
    )

    area_lam_mean = jnp.mean(  # CutMix uses one area lambda for the whole batch.
        area_lam_batch,
    )

    ina_loss_mean = jnp.mean(  # Match official criterion(..., avg_factor=batch_size).
        ina_loss,
    )

    inb_loss_mean = jnp.mean(  # Match official criterion(..., avg_factor=batch_size).
        inb_loss,
    )

    regularization_loss = (  # Official scalar IN regularizer weighted by area lambda.
        area_lam_mean * ina_loss_mean
        + (1.0 - area_lam_mean) * inb_loss_mean
    )

    total_loss = (  # Combine official classification and IN regularization losses.
        jnp.mean(
            classification_loss,
        )
        + gamma * regularization_loss
    )

    metrics = {
        "lam_a_mean": jnp.mean(
            lam_a,
        ),
        "lam_a_min": jnp.min(
            lam_a,
        ),
        "lam_a_max": jnp.max(
            lam_a,
        ),
        "lam_b_mean": jnp.mean(
            lam_b,
        ),
        "area_lam_mean": jnp.mean(
            area_lam_batch,
        ),
        "classification_loss": jnp.mean(
            classification_loss,
        ),
        "regularization_loss": jnp.asarray(
            regularization_loss,
        ),
        "uncertainty_original_mean": jnp.mean(
            uncertainty_original,
        ),
        "uncertainty_mixed_mean": jnp.mean(
            uncertainty_mixed,
        ),
        "semantic_scale": jnp.asarray(
            semantic_scale,
            dtype=jnp.float32,
        ),
        "ina_feature_mean": jnp.mean(
            ina_feature,
        ),
        "ina_feature_max": jnp.max(
            ina_feature,
        ),
        "inb_feature_mean": jnp.mean(
            inb_feature,
        ),
        "inb_feature_max": jnp.max(
            inb_feature,
        ),
    }

    return total_loss, metrics
