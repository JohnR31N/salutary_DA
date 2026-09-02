from __future__ import annotations

import jax.numpy as jnp

from allthemix.methods.output import MixOutput
from allthemix.methods.utils.validation import normalize_method_name
from allthemix.training.losses.cross_entropy import cross_entropy
from allthemix.training.losses.mixup_loss import mixup_loss


def _get_baseline_labels(
    mixer_output: MixOutput,
) -> jnp.ndarray:
    """Extract hard labels for non-mixing methods."""
    return mixer_output.labels_a


def _get_mix_labels_and_lambda(
    mixer_output: tuple[jnp.ndarray, ...] | MixOutput,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Extract paired labels and lambda for mixing methods."""
    if isinstance(mixer_output, MixOutput):
        return (
            mixer_output.labels_a,
            mixer_output.labels_b,
            mixer_output.lam,
        )

    if hasattr(
        mixer_output,
        "labels_a",
    ):
        return (
            mixer_output.labels_a,
            mixer_output.labels_b,
            mixer_output.lam,
        )

    _, labels_a, labels_b, lam = mixer_output

    return labels_a, labels_b, lam


def compute_train_loss_and_targets(
    method: str,
    logits: jnp.ndarray,
    mixer_output: MixOutput,
    num_classes: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute method-specific training loss and accuracy targets."""
    method_name = normalize_method_name(method)

    if method_name in {
        "baseline",
        "alia",
        "diffusemix",
        "diffuse_mix",
        "saspa",
    }:
        labels = _get_baseline_labels(
            mixer_output=mixer_output,
        )

        loss = cross_entropy(
            logits=logits,
            labels=labels,
            num_classes=num_classes,
        )

        target_labels = labels

        return loss, target_labels

    if method_name in (
        "mixup",
        "cutmix",
        "saliencymix",
        "fmix",
        "resizemix",
        "guidedmixup",
        "guided_sr",
        "guidedmixup_sr",
        "catchupmix",
        "catchup_mix",
        "catch_up_mix",
    ):
        labels_a, labels_b, lam = _get_mix_labels_and_lambda(
            mixer_output=mixer_output,
        )

        loss = mixup_loss(
            logits=logits,
            labels_a=labels_a,
            labels_b=labels_b,
            num_classes=num_classes,
            lam=lam,
        )

        target_labels = labels_a

        return loss, target_labels

    raise ValueError(f"Unsupported training method: {method}")
