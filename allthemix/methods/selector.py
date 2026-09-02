from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from allthemix.methods.baseline import baseline_mixer
from allthemix.methods.catchupmix import catchupmix
from allthemix.methods.cutmix import cutmix
from allthemix.methods.fmix import fmix
from allthemix.methods.guidedmixup import guided_sr, guidedmixup
from allthemix.methods.mixup import mixup
from allthemix.methods.output import MixOutput
from allthemix.methods.resizemix import resizemix
from allthemix.methods.saliencymix import saliencymix
from allthemix.methods.utils.validation import (
    normalize_method_name,
    validate_num_classes,
    validate_odd_positive_int,
    validate_positive,
    validate_positive_int,
    validate_probability,
    validate_scope_range,
)

MixerFn = Callable[
    [
        jax.Array,
        jnp.ndarray,
        jnp.ndarray,
        dict[str, jnp.ndarray] | None,
    ],
    MixOutput,
]


def get_mixer(
    name: str,
    num_classes: int,
    mixup_alpha: float = 1.0,
    cutmix_alpha: float = 1.0,
    cutmix_prob: float = 1.0,
    cutmix_no_repeat: bool = False,
    cutmix_variant: str = "standard",
    cutmix_per_sample_lam: bool = False,
    cutmix_min_lam: float = 0.0,
    saliencymix_alpha: float = 1.0,
    saliencymix_prob: float = 1.0,
    saliencymix_per_sample: bool = False,
    fmix_alpha: float = 1.0,
    fmix_decay: float = 3.0,
    fmix_prob: float = 1.0,
    fmix_per_sample: bool = False,
    fmix_no_repeat: bool = False,
    resizemix_scope_min: float = 0.1,
    resizemix_scope_max: float = 0.8,
    resizemix_prob: float = 1.0,
    resizemix_per_sample: bool = False,
    guidedmixup_alpha: float = 1.0,
    guidedmixup_prob: float = 1.0,
    guidedmixup_blur_kernel: int = 7,
    guidedmixup_condition: str = "greedy",
    catchupmix_alpha: float = 1.0,
    catchupmix_cutmix_alpha: float = 1.0,
    catchupmix_num_layers: int = 5,
    catchupmix_no_repeat: bool = False,
    alpha: float | None = None,
) -> MixerFn:
    """Get mixer."""
    validate_num_classes(
        num_classes,
    )

    if alpha is not None:
        mixup_alpha = alpha
        cutmix_alpha = alpha
        saliencymix_alpha = alpha
        fmix_alpha = alpha
        guidedmixup_alpha = alpha
        catchupmix_alpha = alpha

    mixer_name = normalize_method_name(
        name,
    )
    guidedmixup_condition = normalize_method_name(
        guidedmixup_condition,
    )

    if mixer_name in {
        "baseline",
        "alia",
        "diffusemix",
        "diffuse_mix",
        "saspa",
    }:

        def mixer(
            rng: jax.Array,
            images: jnp.ndarray,
            labels: jnp.ndarray,
            aux_info: dict[str, jnp.ndarray] | None = None,
        ) -> MixOutput:
            """Run the selected mixer wrapper."""
            del aux_info

            return baseline_mixer(
                rng=rng,
                images=images,
                labels=labels,
                num_classes=num_classes,
            )

        return mixer

    if mixer_name == "mixup":
        validate_positive(
            name="mixup_alpha",
            value=mixup_alpha,
        )

        def mixer(
            rng: jax.Array,
            images: jnp.ndarray,
            labels: jnp.ndarray,
            aux_info: dict[str, jnp.ndarray] | None = None,
        ) -> MixOutput:
            """Run the selected mixer wrapper."""
            paired_images = None
            paired_labels = None
            paired_perm = None

            if aux_info is not None:
                paired_images = aux_info.get(
                    "paired_images",
                )
                paired_labels = aux_info.get(
                    "paired_labels",
                )
                paired_perm = aux_info.get(
                    "paired_perm",
                )

            return mixup(
                rng=rng,
                images=images,
                labels=labels,
                num_classes=num_classes,
                alpha=mixup_alpha,
                paired_images=paired_images,
                paired_labels=paired_labels,
                paired_perm=paired_perm,
            )

        return mixer

    if mixer_name in (
        "cutmix",
        "cutmix_sumix",
    ):
        validate_positive(
            name="cutmix_alpha",
            value=cutmix_alpha,
        )
        validate_probability(
            name="cutmix_prob",
            value=cutmix_prob,
        )

        def mixer(
            rng: jax.Array,
            images: jnp.ndarray,
            labels: jnp.ndarray,
            aux_info: dict[str, jnp.ndarray] | None = None,
        ) -> MixOutput:
            """Run the selected mixer wrapper."""
            paired_images = None
            paired_labels = None
            paired_perm = None

            if aux_info is not None:
                paired_images = aux_info.get(
                    "paired_images",
                )
                paired_labels = aux_info.get(
                    "paired_labels",
                )
                paired_perm = aux_info.get(
                    "paired_perm",
                )

            return cutmix(
                rng=rng,
                images=images,
                labels=labels,
                num_classes=num_classes,
                alpha=cutmix_alpha,
                prob=cutmix_prob,
                no_repeat=cutmix_no_repeat or mixer_name == "cutmix_sumix",
                paired_images=paired_images,
                paired_labels=paired_labels,
                paired_perm=paired_perm,
                variant=cutmix_variant,
                per_sample_lam=cutmix_per_sample_lam,
                min_lam=cutmix_min_lam,
            )

        return mixer

    if mixer_name == "saliencymix":
        validate_positive(
            name="saliencymix_alpha",
            value=saliencymix_alpha,
        )
        validate_probability(
            name="saliencymix_prob",
            value=saliencymix_prob,
        )

        def mixer(
            rng: jax.Array,
            images: jnp.ndarray,
            labels: jnp.ndarray,
            aux_info: dict[str, jnp.ndarray] | None = None,
        ) -> MixOutput:
            """Run the selected mixer wrapper."""
            if aux_info is None or "saliency_maps" not in aux_info:
                raise ValueError(
                    "SaliencyMix requires aux_info['saliency_maps']. "
                    "Use method saliencymix with basic_aug: false, "
                    "a paired sal_aug_recipe, and generated saliency maps.",
                )

            return saliencymix(
                rng=rng,
                images=images,
                labels=labels,
                saliency_maps=aux_info["saliency_maps"],
                num_classes=num_classes,
                alpha=saliencymix_alpha,
                prob=saliencymix_prob,
                per_sample=saliencymix_per_sample,
                paired_images=aux_info.get(
                    "paired_images",
                ),
                paired_labels=aux_info.get(
                    "paired_labels",
                ),
                paired_saliency_maps=aux_info.get(
                    "paired_saliency_maps",
                ),
            )

        return mixer

    if mixer_name == "fmix":
        validate_positive(
            name="fmix_alpha",
            value=fmix_alpha,
        )
        validate_positive(
            name="fmix_decay",
            value=fmix_decay,
        )
        validate_probability(
            name="fmix_prob",
            value=fmix_prob,
        )

        def mixer(
            rng: jax.Array,
            images: jnp.ndarray,
            labels: jnp.ndarray,
            aux_info: dict[str, jnp.ndarray] | None = None,
        ) -> MixOutput:
            """Run the selected mixer wrapper."""
            paired_images = None
            paired_labels = None

            if aux_info is not None:
                paired_images = aux_info.get(
                    "paired_images",
                )
                paired_labels = aux_info.get(
                    "paired_labels",
                )

            return fmix(
                rng=rng,
                images=images,
                labels=labels,
                num_classes=num_classes,
                alpha=fmix_alpha,
                decay_power=fmix_decay,
                prob=fmix_prob,
                per_sample=fmix_per_sample,
                no_repeat=fmix_no_repeat,
                paired_images=paired_images,
                paired_labels=paired_labels,
            )

        return mixer

    if mixer_name == "resizemix":
        validate_scope_range(
            min_name="resizemix_scope_min",
            min_value=resizemix_scope_min,
            max_name="resizemix_scope_max",
            max_value=resizemix_scope_max,
        )
        validate_probability(
            name="resizemix_prob",
            value=resizemix_prob,
        )

        def mixer(
            rng: jax.Array,
            images: jnp.ndarray,
            labels: jnp.ndarray,
            aux_info: dict[str, jnp.ndarray] | None = None,
        ) -> MixOutput:
            """Run the selected mixer wrapper."""
            paired_images = None
            paired_labels = None

            if aux_info is not None:
                paired_images = aux_info.get(
                    "paired_images",
                )
                paired_labels = aux_info.get(
                    "paired_labels",
                )

            return resizemix(
                rng=rng,
                images=images,
                labels=labels,
                num_classes=num_classes,
                scope_min=resizemix_scope_min,
                scope_max=resizemix_scope_max,
                prob=resizemix_prob,
                per_sample=resizemix_per_sample,
                paired_images=paired_images,
                paired_labels=paired_labels,
            )

        return mixer

    if mixer_name == "guidedmixup":
        validate_positive(
            name="guidedmixup_alpha",
            value=guidedmixup_alpha,
        )
        validate_probability(
            name="guidedmixup_prob",
            value=guidedmixup_prob,
        )
        validate_odd_positive_int(
            name="guidedmixup_blur_kernel",
            value=guidedmixup_blur_kernel,
        )
        if guidedmixup_condition not in {"random", "greedy"}:
            raise ValueError(
                "guidedmixup_condition must be one of: random, greedy. "
                f"Got {guidedmixup_condition}.",
            )

        def mixer(
            rng: jax.Array,
            images: jnp.ndarray,
            labels: jnp.ndarray,
            aux_info: dict[str, jnp.ndarray] | None = None,
        ) -> MixOutput:
            """Run the selected mixer wrapper."""
            if aux_info is None or "saliency_maps" not in aux_info:
                raise ValueError(
                    "GuidedMixup requires aux_info['saliency_maps']. "
                    "Use method guidedmixup with basic_aug: false, "
                    "a paired sal_aug_recipe, and generated saliency maps. "
                    "Use guided_sr if you want online spectral-residual saliency.",
                )

            return guidedmixup(
                rng=rng,
                images=images,
                labels=labels,
                saliency_maps=aux_info["saliency_maps"],
                num_classes=num_classes,
                alpha=guidedmixup_alpha,
                prob=guidedmixup_prob,
                blur_kernel=guidedmixup_blur_kernel,
                condition=guidedmixup_condition,
            )

        return mixer

    if mixer_name in (
        "guided_sr",
        "guidedmixup_sr",
    ):
        validate_positive(
            name="guidedmixup_alpha",
            value=guidedmixup_alpha,
        )
        validate_probability(
            name="guidedmixup_prob",
            value=guidedmixup_prob,
        )
        validate_odd_positive_int(
            name="guidedmixup_blur_kernel",
            value=guidedmixup_blur_kernel,
        )
        if guidedmixup_condition not in {"random", "greedy"}:
            raise ValueError(
                "guidedmixup_condition must be one of: random, greedy. "
                f"Got {guidedmixup_condition}.",
            )

        def mixer(
            rng: jax.Array,
            images: jnp.ndarray,
            labels: jnp.ndarray,
            aux_info: dict[str, jnp.ndarray] | None = None,
        ) -> MixOutput:
            """Run the selected mixer wrapper."""
            del aux_info

            return guided_sr(
                rng=rng,
                images=images,
                labels=labels,
                num_classes=num_classes,
                alpha=guidedmixup_alpha,
                prob=guidedmixup_prob,
                blur_kernel=guidedmixup_blur_kernel,
                condition=guidedmixup_condition,
            )

        return mixer

    if mixer_name in (
        "catchupmix",
        "catchup_mix",
        "catch_up_mix",
    ):
        validate_positive(
            name="catchupmix_alpha",
            value=catchupmix_alpha,
        )
        validate_positive(
            name="catchupmix_cutmix_alpha",
            value=catchupmix_cutmix_alpha,
        )
        validate_positive_int(
            name="catchupmix_num_layers",
            value=catchupmix_num_layers,
        )

        def mixer(
            rng: jax.Array,
            images: jnp.ndarray,
            labels: jnp.ndarray,
            aux_info: dict[str, jnp.ndarray] | None = None,
        ) -> MixOutput:
            """Run the selected mixer wrapper."""
            del aux_info

            return catchupmix(
                rng=rng,
                images=images,
                labels=labels,
                num_classes=num_classes,
                alpha=catchupmix_alpha,
                cutmix_alpha=catchupmix_cutmix_alpha,
                cutmix_variant=cutmix_variant,
                cutmix_per_sample_lam=cutmix_per_sample_lam,
                cutmix_min_lam=cutmix_min_lam,
                num_feature_layers=catchupmix_num_layers,
                no_repeat=catchupmix_no_repeat,
            )

        return mixer

    raise ValueError(f"Unsupported mixer: {name}")
