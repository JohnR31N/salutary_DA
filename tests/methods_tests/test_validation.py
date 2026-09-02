from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from allthemix.cli.train import _validate_method_model_compatibility
from allthemix.methods.cutmix import cutmix
from allthemix.methods.guidedmixup import guidedmixup
from allthemix.methods.mixup import mixup
from allthemix.methods.selector import get_mixer


def test_selector_rejects_invalid_mixup_alpha() -> None:
    """Verify that invalid MixUp alpha fails before training starts."""
    with pytest.raises(
        ValueError,
        match="mixup_alpha",
    ):
        get_mixer(
            name="mixup",
            num_classes=10,
            mixup_alpha=0.0,
        )


def test_selector_rejects_invalid_resizemix_scope() -> None:
    """Verify that invalid ResizeMix scope intervals fail early."""
    with pytest.raises(
        ValueError,
        match="resizemix_scope_min",
    ):
        get_mixer(
            name="resizemix",
            num_classes=10,
            resizemix_scope_min=0.8,
            resizemix_scope_max=0.1,
        )


def test_selector_rejects_even_guidedmixup_kernel() -> None:
    """Verify that GuidedMixup rejects even blur kernels at construction."""
    with pytest.raises(
        ValueError,
        match="guidedmixup_blur_kernel",
    ):
        get_mixer(
            name="guided_sr",
            num_classes=10,
            guidedmixup_blur_kernel=4,
        )


def test_saliencymix_requires_auxiliary_saliency_maps() -> None:
    """Verify that SaliencyMix reports the missing saliency handoff clearly."""
    mixer = get_mixer(
        name="saliencymix",
        num_classes=10,
    )

    with pytest.raises(
        ValueError,
        match="aux_info\\['saliency_maps'\\]",
    ):
        mixer(
            rng=jax.random.PRNGKey(0),
            images=jnp.ones((4, 32, 32, 3)),
            labels=jnp.arange(4),
        )


def test_mixup_rejects_label_batch_mismatch() -> None:
    """Verify that methods reject image/label batch mismatches."""
    with pytest.raises(
        ValueError,
        match="image/label batch mismatch",
    ):
        mixup(
            rng=jax.random.PRNGKey(0),
            images=jnp.ones((4, 32, 32, 3)),
            labels=jnp.arange(3),
            num_classes=10,
        )


def test_cutmix_no_repeat_rejects_single_sample_batch() -> None:
    """Verify that no-repeat pairing requires at least two samples."""
    with pytest.raises(
        ValueError,
        match="no_repeat requires batch size",
    ):
        cutmix(
            rng=jax.random.PRNGKey(0),
            images=jnp.ones((1, 32, 32, 3)),
            labels=jnp.arange(1),
            num_classes=10,
            no_repeat=True,
        )


def test_guidedmixup_rejects_saliency_spatial_mismatch() -> None:
    """Verify that GuidedMixup detects saliency/image size mismatches."""
    with pytest.raises(
        ValueError,
        match="saliency maps must match image height and width",
    ):
        guidedmixup(
            rng=jax.random.PRNGKey(0),
            images=jnp.ones((4, 32, 32, 3)),
            labels=jnp.arange(4),
            saliency_maps=jnp.ones((4, 16, 16)),
            num_classes=10,
            blur_kernel=3,
        )


def test_catchupmix_rejects_more_layers_than_backbone_hooks() -> None:
    """Verify that CatchUpMix cannot request non-existent feature hooks."""
    args = SimpleNamespace(
        model="wide_resnet28_10",
        catchupmix_num_layers=5,
    )

    with pytest.raises(
        ValueError,
        match="exposes 4 feature-hook layers",
    ):
        _validate_method_model_compatibility(
            args=args,
            method_name="catchupmix",
        )


def test_catchupmix_accepts_matching_backbone_hook_count() -> None:
    """Verify that matching CatchUpMix/backbone layer counts pass."""
    args = SimpleNamespace(
        model="preact_resnet18",
        catchupmix_num_layers=5,
    )

    _validate_method_model_compatibility(
        args=args,
        method_name="catchupmix",
    )
