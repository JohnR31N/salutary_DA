from __future__ import annotations

import jax
import numpy as np

from allthemix.competitors.ifaugnet.models import AugmentationNetwork
from allthemix.competitors.ifaugnet.steps import create_augment_state
from allthemix.competitors.ifaugnet.visualize import (
    SPATIAL_CHANNELS,
    apply_learned_policy,
    appearance_channel_names,
    build_appearance_basis_results,
    build_spatial_basis_results,
    save_basis_grid,
)
from allthemix.utils.checkpoint import save_state_file


def _probe_image(size: int = 12) -> np.ndarray:
    """Create an asymmetric RGB image that exposes direction and color errors."""
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, size, dtype=np.float32),
        np.linspace(0.0, 1.0, size, dtype=np.float32),
        indexing="ij",
    )
    return np.stack((xx, yy, 0.25 + 0.5 * xx * yy), axis=-1)


def test_ifaugnet_channel_names_match_decoder_layout() -> None:
    names = appearance_channel_names(3)

    assert len(SPATIAL_CHANNELS) == 6
    assert names == (
        "W[R<-R]",
        "W[R<-G]",
        "W[R<-B]",
        "W[G<-R]",
        "W[G<-G]",
        "W[G<-B]",
        "W[B<-R]",
        "W[B<-G]",
        "W[B<-B]",
        "b[R]",
        "b[G]",
        "b[B]",
    )


def test_ifaugnet_paper_basis_uses_all_production_channels() -> None:
    image = _probe_image()

    spatial = build_spatial_basis_results(
        image,
        parameterization="paper",
        spatial_scale=0.20,
        smoothing_kernel=4,
        displacement_fraction=0.08,
    )
    appearance = build_appearance_basis_results(
        image,
        parameterization="paper",
        appearance_scale=0.25,
        smoothing_kernel=4,
        weight_delta=0.25,
        bias_delta=0.15,
    )

    assert len(spatial) == 6
    assert len(appearance) == 12
    for result in (*spatial, *appearance):
        assert result.negative.shape == image.shape
        assert result.positive.shape == image.shape
        assert np.isfinite(result.negative).all()
        assert np.isfinite(result.positive).all()
        assert result.negative_l1 > 0.0
        assert result.positive_l1 > 0.0


def test_ifaugnet_guarded_basis_and_plot_are_finite(tmp_path) -> None:
    image = _probe_image(8)
    spatial = build_spatial_basis_results(
        image,
        parameterization="guarded",
        spatial_scale=0.20,
        smoothing_kernel=1,
        displacement_fraction=0.05,
    )
    output = tmp_path / "spatial.png"

    save_basis_grid(
        image,
        spatial,
        output,
        title="IF-AugNet spatial test",
        dpi=60,
    )

    assert output.is_file()
    assert output.stat().st_size > 0
    assert all(result.negative_out_of_range == 0.0 for result in spatial)
    assert all(result.positive_out_of_range == 0.0 for result in spatial)


def test_ifaugnet_learned_policy_visualization_restores_checkpoint(tmp_path) -> None:
    config = {
        "ifaugnet_tau_dim": 8,
        "ifaugnet_tau_dropout": 0.0,
        "ifaugnet_spatial_scale": 0.05,
        "ifaugnet_appearance_scale": 0.05,
        "ifaugnet_smoothing_kernel": 1,
        "ifaugnet_use_appearance": True,
        "ifaugnet_encoder_widths": (4, 8),
        "ifaugnet_decoder_widths": (8,),
        "ifaugnet_decoder_base_width": 8,
        "ifaugnet_transform_parameterization": "guarded",
        "ifaugnet_composition": "serial",
        "ifaugnet_architecture": "custom",
        "ifaugnet_learning_rate": 1.0e-4,
    }
    model = AugmentationNetwork(
        image_size=8,
        channels=3,
        tau_dim=8,
        tau_dropout=0.0,
        spatial_scale=0.05,
        appearance_scale=0.05,
        smoothing_kernel=1,
        encoder_widths=(4, 8),
        decoder_widths=(8,),
        decoder_base_width=8,
        parameterization="guarded",
        composition="serial",
        architecture="custom",
    )
    state = create_augment_state(
        rng=jax.random.PRNGKey(0),
        model=model,
        input_shape=(1, 8, 8, 3),
        learning_rate=1.0e-4,
    )
    checkpoint = save_state_file(state, tmp_path, "ifaugnet_influence_final")
    images = np.stack((_probe_image(8), _probe_image(8)[::-1]), axis=0)

    augmented, aux, loaded = apply_learned_policy(
        images,
        config=config,
        checkpoint_path=checkpoint,
        seed=0,
        training=False,
    )

    assert augmented.shape == images.shape
    assert np.isfinite(augmented).all()
    assert aux["fields"].shape == (2, 8, 8, 18)
    assert aux["sample_grid"].shape == (2, 8, 8, 2)
    assert any(path.startswith("params") for path in loaded)
