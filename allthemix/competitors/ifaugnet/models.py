from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax.numpy as jnp

from allthemix.competitors.ifaugnet.transforms import (
    apply_appearance_transform,
    apply_spatial_transform,
    combine_transforms,
)


def resolve_architecture(
    architecture: str,
    image_size: int,
) -> str:
    """Resolve the automatic IF-AugNet architecture profile."""
    if architecture == "auto":
        return "imagenet" if image_size == 224 else "cifar"
    if architecture not in {
        "custom",
        "cifar",
        "imagenet",
    }:
        raise ValueError(
            "architecture must be 'auto', 'custom', 'cifar', or 'imagenet'."
        )

    return architecture


def _encoder_spec(
    architecture: str,
    widths: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Return widths, kernels, and strides for one encoder/discriminator."""
    if architecture == "imagenet":
        return (
            (16, 16, 32, 32, 64, 64, 128),
            (7, 4, 4, 4, 4, 4, 4),
            (1, 2, 2, 2, 2, 2, 2),
        )
    if architecture == "cifar":
        return (
            (16, 32, 64, 128),
            (4, 4, 4, 4),
            (1, 2, 2, 2),
        )

    return (
        widths,
        tuple(
            4
            for _ in widths
        ),
        tuple(
            1 if index == 0 else 2
            for index in range(
                len(widths),
            )
        ),
    )


class ParameterYieldNetwork(nn.Module):
    """Encoder E that maps each image to a latent transform code tau."""

    tau_dim: int = 128
    widths: tuple[int, ...] = (16, 32, 64, 128)
    kernel_sizes: tuple[int, ...] = (4, 4, 4, 4)
    strides: tuple[int, ...] = (1, 2, 2, 2)

    @nn.compact
    def __call__(
        self,
        images: jnp.ndarray,
    ) -> jnp.ndarray:
        """Produce one bounded latent code for each input image."""
        features = images

        if not (
            len(self.widths)
            == len(self.kernel_sizes)
            == len(self.strides)
        ):
            raise ValueError(
                "Encoder widths, kernels, and strides must have equal length."
            )

        for index, (
            width,
            kernel_size,
            stride,
        ) in enumerate(
            zip(
                self.widths,
                self.kernel_sizes,
                self.strides,
                strict=True,
            )
        ):
            features = nn.Conv(
                features=width,
                kernel_size=(kernel_size, kernel_size),
                strides=(stride, stride),
                padding="SAME",
                name=f"conv_{index}",
            )(
                features,
            )
            features = nn.relu(
                features,
            )

        features = features.reshape(
            (features.shape[0], -1),
        )

        return nn.tanh(  # tau = tanh(E(x)).
            nn.Dense(
                self.tau_dim,
                name="tau",
            )(
                features,
            )
        )


class TransformationDecoder(nn.Module):
    """Decoder G that expands tau into dense transform fields."""

    image_size: int = 32
    channels: int = 3
    base_width: int = 128
    widths: tuple[int, ...] = (64, 32, 16)
    spatial_channels: int = 6
    use_appearance: bool = True
    architecture: str = "custom"

    @nn.compact
    def __call__(
        self,
        tau: jnp.ndarray,
    ) -> jnp.ndarray:
        """Decode one latent code into per-pixel transform parameters."""
        output_channels = self.spatial_channels

        if self.use_appearance:
            output_channels += self.channels * self.channels + self.channels

        architecture = resolve_architecture(
            self.architecture,
            self.image_size,
        )

        if architecture == "imagenet":
            if self.image_size != 224:
                raise ValueError(
                    "The ImageNet IF-AugNet decoder requires image_size=224."
                )
            decoder_widths = (64, 64, 32, 32, 16, 16)
            kernel_sizes = (7, 4, 4, 4, 4, 4)
            strides = (1, 2, 2, 2, 2, 2)
            start_size = 4
        else:
            decoder_widths = (
                (64, 32, 16)
                if architecture == "cifar"
                else self.widths
            )
            kernel_sizes = tuple(
                4
                for _ in decoder_widths
            )
            strides = tuple(
                2
                for _ in decoder_widths
            )
            upsample_factor = 2 ** len(
                decoder_widths,
            )
            start_size = max(
                1,
                (self.image_size + upsample_factor - 1) // upsample_factor,
            )
        features = nn.Dense(
            start_size * start_size * self.base_width,
            name="proj",
        )(
            tau,
        )
        features = nn.relu(
            features,
        )
        features = features.reshape(
            (
                tau.shape[0],
                start_size,
                start_size,
                self.base_width,
            )
        )

        for index, (
            width,
            kernel_size,
            stride,
        ) in enumerate(
            zip(
                decoder_widths,
                kernel_sizes,
                strides,
                strict=True,
            )
        ):
            features = nn.ConvTranspose(
                features=width,
                kernel_size=(kernel_size, kernel_size),
                strides=(stride, stride),
                padding=(
                    "VALID"
                    if architecture == "imagenet" and index == 0
                    else "SAME"
                ),
                name=f"deconv_{index}",
            )(
                features,
            )
            features = nn.relu(
                features,
            )

            if architecture == "imagenet" and index == 0:
                # The supplement maps 4x4 to 7x7 with a 7x7 stride-1
                # transposed convolution; its padding is unspecified.
                features = features[:, 1:8, 1:8, :]

        fields = nn.ConvTranspose(
            features=output_channels,
            kernel_size=(4, 4),
            strides=(1, 1),
            padding="SAME",
            name="out",
            kernel_init=nn.initializers.normal(
                stddev=1.0e-3,
            ),
            bias_init=nn.initializers.zeros,
        )(
            features,
        )

        return fields[:, : self.image_size, : self.image_size, :]


class AugmentationNetwork(nn.Module):
    """Coupled IF-AugNet parameter-yield encoder E and transform decoder G."""

    image_size: int = 32
    channels: int = 3
    tau_dim: int = 128
    tau_dropout: float = 0.5
    spatial_scale: float = 0.20
    appearance_scale: float = 0.25
    smoothing_kernel: int = 4
    use_appearance: bool = True
    encoder_widths: tuple[int, ...] = (16, 32, 64, 128)
    decoder_widths: tuple[int, ...] = (64, 32, 16)
    decoder_base_width: int = 128
    parameterization: str = "guarded"
    composition: str = "serial"
    architecture: str = "custom"

    @nn.compact
    def __call__(
        self,
        images: jnp.ndarray,
        training: bool = True,
        return_aux: bool = False,
        tau_override: Any = None,
    ) -> Any:
        """Generate differentiable augmented images and optional diagnostics."""
        architecture = resolve_architecture(
            self.architecture,
            self.image_size,
        )
        encoder_widths, encoder_kernels, encoder_strides = _encoder_spec(
            architecture,
            self.encoder_widths,
        )

        if tau_override is None:
            tau_pre_dropout = ParameterYieldNetwork(
                tau_dim=self.tau_dim,
                widths=encoder_widths,
                kernel_sizes=encoder_kernels,
                strides=encoder_strides,
                name="encoder",
            )(
                images,
            )
            tau = nn.Dropout(
                rate=self.tau_dropout,
                deterministic=not training,
                name="tau_dropout",
            )(
                tau_pre_dropout,
            )
        else:
            tau_pre_dropout = tau_override
            tau = tau_pre_dropout

        tau_pre_dropout = jnp.nan_to_num(
            tau_pre_dropout,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )
        tau = jnp.nan_to_num(
            tau,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )
        fields = TransformationDecoder(
            image_size=self.image_size,
            channels=self.channels,
            base_width=self.decoder_base_width,
            widths=self.decoder_widths,
            use_appearance=(
                self.use_appearance
                and self.channels > 1
            ),
            architecture=architecture,
            name="decoder",
        )(
            tau,
        )
        fields = jnp.nan_to_num(
            fields,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )
        spatial_images, sample_grid = apply_spatial_transform(
            images=images,
            spatial_params=fields[..., :6],
            spatial_scale=self.spatial_scale,
            smoothing_kernel=self.smoothing_kernel,
            parameterization=self.parameterization,
        )
        aux = {
            "tau": tau,
            "tau_pre_dropout": tau_pre_dropout,
            "fields": fields,
            "sample_grid": sample_grid,
            "spatial_images": spatial_images,
            "spatial_oob_fraction": jnp.mean(
                jnp.any(
                    jnp.abs(sample_grid) > 1.0,
                    axis=-1,
                ).astype(jnp.float32)
            ),
            "spatial_l1": jnp.mean(
                jnp.abs(spatial_images - images)
            ),
        }

        if self.use_appearance and self.channels > 1:
            appearance_input = (
                spatial_images
                if self.composition == "serial"
                else images
            )
            appearance_images, appearance_delta = apply_appearance_transform(
                images=appearance_input,
                appearance_params=fields[..., 6:],
                appearance_scale=self.appearance_scale,
                smoothing_kernel=self.smoothing_kernel,
                parameterization=self.parameterization,
            )
            augmented = combine_transforms(
                images=images,
                spatial_images=spatial_images,
                appearance_images=appearance_images,
                composition=self.composition,
                clip_output=self.parameterization == "guarded",
            )
            aux["appearance_delta"] = appearance_delta
            aux["appearance_images"] = appearance_images
            aux["appearance_out_of_range_fraction"] = jnp.mean(
                (
                    (appearance_images < 0.0)
                    | (appearance_images > 1.0)
                ).astype(jnp.float32)
            )
        else:
            augmented = spatial_images

        aux["augmented_out_of_range_fraction"] = jnp.mean(
            (
                (augmented < 0.0)
                | (augmented > 1.0)
            ).astype(jnp.float32)
        )
        aux["augmented_l1"] = jnp.mean(
            jnp.abs(augmented - images)
        )

        if return_aux:
            return augmented, aux

        return augmented


class ImageDiscriminator(nn.Module):
    """Image-space discriminator used during adversarial pretraining."""

    widths: tuple[int, ...] = (16, 32, 64, 128)
    architecture: str = "custom"
    image_size: int = 32

    @nn.compact
    def __call__(
        self,
        images: jnp.ndarray,
    ) -> jnp.ndarray:
        """Return one real/fake logit for each image."""
        features = images
        architecture = resolve_architecture(
            self.architecture,
            self.image_size,
        )
        widths, kernel_sizes, strides = _encoder_spec(
            architecture,
            self.widths,
        )

        for index, (
            width,
            kernel_size,
            stride,
        ) in enumerate(
            zip(
                widths,
                kernel_sizes,
                strides,
                strict=True,
            )
        ):
            features = nn.Conv(
                features=width,
                kernel_size=(kernel_size, kernel_size),
                strides=(stride, stride),
                padding="SAME",
                name=f"conv_{index}",
            )(
                features,
            )
            features = nn.leaky_relu(
                features,
                negative_slope=0.2,
            )

        features = features.reshape(
            (features.shape[0], -1),
        )

        return nn.Dense(
            1,
            name="logit",
        )(
            features,
        ).squeeze(
            -1,
        )


class FeatureDiscriminator(nn.Module):
    """Feature-space discriminator over frozen classifier representations."""

    @nn.compact
    def __call__(
        self,
        features: jnp.ndarray,
    ) -> jnp.ndarray:
        """Return one real/fake logit for each feature vector."""
        return nn.Dense(
            1,
            name="logit",
        )(
            features,
        ).squeeze(
            -1,
        )
