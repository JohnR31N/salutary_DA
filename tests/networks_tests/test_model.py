from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest
from flax.core import freeze, unfreeze
from flax.traverse_util import flatten_dict

from allthemix.networks.backbones.preact_resnet import PreActBasicBlock
from allthemix.networks.backbones.pyramidnet import PyramidNetBackbone
from allthemix.networks.backbones.wide_resnet import WideResNetBackbone
from allthemix.networks.builder import build_model
from allthemix.networks.classifiers.image_classifier import ImageClassifier
from allthemix.networks.heads.linear_head import LinearHead


class TestModel:
    def test_preact_basic_block_identity_shortcut_keeps_raw_input(self) -> None:
        """Verify identity PreAct blocks use the raw residual branch."""
        rng = jax.random.PRNGKey(5)
        block = PreActBasicBlock(
            features=3,
            stride=1,
        )
        images = jnp.linspace(
            -1.0,
            1.0,
            2 * 8 * 8 * 3,
            dtype=jnp.float32,
        ).reshape(
            2,
            8,
            8,
            3,
        )

        variables = block.init(
            rng,
            images,
            training=True,
        )
        mutable_variables = unfreeze(
            variables,
        )

        for module_params in mutable_variables["params"].values():
            if "kernel" in module_params:
                module_params["kernel"] = jnp.zeros_like(
                    module_params["kernel"],
                )

        variables = freeze(
            mutable_variables,
        )
        outputs = block.apply(
            variables,
            images,
            training=False,
        )

        assert jnp.max(
            jnp.abs(
                outputs - images,
            )
        ) < 1e-6

    def test_simple_cnn_output_shape(self) -> None:
        """Verify that simple cnn output shape."""
        rng = jax.random.PRNGKey(0)

        model = build_model(
            name="simple_cnn",
            num_classes=10,
        )

        images = jnp.ones((4, 32, 32, 3))

        variables = model.init(
            rng,
            images,
            training=True,
        )

        logits = model.apply(
            variables,
            images,
            training=True,
        )

        assert logits.shape == (4, 10)

    def test_unknown_model_raises_error(self) -> None:
        """Verify that unknown model raises error."""
        with pytest.raises(ValueError):
            build_model(
                name="unknown",
                num_classes=10,
            )

    def test_resnet50_imagenet_stem_output_shape(self) -> None:
        """Verify that ResNet-50 supports an ImageNet-style stem."""
        rng = jax.random.PRNGKey(2)

        model = build_model(
            name="resnet50",
            num_classes=200,
            resnet_stem_type="imagenet",
        )

        images = jnp.ones(
            (
                2,
                64,
                64,
                3,
            )
        )

        variables = model.init(
            rng,
            images,
            training=True,
        )

        logits = model.apply(
            variables,
            images,
            training=False,
        )

        assert logits.shape == (
            2,
            200,
        )

    def test_preact_resnet18_fmix_repo_stem_output_shape(self) -> None:
        """Verify PreActResNet-18 supports the FMix repo stem variant."""
        rng = jax.random.PRNGKey(3)

        model = build_model(
            name="preact_resnet18",
            num_classes=200,
            preact_stem_bn_relu=True,
        )

        images = jnp.ones(
            (
                2,
                64,
                64,
                3,
            )
        )

        variables = model.init(
            rng,
            images,
            training=True,
        )

        logits = model.apply(
            variables,
            images,
            training=False,
        )

        assert logits.shape == (
            2,
            200,
        )

    def test_preact_resnet18_pytorch_default_init_bounds(self) -> None:
        """Verify PreActResNet-18 can use PyTorch module defaults."""
        rng = jax.random.PRNGKey(4)

        model = build_model(
            name="preact_resnet18",
            num_classes=200,
            preact_stem_bn_relu=True,
            preact_pytorch_default_init=True,
        )

        images = jnp.ones(
            (
                2,
                64,
                64,
                3,
            )
        )

        variables = model.init(
            rng,
            images,
            training=True,
        )

        flat_params = flatten_dict(
            variables["params"],
        )
        stem_kernel = flat_params[
            (
                "backbone",
                "Conv_0",
                "kernel",
            )
        ]
        head_bias = flat_params[
            (
                "head",
                "Dense_0",
                "bias",
            )
        ]

        stem_bound = 1.0 / math.sqrt(3 * 3 * 3)

        assert float(jnp.max(jnp.abs(stem_kernel))) <= stem_bound
        assert float(jnp.max(jnp.abs(head_bias))) > 0.0

    def test_catchup_compatible_backbones_keep_normal_forward_path(self) -> None:
        """Verify that catchup compatible backbones keep normal forward path."""
        rng = jax.random.PRNGKey(1)
        images = jnp.ones((2, 32, 32, 3))

        backbones = [
            WideResNetBackbone(
                depth=16,
                widen_factor=1,
                dropout_rate=0.0,
            ),
            PyramidNetBackbone(
                depth=20,
                alpha=48,
                initial_channels=8,
            ),
        ]

        for index, backbone in enumerate(backbones):
            model = ImageClassifier(
                backbone=backbone,
                head=LinearHead(
                    num_classes=10,
                ),
            )

            init_rng = jax.random.fold_in(
                rng,
                index,
            )

            variables = model.init(
                init_rng,
                images,
                training=True,
            )

            logits = model.apply(
                variables,
                images,
                training=False,
            )

            assert logits.shape == (2, 10)
