from __future__ import annotations

import flax.linen as nn

from allthemix.networks.backbones.preact_resnet import preact_resnet18_backbone
from allthemix.networks.backbones.pyramidnet import pyramidnet200_backbone
from allthemix.networks.backbones.resnet import (
    resnet18_backbone,
    resnet34_backbone,
    resnet50_backbone,
    resnet101_backbone,
    resnet152_backbone,
)
from allthemix.networks.backbones.simple_cnn import SimpleCNNBackbone
from allthemix.networks.backbones.wide_resnet import wide_resnet28_10_backbone
from allthemix.networks.classifiers.image_classifier import ImageClassifier
from allthemix.networks.heads.linear_head import LinearHead

FEATURE_HOOK_COUNTS = {
    "simple_cnn": 2,
    "resnet18": 5,
    "resnet34": 5,
    "resnet50": 5,
    "resnet101": 5,
    "resnet152": 5,
    "preact_resnet18": 5,
    "wide_resnet28_10": 4,
    "pyramidnet200": 4,
}


def normalize_model_name(
    name: str,
) -> str:
    """Normalize user-facing model names into internal identifiers."""
    return name.lower().replace("-", "_").replace(" ", "_")


def get_feature_hook_count(
    name: str,
) -> int:
    """Return how many feature-hook layers a named backbone exposes."""
    model_name = normalize_model_name(
        name,
    )

    try:
        return FEATURE_HOOK_COUNTS[model_name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported model: {name}",
        ) from exc


def build_model(
    name: str,
    num_classes: int,
    resnet_stem_type: str = "cifar",
    preact_stem_bn_relu: bool = False,
    preact_pytorch_default_init: bool = False,
) -> nn.Module:
    """Build a named classifier model for the requested class count."""
    model_name = normalize_model_name(
        name,
    )

    if model_name == "simple_cnn":
        backbone = SimpleCNNBackbone(
            feature_dim=128,
        )

        head = LinearHead(
            num_classes=num_classes,
        )

        return ImageClassifier(
            backbone=backbone,
            head=head,
        )

    if model_name == "resnet18":
        backbone = resnet18_backbone(
            stem_type=resnet_stem_type,
        )
        head = LinearHead(num_classes=num_classes)
        return ImageClassifier(backbone=backbone, head=head)

    if model_name == "resnet34":
        backbone = resnet34_backbone(
            stem_type=resnet_stem_type,
        )
        head = LinearHead(num_classes=num_classes)
        return ImageClassifier(backbone=backbone, head=head)

    if model_name == "resnet50":
        backbone = resnet50_backbone(
            stem_type=resnet_stem_type,
        )
        head = LinearHead(num_classes=num_classes)
        return ImageClassifier(backbone=backbone, head=head)

    if model_name == "resnet101":
        backbone = resnet101_backbone(
            stem_type=resnet_stem_type,
        )
        head = LinearHead(num_classes=num_classes)
        return ImageClassifier(backbone=backbone, head=head)

    if model_name == "resnet152":
        backbone = resnet152_backbone(
            stem_type=resnet_stem_type,
        )
        head = LinearHead(num_classes=num_classes)
        return ImageClassifier(backbone=backbone, head=head)

    if model_name == "preact_resnet18":
        backbone = preact_resnet18_backbone(
            stem_type=resnet_stem_type,
            stem_bn_relu=preact_stem_bn_relu,
            pytorch_default_init=preact_pytorch_default_init,
        )

        head = LinearHead(
            num_classes=num_classes,
            init_style=(
                "pytorch_default"
                if preact_pytorch_default_init
                else "normal_0.01"
            ),
        )

        return ImageClassifier(
            backbone=backbone,
            head=head,
        )

    if model_name == "wide_resnet28_10":
        backbone = wide_resnet28_10_backbone(
            dropout_rate=0.3,
        )

        head = LinearHead(
            num_classes=num_classes,
        )

        return ImageClassifier(
            backbone=backbone,
            head=head,
        )

    if model_name == "pyramidnet200":
        backbone = pyramidnet200_backbone(
            alpha=240,
        )

        head = LinearHead(
            num_classes=num_classes,
        )

        return ImageClassifier(
            backbone=backbone,
            head=head,
        )

    raise ValueError(f"Unsupported model: {name}")
