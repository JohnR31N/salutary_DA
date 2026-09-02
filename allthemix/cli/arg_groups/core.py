"""Config file, dataset/model/method selection, and model shape."""

from __future__ import annotations

import argparse

from allthemix.utils.cli import str2bool


def add_core_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    """Register config file, dataset/model/method selection, and model shape flags."""
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--model", type=str, default="simple_cnn")
    parser.add_argument("--method", type=str, default="baseline")
    parser.add_argument(
        "--resnet_stem_type",
        type=str,
        default="cifar",
        choices=("cifar", "imagenet"),
    )
    parser.add_argument("--preact_stem_bn_relu", type=str2bool, default=False)
    parser.add_argument(
        "--preact_pytorch_default_init",
        type=str2bool,
        default=False,
    )
