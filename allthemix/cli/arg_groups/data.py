"""Data pipeline, splits, and augmentation recipe."""

from __future__ import annotations

import argparse

from allthemix.utils.cli import str2bool


def add_data_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    """Register data pipeline, splits, and augmentation recipe flags."""
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--shuffle_buffer_size", type=int, default=10_000)
    parser.add_argument("--validation_split", type=float, default=0.0)
    parser.add_argument(
        "--val_source",
        type=str,
        default=None,
        choices=("train", "test"),
        help=(
            "Where the validation partition comes from. If omitted, "
            "validation_split > 0 selects test (the formal protocol), "
            "while validation_split = 0 selects train. train (legacy): "
            "carve validation_split of the TRAINING split. test: use a "
            "deterministic class-stratified validation_split fraction of "
            "the OFFICIAL evaluation split (test, or official val where "
            "one exists); the complement stays sealed for final test and "
            "the training split is used in full."
        ),
    )
    parser.add_argument("--train_subset_fraction", type=float, default=1.0)
    parser.add_argument(
        "--val_select_split_fraction",
        type=float,
        default=0.0,
        help=(
            "Matched-control checkpoint selection: carve the validation "
            "partition with a deterministic seed and permutation, discard "
            "the first fraction, and evaluate/"
            "select checkpoints on the select remainder only. Gives "
            "baselines the identical selection-set size as val-guided "
            "method arms. 0 disables."
        ),
    )
    parser.add_argument("--basic_aug", type=str2bool, default=False)
    parser.add_argument(
        "--aug_recipe",
        type=str,
        default="",
        choices=(
            "",
            "none",
            "basic",
            "hflip",
            "horizontal_flip",
            "cub",
            "fine_grained",
            "imagenet",
            "tiny_official",
            "tiny_openmixup",
        ),
    )
    parser.add_argument(
        "--tiny_imagenet_normalization",
        type=str,
        default="imagenet",
        choices=("imagenet", "none"),
    )
    parser.add_argument(
        "--data_seed",
        type=int,
        default=-1,
        help="Data-order/augmentation seed; -1 reuses --seed.",
    )
    parser.add_argument("--sal_basic_aug", type=str2bool, default=False)
    parser.add_argument(
        "--sal_aug_recipe",
        type=str,
        default="",
        choices=(
            "",
            "none",
            "basic",
            "hflip",
            "horizontal_flip",
            "cub",
            "fine_grained",
            "imagenet",
            "tiny_official",
            "tiny_openmixup",
        ),
    )
    parser.add_argument("--saliency_dir", type=str, default="./data")
