"""Mixing-method hyperparameters."""

from __future__ import annotations

import argparse

from allthemix.utils.cli import str2bool


def add_methods_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    """Register mixing-method hyperparameters flags."""
    parser.add_argument("--mixup_alpha", type=float, default=1.0)
    parser.add_argument("--cutmix_alpha", type=float, default=1.0)
    parser.add_argument("--cutmix_prob", type=float, default=1.0)
    parser.add_argument("--cutmix_no_repeat", type=str2bool, default=False)
    parser.add_argument("--cutmix_per_sample_lam", type=str2bool, default=False)
    parser.add_argument("--cutmix_min_lam", type=float, default=0.0)
    parser.add_argument(
        "--cutmix_variant",
        type=str,
        default="standard",
        choices=(
            "standard",
            "area_adjusted",
            "torchbearer",
            "fmix_repo",
            "torchbearer_area",
            "fmix_repo_area",
            "torchbearer_inside",
        ),
    )
    parser.add_argument("--sumix_gamma", type=float, default=0.5)
    parser.add_argument("--sumix_semantic_scale", type=float, default=-1.0)
    parser.add_argument("--fmix_alpha", type=float, default=1.0)
    parser.add_argument("--fmix_decay", type=float, default=3.0)
    parser.add_argument("--fmix_prob", type=float, default=1.0)
    parser.add_argument("--fmix_per_sample", type=str2bool, default=False)
    parser.add_argument("--fmix_no_repeat", type=str2bool, default=False)
    parser.add_argument("--resizemix_scope_min", type=float, default=0.1)
    parser.add_argument("--resizemix_scope_max", type=float, default=0.8)
    parser.add_argument("--resizemix_prob", type=float, default=1.0)
    parser.add_argument("--resizemix_per_sample", type=str2bool, default=False)
    parser.add_argument("--guidedmixup_alpha", type=float, default=1.0)
    parser.add_argument("--guidedmixup_prob", type=float, default=1.0)
    parser.add_argument("--guidedmixup_blur_kernel", type=int, default=7)
    parser.add_argument(
        "--guidedmixup_condition",
        type=str,
        default="greedy",
        choices=("random", "greedy"),
    )
    parser.add_argument("--catchupmix_alpha", type=float, default=1.0)
    parser.add_argument("--catchupmix_cutmix_alpha", type=float, default=1.0)
    parser.add_argument("--catchupmix_num_layers", type=int, default=5)
    parser.add_argument("--catchupmix_no_repeat", type=str2bool, default=False)
    parser.add_argument("--saliencymix_alpha", type=float, default=1.0)
    parser.add_argument("--saliencymix_prob", type=float, default=1.0)
    parser.add_argument("--saliencymix_per_sample", type=str2bool, default=False)
