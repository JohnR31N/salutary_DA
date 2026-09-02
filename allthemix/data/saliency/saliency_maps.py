"""Command-line entry point for saliency map preprocessing."""

from __future__ import annotations

import argparse
import logging

from allthemix.data.saliency.saliency_io import (
    build_train_saliency_maps,
    canonical_dataset_name,
    get_train_saliency_path,
    load_train_saliency_maps,
)
from allthemix.data.saliency.saliency_methods import (
    COLUMN_STRIPE_STD_RATIO,
    GRADIENT_METHOD_NAMES,
    OPENCV_METHOD_NAMES,
    SALIENCY_METHOD_CHOICES,
    SPECTRAL_RESIDUAL_METHOD_NAMES,
    SPECTRAL_RESIDUAL_NOFALLBACK_METHOD_NAMES,
    compute_gradient_saliency_map,
    compute_opencv_finegrained_saliency_map,
    compute_saliency_map,
    compute_spectral_residual_saliency_map,
    compute_spectral_residual_saliency_map_core,
    compute_spectral_residual_saliency_map_nofallback,
    is_saliency_map_suspicious,
    normalize_saliency_map,
)
from allthemix.utils.cli import str2bool


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./data")
    parser.add_argument(
        "--method",
        type=str,
        default="opencv",
        choices=list(SALIENCY_METHOD_CHOICES),
    )
    parser.add_argument("--overwrite", type=str2bool, default=False)

    args = parser.parse_args()

    build_train_saliency_maps(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        saliency_dir=args.output_dir,
        method=args.method,
        overwrite=args.overwrite,
    )


__all__ = [
    "COLUMN_STRIPE_STD_RATIO",
    "GRADIENT_METHOD_NAMES",
    "OPENCV_METHOD_NAMES",
    "SALIENCY_METHOD_CHOICES",
    "SPECTRAL_RESIDUAL_METHOD_NAMES",
    "SPECTRAL_RESIDUAL_NOFALLBACK_METHOD_NAMES",
    "build_train_saliency_maps",
    "canonical_dataset_name",
    "compute_gradient_saliency_map",
    "compute_opencv_finegrained_saliency_map",
    "compute_saliency_map",
    "compute_spectral_residual_saliency_map",
    "compute_spectral_residual_saliency_map_core",
    "compute_spectral_residual_saliency_map_nofallback",
    "get_train_saliency_path",
    "is_saliency_map_suspicious",
    "load_train_saliency_maps",
    "main",
    "normalize_saliency_map",
    "str2bool",
]


if __name__ == "__main__":
    main()
