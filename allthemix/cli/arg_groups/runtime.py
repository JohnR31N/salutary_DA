"""Devices, distribution, and determinism."""

from __future__ import annotations

import argparse

from allthemix.utils.cli import str2bool


def add_runtime_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    """Register devices, distribution, and determinism flags."""
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--deterministic_data",
        type=str2bool,
        default=True,
        help="Use seeded shuffle and stateless train augmentation.",
    )
    parser.add_argument(
        "--strict_determinism",
        type=str2bool,
        default=False,
        help="Also enable TensorFlow deterministic kernels (slower).",
    )
    parser.add_argument("--distributed", type=str2bool, default=False)
    parser.add_argument("--cross_device_shuffle", type=str2bool, default=False)
    parser.add_argument("--sync_batch_stats", type=str2bool, default=False)
