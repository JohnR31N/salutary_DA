"""CLI adapter for CMC ImageNet-100 preparation/verification.

The preparation logic lives in allthemix.data.datasets.imagenet100_prepare;
this module only parses arguments and prints the summary.
"""

from __future__ import annotations

import argparse
import json

from allthemix.data.datasets.imagenet100_prepare import (
    DATASET_INFO_FILE_NAME,
    LABEL_MAP_FILE_NAME,
    MANIFEST_FILE_NAME,
    prepare_cmc_imagenet100_dataset,
    verify_cmc_imagenet100_dataset,
)


def _build_parser() -> argparse.ArgumentParser:
    """Build the CMC preparation command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Prepare or verify the fixed CMC ImageNet-100 subset from a "
            "licensed local ImageNet-1K class-folder copy."
        )
    )
    parser.add_argument(
        "--source_dir",
        default="",
        help="ImageNet-1K root containing train/<synset> and val/<synset>.",
    )
    parser.add_argument(
        "--output_dir",
        default="./data/imagenet100",
        help="Canonical CMC output root.",
    )
    parser.add_argument(
        "--link_mode",
        choices=(
            "symlink",
            "hardlink",
            "copy",
        ),
        default="symlink",
    )
    parser.add_argument(
        "--verify_only",
        action="store_true",
    )
    return parser


def main() -> None:
    """Prepare or verify CMC ImageNet-100 and print its exact summary."""
    args = _build_parser().parse_args()
    if args.verify_only:
        summary = verify_cmc_imagenet100_dataset(
            data_dir=args.output_dir,
        )
    else:
        if not args.source_dir:
            raise ValueError("--source_dir is required unless --verify_only is set.")
        summary = prepare_cmc_imagenet100_dataset(
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            link_mode=args.link_mode,
        )

    print("CMC ImageNet-100 validation passed.")
    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "DATASET_INFO_FILE_NAME",
    "LABEL_MAP_FILE_NAME",
    "MANIFEST_FILE_NAME",
    "prepare_cmc_imagenet100_dataset",
    "verify_cmc_imagenet100_dataset",
]
