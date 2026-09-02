"""Training argument surface: assembly and post-processing.

Flag definitions live in ``allthemix.cli.arg_groups`` (one module per
domain); validation lives in ``allthemix.cli.arg_validation``. This
module assembles the parser and derives/validates the final namespace.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from allthemix.cli.arg_groups.competitors import add_competitors_arguments
from allthemix.cli.arg_groups.core import add_core_arguments
from allthemix.cli.arg_groups.data import add_data_arguments
from allthemix.cli.arg_groups.logging import add_logging_arguments
from allthemix.cli.arg_groups.methods import add_methods_arguments
from allthemix.cli.arg_groups.runtime import add_runtime_arguments
from allthemix.cli.arg_groups.salutary import add_salutary_arguments
from allthemix.cli.arg_groups.train import add_train_arguments
from allthemix.cli.arg_validation import (
    validate_ifaugnet_args,
    validate_salda_ga_args,
)
from allthemix.config import (
    load_yaml_config,
    validate_config_keys,
)
from allthemix.methods.utils.validation import normalize_method_name


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()

    add_core_arguments(parser)
    add_data_arguments(parser)
    add_methods_arguments(parser)
    add_train_arguments(parser)
    add_runtime_arguments(parser)
    add_logging_arguments(parser)
    add_competitors_arguments(parser)
    add_salutary_arguments(parser)

    prelim_args, _ = parser.parse_known_args()
    valid_keys = {action.dest for action in parser._actions}
    config: dict[str, Any] = {}

    if prelim_args.config:
        config_path = Path(prelim_args.config)
        config = load_yaml_config(config_path)
        validate_config_keys(
            config=config,
            valid_keys=valid_keys,
            config_path=config_path,
        )

    if config:
        parser.set_defaults(**config)

    args = parser.parse_args()

    if args.val_source is None:
        args.val_source = (
            "test"
            if args.validation_split > 0.0
            else "train"
        )

    if not isinstance(
        args.diffusemix_train_mode,
        str,
    ):
        raise TypeError(
            "diffusemix_train_mode must be 'replace' or 'append'. "
            f"Got {args.diffusemix_train_mode!r}."
        )
    args.diffusemix_train_mode = args.diffusemix_train_mode.lower()
    if args.diffusemix_train_mode not in {
        "replace",
        "append",
    }:
        raise ValueError(
            "diffusemix_train_mode must be 'replace' or 'append'. "
            f"Got {args.diffusemix_train_mode!r}."
        )

    if not isinstance(
        args.alia_train_mode,
        str,
    ):
        raise TypeError(
            "alia_train_mode must be 'replace' or 'append'. "
            f"Got {args.alia_train_mode!r}."
        )
    args.alia_train_mode = args.alia_train_mode.lower()
    if args.alia_train_mode not in {
        "replace",
        "append",
    }:
        raise ValueError(
            "alia_train_mode must be 'replace' or 'append'. "
            f"Got {args.alia_train_mode!r}."
        )

    if not args.aug_recipe:
        args.aug_recipe = "basic" if args.basic_aug else "none"

    if not args.sal_aug_recipe:
        args.sal_aug_recipe = "basic" if args.sal_basic_aug else "none"

    if args.validation_split < 0.0 or args.validation_split >= 1.0:
        raise ValueError(
            "validation_split must be in [0, 1). "
            f"Got {args.validation_split}."
        )

    if (
        args.train_subset_fraction <= 0.0
        or args.train_subset_fraction > 1.0
    ):
        raise ValueError(
            "train_subset_fraction must be in (0, 1]. "
            f"Got {args.train_subset_fraction}."
        )

    if args.early_stop_start_epoch < 0:
        raise ValueError(
            "early_stop_start_epoch must be >= 0. "
            f"Got {args.early_stop_start_epoch}."
        )

    if args.early_stop_patience < 1:
        raise ValueError(
            "early_stop_patience must be >= 1. "
            f"Got {args.early_stop_patience}."
        )

    if args.early_stop_min_delta < 0.0:
        raise ValueError(
            "early_stop_min_delta must be >= 0. "
            f"Got {args.early_stop_min_delta}."
        )

    if not args.eval_on_test_each_epoch and args.validation_split == 0.0:
        raise ValueError(
            "eval_on_test_each_epoch=false requires validation_split > 0."
        )

    if args.val_select_split_fraction != 0.0:
        if not 0.0 < args.val_select_split_fraction < 1.0:
            raise ValueError(
                "val_select_split_fraction must be in (0, 1) when set. "
                f"Got {args.val_select_split_fraction}."
            )

        if args.validation_split <= 0.0 or args.eval_on_test_each_epoch:
            raise ValueError(
                "val_select_split_fraction requires validation_split > 0 "
                "and eval_on_test_each_epoch=false (it subsets the "
                "held-out validation partition)."
            )

        if not args.deterministic_data:
            raise ValueError(
                "val_select_split_fraction requires "
                "deterministic_data=true so the select half aligns with "
                "the deterministic validation partition."
            )

    if args.debug_train_source != "none":
        if args.validation_split <= 0.0 or args.eval_on_test_each_epoch:
            raise ValueError(
                "debug_train_source requires validation_split > 0 and "
                "eval_on_test_each_epoch=false so the per-epoch eval "
                "measures the held-out validation partition it leaks."
            )

        if not args.deterministic_data:
            raise ValueError(
                "debug_train_source requires deterministic_data=true so "
                "train-side and eval-side split membership is identical."
            )

        if args.train_subset_fraction < 1.0:
            raise ValueError(
                "debug_train_source is not supported with "
                "train_subset_fraction < 1.0."
            )

        debug_source_method = normalize_method_name(args.method)
        debug_source_unsupported = {
            "ifaugnet",
            "metaaugment",
            "saliencymix",
            "guidedmixup",
            "diffusemix",
            "diffuse_mix",
            "alia",
            "saspa",
        }
        if debug_source_method in debug_source_unsupported:
            raise ValueError(
                "debug_train_source only supports methods on the "
                "standard pipeline path (baseline/mixup/cutmix/...). "
                f"Got method={args.method!r}."
            )

    if (
        args.final_test
        and args.final_test_checkpoint == "best"
        and (
            args.validation_split == 0.0
            or args.eval_on_test_each_epoch
        )
    ):
        raise ValueError(
            "final_test_checkpoint=best requires a held-out validation split: "
            "set validation_split > 0 and eval_on_test_each_epoch=false."
        )

    if args.resume_checkpoint and args.pretrained_checkpoint:
        raise ValueError(
            "Use either resume_checkpoint or pretrained_checkpoint, not both."
        )

    if args.seed < 0:
        raise ValueError("seed must be >= 0.")

    if args.data_seed < -1:
        raise ValueError("data_seed must be -1 or >= 0.")

    method_name = normalize_method_name(args.method)
    diffusemix_method_names = {
        "diffusemix",
        "diffuse_mix",
    }
    if method_name in diffusemix_method_names and not args.diffusemix_manifest:
        raise ValueError(
            "method=diffusemix requires diffusemix_manifest from the offline "
            "generation command."
        )
    if (
        method_name not in diffusemix_method_names
        and args.diffusemix_manifest
    ):
        raise ValueError(
            "diffusemix_manifest is only valid with method=diffusemix."
        )
    alia_method_names = {
        "alia",
    }
    if method_name in alia_method_names and not args.alia_manifest:
        raise ValueError(
            "method=alia requires alia_manifest from the offline ALIA "
            "filter command."
        )
    if method_name not in alia_method_names and args.alia_manifest:
        raise ValueError("alia_manifest is only valid with method=alia.")
    saspa_method_names = {
        "saspa",
    }
    if method_name in saspa_method_names and not args.saspa_manifest:
        raise ValueError(
            "method=saspa requires saspa_manifest from the offline SaSPA "
            "filter command."
        )
    if method_name not in saspa_method_names and args.saspa_manifest:
        raise ValueError("saspa_manifest is only valid with method=saspa.")
    if not 0.0 <= args.saspa_replacement_probability <= 1.0:
        raise ValueError(
            "saspa_replacement_probability must be in [0, 1]."
        )
    offline_manifests = [
        value
        for value in (
            args.diffusemix_manifest,
            args.alia_manifest,
            args.saspa_manifest,
        )
        if value
    ]
    if len(offline_manifests) > 1:
        raise ValueError(
            "Use only one offline training manifest per experiment."
        )

    if method_name == "metaaugment":
        if args.validation_split <= 0.0 or args.eval_on_test_each_epoch:
            raise ValueError(
                "MetaAugment requires validation_split > 0 and "
                "eval_on_test_each_epoch=false because its policy consumes "
                "the held-out validation partition."
            )
        if args.distributed and not args.sync_batch_stats:
            raise ValueError(
                "Distributed MetaAugment requires sync_batch_stats=true so "
                "its global batch matches single-device BatchNorm semantics."
            )

    if method_name == "ifaugnet":
        validate_ifaugnet_args(
            args,
        )

    validate_salda_ga_args(args)

    if args.val_source == "test":
        if args.validation_split <= 0.0:
            raise ValueError(
                "val_source=test requires validation_split > 0 (the "
                "fraction of the OFFICIAL eval split used as validation)."
            )

        if args.train_subset_fraction < 1.0:
            raise ValueError(
                "val_source=test does not support train_subset_fraction "
                "< 1 (the training split is used in full)."
            )

        if args.eval_on_test_each_epoch:
            raise ValueError(
                "val_source=test requires eval_on_test_each_epoch=false: "
                "per-epoch eval / checkpoint selection must run on the "
                "test-sourced VALIDATION half, never on the sealed test "
                "half."
            )
        if args.debug_train_source != "none":
            raise ValueError(
                "val_source=test is incompatible with debug_train_source: "
                "the debug mode intentionally injects validation examples "
                "into training."
            )

    if args.metaaugment_policy_learning_rate <= 0.0:
        raise ValueError("metaaugment_policy_learning_rate must be > 0.")
    if args.metaaugment_policy_momentum < 0.0:
        raise ValueError("metaaugment_policy_momentum must be >= 0.")
    if args.metaaugment_policy_weight_decay < 0.0:
        raise ValueError("metaaugment_policy_weight_decay must be >= 0.")
    if args.metaaugment_inner_learning_rate <= 0.0:
        raise ValueError("metaaugment_inner_learning_rate must be > 0.")
    if not 0.0 <= args.metaaugment_epsilon <= 1.0:
        raise ValueError("metaaugment_epsilon must be in [0, 1].")
    if args.metaaugment_num_transforms_per_sample < 1:
        raise ValueError("metaaugment_num_transforms_per_sample must be >= 1.")
    if args.metaaugment_cutout_size < 0:
        raise ValueError("metaaugment_cutout_size must be >= 0.")
    if args.metaaugment_sampler_update_epochs < 1:
        raise ValueError("metaaugment_sampler_update_epochs must be >= 1.")
    if args.metaaugment_sampler_history_epochs < 1:
        raise ValueError("metaaugment_sampler_history_epochs must be >= 1.")
    if args.metaaugment_translate_const < 0.0:
        raise ValueError("metaaugment_translate_const must be >= 0.")

    return args
