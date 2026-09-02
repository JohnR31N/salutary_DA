"""Post-parse validation for competitor and instantaneous-GA flags."""

from __future__ import annotations

import argparse
import math

from allthemix.methods.utils.validation import normalize_method_name
from salutary_da.protocol import (
    get_instantaneous_ga_dataset_protocol,
    resolve_registered_timing_epochs,
)


def validate_ifaugnet_args(
    args: argparse.Namespace,
) -> None:
    """Validate options that are meaningful only for integrated IF-AugNet."""
    if args.validation_split <= 0.0 or args.eval_on_test_each_epoch:
        raise ValueError(
            "IF-AugNet requires validation_split > 0 and "
            "eval_on_test_each_epoch=false because influence training "
            "consumes the held-out validation partition."
        )
    if args.distributed and not args.sync_batch_stats:
        raise ValueError(
            "Distributed IF-AugNet requires sync_batch_stats=true so its "
            "classifier stages match global-batch BatchNorm semantics."
        )
    if args.pretrained_checkpoint:
        raise ValueError(
            "IF-AugNet stage dependencies use resume_checkpoint as a "
            "stage-root directory; pretrained_checkpoint is unsupported."
        )

    positive_values = {
        "ifaugnet_influence_steps": args.ifaugnet_influence_steps,
        "ifaugnet_log_every": args.ifaugnet_log_every,
        "ifaugnet_health_check_every": args.ifaugnet_health_check_every,
        "ifaugnet_pretrain_learning_rate": (args.ifaugnet_pretrain_learning_rate),
        "ifaugnet_learning_rate": args.ifaugnet_learning_rate,
        "ifaugnet_tau_dim": args.ifaugnet_tau_dim,
        "ifaugnet_smoothing_kernel": args.ifaugnet_smoothing_kernel,
        "ifaugnet_decoder_base_width": args.ifaugnet_decoder_base_width,
        "ifaugnet_damping": args.ifaugnet_damping,
        "ifaugnet_cg_iters": args.ifaugnet_cg_iters,
        "ifaugnet_s_test_batches": args.ifaugnet_s_test_batches,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"{name} must be > 0.")

    if args.ifaugnet_pretrain_steps < 0:
        raise ValueError("ifaugnet_pretrain_steps must be >= 0.")
    if args.ifaugnet_policy_classifier_save_epoch == 0:
        raise ValueError("ifaugnet_policy_classifier_save_epoch must be -1 or > 0.")
    if args.ifaugnet_policy_classifier_save_epoch < -1:
        raise ValueError("ifaugnet_policy_classifier_save_epoch must be -1 or > 0.")
    if (
        args.ifaugnet_policy_classifier_save_epoch > args.epochs
        and args.ifaugnet_stage in {"all", "classifier"}
    ):
        raise ValueError(
            "ifaugnet_policy_classifier_save_epoch must not exceed epochs."
        )
    if (
        args.ifaugnet_policy_classifier_checkpoint == "early"
        and args.ifaugnet_stage in {"all", "classifier"}
    ):
        if args.ifaugnet_policy_classifier_save_epoch <= 0:
            raise ValueError(
                "An early policy classifier requires "
                "ifaugnet_policy_classifier_save_epoch > 0 during the "
                "classifier stage."
            )
        if not args.save_checkpoint:
            raise ValueError(
                "An early policy classifier requires save_checkpoint=true."
            )
    if args.ifaugnet_retrain_policy_source == "pretrain":
        if args.ifaugnet_stage != "retrain":
            raise ValueError(
                "ifaugnet_retrain_policy_source=pretrain requires "
                "ifaugnet_stage=retrain."
            )
        if args.ifaugnet_pretrain_steps == 0:
            raise ValueError(
                "ifaugnet_retrain_policy_source=pretrain requires "
                "ifaugnet_pretrain_steps > 0."
            )
    if args.ifaugnet_warmup_steps < 0:
        raise ValueError("ifaugnet_warmup_steps must be >= 0.")
    if args.ifaugnet_min_learning_rate < 0.0:
        raise ValueError("ifaugnet_min_learning_rate must be >= 0.")
    if args.ifaugnet_min_learning_rate > args.ifaugnet_learning_rate:
        raise ValueError(
            "ifaugnet_min_learning_rate must not exceed ifaugnet_learning_rate."
        )
    if args.ifaugnet_lr_schedule == "warmup_cosine" and not (
        0 < args.ifaugnet_warmup_steps < args.ifaugnet_influence_steps
    ):
        raise ValueError(
            "warmup_cosine requires ifaugnet_warmup_steps to be in "
            "(0, ifaugnet_influence_steps)."
        )
    if args.ifaugnet_collapse_patience <= 0:
        raise ValueError("ifaugnet_collapse_patience must be > 0.")

    if args.ifaugnet_retrain_epochs == 0 or args.ifaugnet_retrain_epochs < -1:
        raise ValueError("ifaugnet_retrain_epochs must be -1 or > 0.")
    if (
        args.ifaugnet_retrain_learning_rate != -1.0
        and args.ifaugnet_retrain_learning_rate <= 0.0
    ):
        raise ValueError("ifaugnet_retrain_learning_rate must be -1 or > 0.")

    beta_values = {
        "ifaugnet_pretrain_beta1": args.ifaugnet_pretrain_beta1,
        "ifaugnet_pretrain_beta2": args.ifaugnet_pretrain_beta2,
        "ifaugnet_beta1": args.ifaugnet_beta1,
        "ifaugnet_beta2": args.ifaugnet_beta2,
    }
    for name, value in beta_values.items():
        if value < 0.0 or value >= 1.0:
            raise ValueError(f"{name} must be in [0, 1).")

    probability_values = {
        "ifaugnet_tau_dropout": args.ifaugnet_tau_dropout,
        "ifaugnet_learned_aug_probability": (args.ifaugnet_learned_aug_probability),
        "ifaugnet_min_accuracy_retention": (args.ifaugnet_min_accuracy_retention),
        "ifaugnet_max_tau_saturation_fraction": (
            args.ifaugnet_max_tau_saturation_fraction
        ),
    }
    for name, value in probability_values.items():
        if value < 0.0 or value > 1.0:
            raise ValueError(f"{name} must be in [0, 1].")

    nonnegative_values = {
        "ifaugnet_spatial_scale": args.ifaugnet_spatial_scale,
        "ifaugnet_appearance_scale": args.ifaugnet_appearance_scale,
        "ifaugnet_image_loss_weight": args.ifaugnet_image_loss_weight,
        "ifaugnet_feature_loss_weight": args.ifaugnet_feature_loss_weight,
        "ifaugnet_pretrain_identity_l2_weight": (
            args.ifaugnet_pretrain_identity_l2_weight
        ),
        "ifaugnet_identity_l2_weight": args.ifaugnet_identity_l2_weight,
        "ifaugnet_label_preservation_weight": (args.ifaugnet_label_preservation_weight),
        "ifaugnet_influence_clip_value": args.ifaugnet_influence_clip_value,
        "ifaugnet_gradient_clip_norm": args.ifaugnet_gradient_clip_norm,
        "ifaugnet_max_pretrain_generator_loss": (
            args.ifaugnet_max_pretrain_generator_loss
        ),
        "ifaugnet_max_pretrain_identity_l2": (args.ifaugnet_max_pretrain_identity_l2),
    }
    for name, value in nonnegative_values.items():
        if value < 0.0:
            raise ValueError(f"{name} must be >= 0.")

    width_values = {
        "ifaugnet_encoder_widths": args.ifaugnet_encoder_widths,
        "ifaugnet_decoder_widths": args.ifaugnet_decoder_widths,
    }
    for name, values in width_values.items():
        if not values or any(value <= 0 for value in values):
            raise ValueError(f"{name} must contain positive integers.")

    if args.ifaugnet_architecture == "imagenet" and args.dataset not in {
        "caltech_birds2011",
        "cars196",
        "imagenet100",
    }:
        raise ValueError(
            "ifaugnet_architecture=imagenet is reserved for 224x224 datasets."
        )


def validate_salda_ga_args(args: argparse.Namespace) -> None:
    """Fail closed on a registered instantaneous-GA dataset protocol."""

    if args.salda_ga_mode == "off":
        return
    protocol = get_instantaneous_ga_dataset_protocol(args.dataset)
    protocol_name = f"{protocol.dataset} SalDA"
    if args.seed not in {0, 1, 2}:
        raise ValueError(f"{protocol_name} seed must be exactly one of 0 1 2")
    if args.data_seed != -1:
        raise ValueError(
            f"{protocol_name} data_seed must be exactly -1 so it resolves "
            "from the registered training seed"
        )
    method_name = normalize_method_name(args.method)
    if method_name not in {"baseline", "mixup"}:
        raise ValueError(f"{protocol_name} method must be exactly baseline or mixup")
    if protocol.dataset == "stl10" and (
        args.resume_checkpoint or args.pretrained_checkpoint
    ):
        raise ValueError(
            "stl10 SalDA requires a fresh initialization; resume_checkpoint "
            "and pretrained_checkpoint must both be empty"
        )
    locked_values = {
        "dataset": (args.dataset, protocol.dataset),
        "model": (args.model, "preact_resnet18"),
        "resnet_stem_type": (args.resnet_stem_type, "cifar"),
        "preact_stem_bn_relu": (args.preact_stem_bn_relu, False),
        "preact_pytorch_default_init": (
            args.preact_pytorch_default_init,
            False,
        ),
        "batch_size": (args.batch_size, 128),
        "epochs": (args.epochs, 200),
        "learning_rate": (args.learning_rate, 0.1),
        "momentum": (args.momentum, 0.9),
        "nesterov": (args.nesterov, False),
        "weight_decay": (args.weight_decay, protocol.weight_decay),
        "lr_schedule": (args.lr_schedule, "cosine"),
        "min_learning_rate": (args.min_learning_rate, 0.0),
        "warmup_epochs": (args.warmup_epochs, 0),
        "lr_decay_epochs": (args.lr_decay_epochs, [100, 150]),
        "lr_decay_rate": (args.lr_decay_rate, 0.1),
        "mixup_alpha": (args.mixup_alpha, protocol.mixup_alpha),
        "shuffle_buffer_size": (args.shuffle_buffer_size, 10_000),
        "val_source": (args.val_source, "test"),
        "validation_split": (
            args.validation_split,
            protocol.validation_split,
        ),
        "max_eval_steps": (args.max_eval_steps, -1),
        "train_subset_fraction": (args.train_subset_fraction, 1.0),
        "val_select_split_fraction": (args.val_select_split_fraction, 0.0),
        "debug_train_source": (args.debug_train_source, "none"),
        "eval_on_test_each_epoch": (args.eval_on_test_each_epoch, False),
        "distributed": (args.distributed, True),
        "sync_batch_stats": (args.sync_batch_stats, True),
        "cross_device_shuffle": (args.cross_device_shuffle, False),
        "basic_aug": (args.basic_aug, True),
        "aug_recipe": (args.aug_recipe, "basic"),
        "deterministic_data": (args.deterministic_data, True),
        "strict_determinism": (args.strict_determinism, False),
        "early_stop_enabled": (args.early_stop_enabled, False),
        "save_csv": (args.save_csv, True),
        "log_time": (args.log_time, True),
        "wandb": (args.wandb, True),
        "wandb_mode": (args.wandb_mode, "online"),
    }
    mismatches = [
        f"{name}={actual!r} (required {expected!r})"
        for name, (actual, expected) in locked_values.items()
        if actual != expected
    ]
    if mismatches:
        raise ValueError(f"{protocol_name} protocol mismatch: " + ", ".join(mismatches))
    is_complete_training = args.salda_ga_stop_epoch == -1
    registered_short_epochs = resolve_registered_timing_epochs(
        stop_epoch=args.salda_ga_stop_epoch,
        protocol=protocol,
    )
    is_registered_stl10_endpoint = bool(
        protocol.dataset == "stl10" and registered_short_epochs == 30
    )
    if is_complete_training and not args.final_test:
        raise ValueError(
            f"{protocol_name} requires final_test=true for complete 200-epoch "
            "training"
        )
    if args.final_test:
        if not is_complete_training and not is_registered_stl10_endpoint:
            raise ValueError(
                f"{protocol_name} permits final_test=true only for complete "
                "training or the registered STL-10 30-epoch endpoint workload"
            )
        endpoint_values = {
            "max_train_steps": (args.max_train_steps, -1),
            "save_checkpoint": (args.save_checkpoint, True),
            "save_best_only": (args.save_best_only, True),
            "final_test_checkpoint": (args.final_test_checkpoint, "best"),
        }
        endpoint_mismatches = [
            f"{name}={actual!r} (required {expected!r})"
            for name, (actual, expected) in endpoint_values.items()
            if actual != expected
        ]
        if endpoint_mismatches:
            raise ValueError(
                f"{protocol_name} final-test endpoint mismatch: "
                + ", ".join(endpoint_mismatches)
            )
    if args.salda_ga_stop_epoch != -1 and not (
        1 <= args.salda_ga_stop_epoch <= args.epochs
    ):
        raise ValueError("salda_ga_stop_epoch must be -1 or within [1, epochs]")
    phase_epochs = {
        "salda_ga_score_start_epoch": args.salda_ga_score_start_epoch,
        "salda_ga_action_start_epoch": args.salda_ga_action_start_epoch,
    }
    invalid_phase_epochs = [
        name
        for name, value in phase_epochs.items()
        if isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= args.epochs
    ]
    if invalid_phase_epochs:
        raise ValueError(
            "invalid SalDA phase epochs: " + ", ".join(invalid_phase_epochs)
        )
    if args.salda_ga_score_start_epoch > args.salda_ga_action_start_epoch:
        raise ValueError(
            "salda_ga_score_start_epoch must not exceed "
            "salda_ga_action_start_epoch"
        )
    if args.salda_ga_score_stop_epoch != -1 and not (
        args.salda_ga_score_start_epoch
        < args.salda_ga_score_stop_epoch
        <= args.epochs
    ):
        raise ValueError(
            "salda_ga_score_stop_epoch must be -1 or greater than the score "
            "start and no greater than epochs"
        )
    if args.salda_ga_action_stop_epoch != -1 and not (
        args.salda_ga_action_start_epoch
        < args.salda_ga_action_stop_epoch
        <= args.epochs
    ):
        raise ValueError(
            "salda_ga_action_stop_epoch must be -1 or greater than the action "
            "start and no greater than epochs"
        )
    effective_run_stop_epoch = (
        args.epochs
        if args.salda_ga_stop_epoch == -1
        else args.salda_ga_stop_epoch
    )
    if (
        args.salda_ga_mode != "baseline"
        and args.salda_ga_score_start_epoch >= effective_run_stop_epoch
    ):
        raise ValueError(
            "salda_ga_score_start_epoch must precede the run stop epoch"
        )
    if (
        args.salda_ga_score_stop_epoch != -1
        and args.salda_ga_score_stop_epoch > effective_run_stop_epoch
    ):
        raise ValueError(
            "salda_ga_score_stop_epoch must not exceed the run stop epoch"
        )
    if (
        args.salda_ga_action_stop_epoch != -1
        and args.salda_ga_action_stop_epoch > effective_run_stop_epoch
    ):
        raise ValueError(
            "salda_ga_action_stop_epoch must not exceed the run stop epoch"
        )
    if (
        args.salda_ga_score_stop_epoch != -1
        and args.salda_ga_action_stop_epoch != -1
        and args.salda_ga_action_stop_epoch > args.salda_ga_score_stop_epoch
    ):
        raise ValueError(
            "salda_ga_action_stop_epoch must not exceed "
            "salda_ga_score_stop_epoch"
        )
    if (
        args.salda_ga_score_stop_epoch != -1
        and args.salda_ga_mode
        in {
            "soft_label",
            "reweight",
            "shuffled_soft_label",
            "shuffled_reweight",
        }
        and args.salda_ga_action_start_epoch >= args.salda_ga_score_stop_epoch
    ):
        raise ValueError(
            "salda_ga_action_start_epoch must precede "
            "salda_ga_score_stop_epoch"
        )
    if len(args.salda_ga_git_commit) != 40 or any(
        character not in "0123456789abcdef"
        for character in args.salda_ga_git_commit.lower()
    ):
        raise ValueError("salda_ga_git_commit must be an exact 40-character SHA")
    if args.salda_ga_parameter_scope not in {"classifier_head", "full"}:
        raise ValueError("salda_ga_parameter_scope must be classifier_head or full")
    if args.salda_ga_validation_direction_mode not in {
        "full",
        "batch_aggregate",
    }:
        raise ValueError(
            "salda_ga_validation_direction_mode must be full or batch_aggregate"
        )
    if args.salda_ga_validation_batch_size != protocol.validation_batch_size:
        raise ValueError(
            "salda_ga_validation_batch_size must be exactly "
            f"{protocol.validation_batch_size} for {protocol.dataset}"
        )
    if (
        args.salda_ga_validation_reanchor_interval
        != protocol.validation_reanchor_interval
    ):
        raise ValueError(
            "salda_ga_validation_reanchor_interval must be exactly "
            f"{protocol.validation_reanchor_interval} for {protocol.dataset}"
        )
    if (
        args.salda_ga_validation_direction_mode == "batch_aggregate"
        and not protocol.allow_batch_aggregate
    ):
        raise ValueError(
            f"batch-aggregate SalDA is not registered for {protocol.dataset}"
        )
    if (
        args.salda_ga_validation_direction_mode == "batch_aggregate"
        and args.resume_checkpoint
    ):
        raise ValueError("batch-aggregate SalDA does not support resume_checkpoint")
    if (
        args.salda_ga_validation_direction_mode == "batch_aggregate"
        and args.salda_ga_parameter_scope != "classifier_head"
    ):
        raise ValueError("batch-aggregate SalDA requires classifier_head scope")
    bounded_values = {
        "salda_ga_soft_label_dose": (args.salda_ga_soft_label_dose, 0.0, 1.0),
        "salda_ga_fallback_soft_label_dose": (
            args.salda_ga_fallback_soft_label_dose,
            0.0,
            1.0,
        ),
        "salda_ga_max_weight_deviation": (
            args.salda_ga_max_weight_deviation,
            0.0,
            1.0,
        ),
        "salda_ga_minimum_relative_ess": (
            args.salda_ga_minimum_relative_ess,
            0.0,
            1.0,
        ),
    }
    invalid_bounded = [
        name
        for name, (value, lower, upper) in bounded_values.items()
        if not math.isfinite(value) or not (lower < value <= upper)
    ]
    if invalid_bounded:
        raise ValueError(
            f"invalid {protocol_name} bounded values: " + ", ".join(invalid_bounded)
        )
    if args.salda_ga_max_weight_deviation >= 1.0:
        raise ValueError("salda_ga_max_weight_deviation must be below 1")
    if args.salda_ga_maximum_rows not in {1, 2, 4, 8, 16, 32, 64, 128}:
        raise ValueError("salda_ga_maximum_rows must be one of 1 2 4 8 16 32 64 128")
    if (
        not math.isfinite(args.salda_ga_weight_temperature)
        or args.salda_ga_weight_temperature <= 0.0
    ):
        raise ValueError("salda_ga_weight_temperature must be finite and positive")
    threshold_values = {
        "salda_ga_minimum_gain": args.salda_ga_minimum_gain,
        "salda_ga_minimum_label_margin": args.salda_ga_minimum_label_margin,
        "salda_ga_minimum_relative_label_margin": (
            args.salda_ga_minimum_relative_label_margin
        ),
    }
    invalid_thresholds = [
        name
        for name, value in threshold_values.items()
        if not math.isfinite(value) or value < 0.0
    ]
    if invalid_thresholds:
        raise ValueError(
            f"invalid {protocol_name} threshold values: "
            + ", ".join(invalid_thresholds)
        )
    continuous_modes = {
        "soft_label",
        "reweight",
        "shuffled_soft_label",
        "shuffled_reweight",
    }
    if args.salda_ga_mode in continuous_modes:
        if args.salda_ga_parameter_scope != "classifier_head":
            raise ValueError(
                "production continuous actions require classifier_head scope"
            )
        if args.salda_ga_mode in {"soft_label", "shuffled_soft_label"} and (
            args.salda_ga_soft_label_dose not in {0.01, 0.025, 0.05, 0.1}
        ):
            raise ValueError(
                "soft-label action dose must be one of 0.01 0.025 0.05 0.1"
            )
        if args.salda_ga_mode in {"reweight", "shuffled_reweight"} and (
            args.salda_ga_max_weight_deviation not in {0.05, 0.1, 0.2}
        ):
            raise ValueError("reweight deviation must be one of 0.05 0.10 0.20")
        if args.salda_ga_fallback_enabled and (
            args.salda_ga_fallback_soft_label_dose != 0.01
        ):
            raise ValueError("the registered fallback soft-label dose is exactly 0.01")
    if args.salda_ga_fallback_enabled and args.salda_ga_mode not in {
        "soft_label",
        "shuffled_soft_label",
    }:
        raise ValueError("salda_ga_fallback_enabled requires a soft-label action mode")
