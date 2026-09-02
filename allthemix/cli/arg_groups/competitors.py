"""Competitor-reproduction flags (ifaugnet/metaaugment/generative)."""

from __future__ import annotations

import argparse

from allthemix.utils.cli import str2bool


def add_competitors_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    """Register competitor-reproduction flags (ifaugnet/metaaugment/generative) flags."""
    parser.add_argument("--diffusemix_manifest", type=str, default="")
    parser.add_argument(
        "--diffusemix_train_mode",
        type=str,
        default="replace",
        choices=(
            "replace",
            "append",
        ),
    )
    parser.add_argument("--alia_manifest", type=str, default="")
    parser.add_argument(
        "--alia_train_mode",
        type=str,
        default="append",
        choices=(
            "replace",
            "append",
        ),
    )
    parser.add_argument("--saspa_manifest", type=str, default="")
    parser.add_argument(
        "--saspa_replacement_probability",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--metaaugment_policy_learning_rate",
        type=float,
        default=1.0e-3,
    )
    parser.add_argument(
        "--metaaugment_policy_momentum",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--metaaugment_policy_weight_decay",
        type=float,
        default=5.0e-4,
    )
    parser.add_argument(
        "--metaaugment_policy_nesterov",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--metaaugment_inner_learning_rate",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--metaaugment_learn_inner_learning_rate",
        type=str2bool,
        default=True,
    )
    parser.add_argument("--metaaugment_epsilon", type=float, default=0.1)
    parser.add_argument(
        "--metaaugment_num_transforms_per_sample",
        type=int,
        default=1,
    )
    parser.add_argument("--metaaugment_cutout_size", type=int, default=16)
    parser.add_argument(
        "--metaaugment_sampler_update_epochs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--metaaugment_sampler_history_epochs",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--metaaugment_translate_const",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--ifaugnet_stage",
        type=str,
        default="all",
        choices=(
            "all",
            "classifier",
            "pretrain",
            "influence",
            "retrain",
        ),
    )
    parser.add_argument(
        "--ifaugnet_retrain_policy_source",
        type=str,
        default="influence",
        choices=(
            "influence",
            "pretrain",
        ),
    )
    parser.add_argument(
        "--ifaugnet_policy_classifier_checkpoint",
        type=str,
        default="final",
        choices=(
            "final",
            "best",
            "early",
        ),
    )
    parser.add_argument(
        "--ifaugnet_policy_classifier_save_epoch",
        type=int,
        default=-1,
    )
    parser.add_argument("--ifaugnet_pretrain_steps", type=int, default=0)
    parser.add_argument("--ifaugnet_influence_steps", type=int, default=10_000)
    parser.add_argument("--ifaugnet_retrain_epochs", type=int, default=-1)
    parser.add_argument(
        "--ifaugnet_retrain_learning_rate",
        type=float,
        default=-1.0,
    )
    parser.add_argument("--ifaugnet_log_every", type=int, default=100)
    parser.add_argument("--ifaugnet_health_check_every", type=int, default=10)
    parser.add_argument(
        "--ifaugnet_pretrain_learning_rate",
        type=float,
        default=2.0e-4,
    )
    parser.add_argument("--ifaugnet_pretrain_beta1", type=float, default=0.5)
    parser.add_argument("--ifaugnet_pretrain_beta2", type=float, default=0.999)
    parser.add_argument("--ifaugnet_learning_rate", type=float, default=0.01)
    parser.add_argument(
        "--ifaugnet_lr_schedule",
        type=str,
        default="constant",
        choices=(
            "constant",
            "cosine",
            "warmup_cosine",
        ),
    )
    parser.add_argument(
        "--ifaugnet_min_learning_rate",
        type=float,
        default=0.0,
    )
    parser.add_argument("--ifaugnet_warmup_steps", type=int, default=0)
    parser.add_argument("--ifaugnet_beta1", type=float, default=0.9)
    parser.add_argument("--ifaugnet_beta2", type=float, default=0.99)
    parser.add_argument("--ifaugnet_tau_dim", type=int, default=128)
    parser.add_argument("--ifaugnet_tau_dropout", type=float, default=0.5)
    parser.add_argument(
        "--ifaugnet_pretrain_tau_dropout_match",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--ifaugnet_transform_parameterization",
        type=str,
        default="guarded",
        choices=(
            "guarded",
            "paper",
        ),
    )
    parser.add_argument(
        "--ifaugnet_composition",
        type=str,
        default="serial",
        choices=(
            "serial",
            "parallel",
        ),
    )
    parser.add_argument(
        "--ifaugnet_architecture",
        type=str,
        default="auto",
        choices=(
            "auto",
            "custom",
            "cifar",
            "imagenet",
        ),
    )
    parser.add_argument("--ifaugnet_spatial_scale", type=float, default=0.20)
    parser.add_argument(
        "--ifaugnet_appearance_scale",
        type=float,
        default=0.25,
    )
    parser.add_argument("--ifaugnet_smoothing_kernel", type=int, default=4)
    parser.add_argument(
        "--ifaugnet_use_appearance",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--ifaugnet_encoder_widths",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128],
    )
    parser.add_argument(
        "--ifaugnet_decoder_widths",
        type=int,
        nargs="+",
        default=[64, 32, 16],
    )
    parser.add_argument(
        "--ifaugnet_decoder_base_width",
        type=int,
        default=128,
    )
    parser.add_argument("--ifaugnet_damping", type=float, default=0.01)
    parser.add_argument("--ifaugnet_cg_iters", type=int, default=50)
    parser.add_argument("--ifaugnet_s_test_batches", type=int, default=16)
    parser.add_argument(
        "--ifaugnet_image_loss_weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--ifaugnet_feature_loss_weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--ifaugnet_pretrain_identity_l2_weight",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--ifaugnet_identity_l2_weight",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--ifaugnet_label_preservation_weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--ifaugnet_influence_clip_value",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--ifaugnet_learned_aug_probability",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--ifaugnet_gradient_clip_norm",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--ifaugnet_zero_nonfinite_grads",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--ifaugnet_restore_last_healthy_pretrain",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--ifaugnet_max_pretrain_generator_loss",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--ifaugnet_max_pretrain_identity_l2",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--ifaugnet_restore_best_healthy",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--ifaugnet_min_accuracy_retention",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--ifaugnet_max_tau_saturation_fraction",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--ifaugnet_collapse_patience",
        type=int,
        default=3,
    )
