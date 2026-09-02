"""Instantaneous gradient-alignment training flags."""

from __future__ import annotations

import argparse

from allthemix.utils.cli import str2bool


def add_salutary_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    """Register the single supported SalDA runtime surface."""

    parser.add_argument(
        "--salda_ga_mode",
        type=str,
        default="off",
        choices=(
            "off",
            "baseline",
            "noop",
            "score_only",
            "soft_label",
            "reweight",
            "shuffled_soft_label",
            "shuffled_reweight",
        ),
    )
    parser.add_argument(
        "--salda_ga_parameter_scope",
        type=str,
        default="classifier_head",
        choices=("classifier_head", "full"),
    )
    parser.add_argument(
        "--salda_ga_validation_direction_mode",
        type=str,
        default="full",
        choices=("full", "batch_aggregate"),
    )
    parser.add_argument(
        "--salda_ga_validation_batch_size",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--salda_ga_validation_reanchor_interval",
        type=int,
        default=50,
    )
    parser.add_argument("--salda_ga_stop_epoch", type=int, default=-1)
    parser.add_argument("--salda_ga_score_start_epoch", type=int, default=0)
    parser.add_argument("--salda_ga_score_stop_epoch", type=int, default=-1)
    parser.add_argument("--salda_ga_action_start_epoch", type=int, default=0)
    parser.add_argument("--salda_ga_action_stop_epoch", type=int, default=-1)
    parser.add_argument("--salda_ga_git_commit", type=str, default="")
    parser.add_argument("--salda_ga_soft_label_dose", type=float, default=0.01)
    parser.add_argument("--salda_ga_maximum_rows", type=int, default=128)
    parser.add_argument(
        "--salda_ga_max_weight_deviation",
        type=float,
        default=0.05,
    )
    parser.add_argument("--salda_ga_weight_temperature", type=float, default=1.0)
    parser.add_argument("--salda_ga_minimum_relative_ess", type=float, default=0.9)
    parser.add_argument("--salda_ga_minimum_gain", type=float, default=0.0)
    parser.add_argument("--salda_ga_minimum_label_margin", type=float, default=0.0)
    parser.add_argument(
        "--salda_ga_minimum_relative_label_margin",
        type=float,
        default=0.0,
    )
    parser.add_argument("--salda_ga_fallback_enabled", type=str2bool, default=False)
    parser.add_argument(
        "--salda_ga_fallback_soft_label_dose",
        type=float,
        default=0.01,
    )
    parser.add_argument("--salda_ga_audit_mode", type=str2bool, default=False)
    parser.add_argument(
        "--salda_ga_profile_components",
        type=str2bool,
        default=False,
    )
