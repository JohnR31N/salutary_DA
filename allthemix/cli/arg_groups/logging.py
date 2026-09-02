"""Run naming, checkpoints, logging, and debug probes."""

from __future__ import annotations

import argparse

from allthemix.utils.cli import str2bool


def add_logging_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    """Register run naming, checkpoints, logging, and debug probes flags."""
    parser.add_argument(
        "--debug_train_source",
        type=str,
        default="none",
        choices=("none", "val_only", "train_plus_val"),
        help=(
            "Leakage diagnostics ONLY: swap what the TRAIN pipeline "
            "consumes while the held-out validation/eval pipelines stay "
            "untouched. val_only trains on the validation partition "
            "itself (val loss should collapse toward 0 if the split and "
            "preprocessing are aligned); train_plus_val trains on the "
            "full source split including the validation examples "
            "(leakage oracle). Results are diagnostics, never table "
            "rows."
        ),
    )
    parser.add_argument("--debug_sumix_metrics", type=str2bool, default=False)
    parser.add_argument("--debug_mix_metrics", type=str2bool, default=False)
    parser.add_argument("--save_csv", type=str2bool, default=False)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--output_name", type=str, default="")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--wandb", type=str2bool, default=False)
    parser.add_argument("--wandb_project", type=str, default="allthemix")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument("--wandb_mode", type=str, default="")
    parser.add_argument("--wandb_tags", type=str, default="")
    parser.add_argument(
        "--run_protocol_path",
        type=str,
        default="",
        help=(
            "Immutable formal-run protocol artifact. When supplied, the "
            "trainer verifies provenance, workload counts, checkpoint "
            "selection, and sealed endpoint access before reporting success."
        ),
    )
    parser.add_argument("--save_checkpoint", type=str2bool, default=False)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--save_best_only", type=str2bool, default=False)
    parser.add_argument("--resume_checkpoint", type=str, default="")
    parser.add_argument("--pretrained_checkpoint", type=str, default="")
    parser.add_argument("--log_time", type=str2bool, default=False)
