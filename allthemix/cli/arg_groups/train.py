"""Optimization schedule and evaluation cadence."""

from __future__ import annotations

import argparse

from allthemix.utils.cli import str2bool


def add_train_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    """Register optimization schedule and evaluation cadence flags."""
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=-1)
    parser.add_argument("--max_eval_steps", type=int, default=-1)
    parser.add_argument("--early_stop_enabled", type=str2bool, default=False)
    parser.add_argument("--early_stop_start_epoch", type=int, default=0)
    parser.add_argument("--early_stop_patience", type=int, default=20)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0)
    parser.add_argument("--eval_on_test_each_epoch", type=str2bool, default=True)
    parser.add_argument("--final_test", type=str2bool, default=False)
    parser.add_argument(
        "--final_test_checkpoint",
        type=str,
        default="best",
        choices=("last", "best"),
    )
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--nesterov", type=str2bool, default=False)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument(
        "--lr_schedule",
        type=str,
        default="step",
        choices=("step", "cosine", "warmup_cosine", "step_cosine"),
    )
    parser.add_argument("--min_learning_rate", type=float, default=0.0)
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--lr_decay_epochs", type=int, nargs="+", default=[100, 150])
    parser.add_argument("--lr_decay_rate", type=float, default=0.1)
