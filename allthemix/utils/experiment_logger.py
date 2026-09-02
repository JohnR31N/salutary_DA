from __future__ import annotations

import argparse
import csv
from pathlib import Path


def build_run_name(args: argparse.Namespace) -> str:
    """Build a stable experiment run name."""
    if args.run_name:
        return args.run_name

    return f"{args.dataset}_{args.method}_{args.model}"


def build_output_path(args: argparse.Namespace) -> Path:
    """Build the CSV output path and create its parent directory."""
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.output_name:
        output_name = args.output_name
    else:
        output_name = f"{build_run_name(args)}.csv"

    output_path = output_dir / output_name

    return output_path


def build_final_test_output_path(
    output_path: Path,
) -> Path:
    """Build the companion CSV path for final test metrics."""
    return output_path.with_name(
        f"{output_path.stem}_final_test{output_path.suffix}"
    )


def write_csv_header(
    output_path: Path,
    extra_metric_names: list[str] | None = None,
) -> None:
    """Write the metrics CSV header row."""
    if extra_metric_names is None:
        extra_metric_names = []

    with output_path.open(
        mode="w",
        newline="",
    ) as file:
        writer = csv.writer(file)

        header = [
            "epoch",
            "train_loss",
            "train_accuracy",
            "eval_loss",
            "eval_top1_accuracy",
            "eval_top5_accuracy",
            "eval_top1_error",
            "eval_top5_error",
            "best_top1_error",
            "best_epoch",
            "epoch_time",
            *extra_metric_names,
        ]

        writer.writerow(header)


def write_final_test_result(
    output_path: Path,
    test_loss: float,
    test_top1_accuracy: float,
    test_top5_accuracy: float,
    test_top1_error: float,
    test_top5_error: float,
) -> None:
    """Write final test metrics to a companion CSV file."""
    final_test_output_path = build_final_test_output_path(
        output_path=output_path,
    )

    with final_test_output_path.open(
        mode="w",
        newline="",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "test_loss",
                "test_top1_accuracy",
                "test_top5_accuracy",
                "test_top1_error",
                "test_top5_error",
            ]
        )

        writer.writerow(
            [
                test_loss,
                test_top1_accuracy,
                test_top5_accuracy,
                test_top1_error,
                test_top5_error,
            ]
        )


def append_epoch_result(
    output_path: Path,
    epoch: int,
    train_loss: float,
    train_accuracy: float,
    eval_loss: float,
    eval_top1_accuracy: float,
    eval_top5_accuracy: float,
    eval_top1_error: float,
    eval_top5_error: float,
    best_top1_error: float,
    best_epoch: int,
    epoch_time: float | None,
    extra_metrics: dict[str, float] | None = None,
    extra_metric_names: list[str] | None = None,
) -> None:
    """Append one epoch of train/eval metrics to the CSV file."""
    if extra_metrics is None:
        extra_metrics = {}

    if extra_metric_names is None:
        extra_metric_names = []

    with output_path.open(
        mode="a",
        newline="",
    ) as file:
        writer = csv.writer(file)

        row = [
            epoch,
            train_loss,
            train_accuracy,
            eval_loss,
            eval_top1_accuracy,
            eval_top5_accuracy,
            eval_top1_error,
            eval_top5_error,
            best_top1_error,
            best_epoch,
            epoch_time,
        ]

        row.extend(
            extra_metrics.get(
                name,
                "",
            )
            for name in extra_metric_names
        )

        writer.writerow(row)


def format_epoch_message(
    epoch: int,
    total_epochs: int,
    train_loss: float,
    train_accuracy: float,
    eval_loss: float,
    eval_top1_accuracy: float,
    eval_top5_accuracy: float,
    eval_top1_error: float,
    eval_top5_error: float,
    best_top1_error: float,
    epoch_time: float | None,
    extra_metrics: dict[str, float] | None = None,
    eval_name: str = "eval",
) -> str:
    """Format one human-readable epoch progress message."""
    message = (
        f"Epoch {epoch}/{total_epochs} | "
        f"train loss: {train_loss:.4f} | "
        f"train acc: {train_accuracy:.4f} | "
        f"{eval_name} loss: {eval_loss:.4f} | "
        f"top1 acc: {eval_top1_accuracy:.4f} | "
        f"top5 acc: {eval_top5_accuracy:.4f} | "
        f"top1 error: {eval_top1_error * 100:.2f}% | "  # Display error as percent.
        f"top5 error: {eval_top5_error * 100:.2f}% | "  # Display error as percent.
        f"best top1 error: {best_top1_error * 100:.2f}%"  # Display best error as percent.
    )

    if epoch_time is not None:
        message += f" | time: {epoch_time:.2f}s"

    if extra_metrics:
        if "mix_lam_mean" in extra_metrics:
            message += (
                f" | mix lam: {extra_metrics['mix_lam_mean']:.3f}"
                f" [{extra_metrics.get('mix_lam_min', 0.0):.3f},"
                f" {extra_metrics.get('mix_lam_max', 0.0):.3f}]"
                f" | mix changed: "
                f"{extra_metrics.get('mix_changed_ratio', 0.0):.3f}"
            )

        if "sumix_lam_a_mean" in extra_metrics:
            message += (
                f" | sumix lam_a: {extra_metrics['sumix_lam_a_mean']:.3f}"
                f" [{extra_metrics.get('sumix_lam_a_min', 0.0):.3f},"
                f" {extra_metrics.get('sumix_lam_a_max', 0.0):.3f}]"
            )

        if "metaaugment_policy_loss" in extra_metrics:
            message += (
                " | meta policy loss: "
                f"{extra_metrics['metaaugment_policy_loss']:.4f}"
                " | inner lr: "
                f"{extra_metrics.get('metaaugment_inner_lr', 0.0):.5f}"
            )

    return message


def format_final_test_message(
    test_loss: float,
    test_top1_accuracy: float,
    test_top5_accuracy: float,
    test_top1_error: float,
    test_top5_error: float,
) -> str:
    """Format final test metrics."""
    return (
        "Final test | "
        f"loss: {test_loss:.4f} | "
        f"top1 acc: {test_top1_accuracy:.4f} | "
        f"top5 acc: {test_top5_accuracy:.4f} | "
        f"top1 error: {test_top1_error * 100:.2f}% | "
        f"top5 error: {test_top5_error * 100:.2f}%"
    )
