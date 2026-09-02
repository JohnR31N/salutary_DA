from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from allthemix.config import load_yaml_config
from allthemix.methods.utils.validation import normalize_method_name
from allthemix.utils.cli import str2bool
from allthemix.utils.experiment_logger import build_final_test_output_path

DEFAULT_CONFIG_NAMES = (
    "baseline",
    "mixup",
    "cutmix",
    "cutmix_sumix",
    "resize",
    "fmix",
    "saliencymix",
    "guided_sr",
    "catchupmix",
)

SALIENCY_METHODS = (
    "saliencymix",
    "guidedmixup",
)


def _timestamp() -> str:
    """Build a filesystem-friendly timestamp."""
    return datetime.now().strftime(
        "%Y%m%d_%H%M%S",
    )


def _default_run_prefix(
    config_dir: Path,
) -> str:
    """Infer a compact run prefix from a config directory path."""
    parts = config_dir.parts

    if len(parts) >= 2:
        return f"{parts[-2]}_{parts[-1]}"

    return config_dir.name


def _write_status_header(
    status_csv: Path,
) -> None:
    """Create the run status CSV header."""
    with status_csv.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(
            file,
        )
        writer.writerow(
            [
                "config",
                "metric_csv",
                "final_test_csv",
                "log_path",
                "status",
                "exit_code",
                "started_at",
                "ended_at",
            ]
        )


def _append_status(
    status_csv: Path,
    row: dict[str, Any],
) -> None:
    """Append one config run status row."""
    with status_csv.open(
        mode="a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "config",
                "metric_csv",
                "final_test_csv",
                "log_path",
                "status",
                "exit_code",
                "started_at",
                "ended_at",
            ],
        )
        writer.writerow(
            row,
        )


def _load_metric_rows(
    metric_csv: Path,
) -> list[dict[str, str]]:
    """Load metric rows if a metric file exists."""
    if not metric_csv.exists() or metric_csv.stat().st_size == 0:
        return []

    with metric_csv.open(
        newline="",
        encoding="utf-8",
    ) as file:
        return list(
            csv.DictReader(
                file,
            )
        )


def _is_metric_complete(
    metric_csv: Path,
    config: dict[str, Any],
) -> bool:
    """Return whether a metric CSV reaches the configured epoch count."""
    rows = _load_metric_rows(
        metric_csv=metric_csv,
    )

    if not rows:
        return False

    target_epochs = int(
        config.get(
            "epochs",
            0,
        )
    )

    if target_epochs <= 0:
        return False

    try:
        final_epoch = int(
            float(
                rows[-1].get(
                    "epoch",
                    "0",
                )
            )
        )
    except ValueError:
        return False

    return final_epoch >= target_epochs


def _load_final_test_row(
    final_test_csv: Path,
) -> dict[str, str] | None:
    """Load the final-test row if present."""
    if not final_test_csv.exists() or final_test_csv.stat().st_size == 0:
        return None

    with final_test_csv.open(
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(
            csv.DictReader(
                file,
            )
        )

    if not rows:
        return None

    return rows[0]


def _to_percent_string(
    value: str | None,
) -> str:
    """Format a stored 0-1 metric value as a percentage string."""
    if value in (None, ""):
        return ""

    return f"{float(value) * 100.0:.2f}"


def _build_summary_rows(
    metrics_dir: Path,
    config_paths: list[Path],
) -> list[dict[str, str]]:
    """Build compact per-config summary rows from metric CSVs."""
    summary_rows = []

    for config_path in config_paths:
        config_name = config_path.stem
        metric_csv = metrics_dir / f"{config_name}.csv"
        final_test_csv = build_final_test_output_path(
            metric_csv,
        )
        rows = _load_metric_rows(
            metric_csv=metric_csv,
        )

        if not rows:
            summary_rows.append(
                {
                    "config_name": config_name,
                    "method": "",
                    "epochs_done": "0",
                    "final_eval_top1_error": "",
                    "best_eval_top1_error": "",
                    "best_epoch": "",
                    "final_test_top1_error": "",
                }
            )
            continue

        last_row = rows[-1]
        final_test_row = _load_final_test_row(
            final_test_csv=final_test_csv,
        )

        summary_rows.append(
            {
                "config_name": config_name,
                "method": config_name,
                "epochs_done": last_row.get(
                    "epoch",
                    "",
                ),
                "final_eval_top1_error": _to_percent_string(
                    last_row.get(
                        "eval_top1_error",
                    )
                ),
                "best_eval_top1_error": _to_percent_string(
                    last_row.get(
                        "best_top1_error",
                    )
                ),
                "best_epoch": last_row.get(
                    "best_epoch",
                    "",
                ),
                "final_test_top1_error": _to_percent_string(
                    None
                    if final_test_row is None
                    else final_test_row.get(
                        "test_top1_error",
                    )
                ),
            }
        )

    return summary_rows


def _write_summary(
    summary_csv: Path,
    metrics_dir: Path,
    config_paths: list[Path],
) -> None:
    """Write a compact summary CSV for finished runs."""
    rows = _build_summary_rows(
        metrics_dir=metrics_dir,
        config_paths=config_paths,
    )

    with summary_csv.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "config_name",
                "method",
                "epochs_done",
                "final_eval_top1_error",
                "best_eval_top1_error",
                "best_epoch",
                "final_test_top1_error",
            ],
        )
        writer.writeheader()
        writer.writerows(
            rows,
        )


def _collect_config_paths(
    config_dir: Path,
    config_names: list[str],
    missing: str,
) -> list[Path]:
    """Collect suite config paths in the requested order."""
    if not config_dir.exists():
        raise FileNotFoundError(
            f"Config directory not found: {config_dir}"
        )

    if not config_names:
        return sorted(
            config_dir.glob(
                "*.yaml",
            )
        )

    config_paths = []
    missing_paths = []

    for config_name in config_names:
        config_path = config_dir / f"{config_name}.yaml"
        if config_path.exists():
            config_paths.append(
                config_path,
            )
        else:
            missing_paths.append(
                config_path,
            )

    if missing_paths and missing == "error":
        missing_list = ", ".join(
            str(path)
            for path in missing_paths
        )
        raise FileNotFoundError(
            f"Missing config files: {missing_list}"
        )

    return config_paths


def _ensure_saliency_maps(
    config_paths: list[Path],
) -> None:
    """Generate required train saliency maps if saliency configs need them."""
    from allthemix.data.saliency.saliency_io import (
        get_train_saliency_path,
    )

    jobs: dict[tuple[str, str, str], Path] = {}

    for config_path in config_paths:
        config = load_yaml_config(
            config_path,
        )
        method = normalize_method_name(
            str(
                config.get(
                    "method",
                    "",
                )
            )
        )

        if method not in SALIENCY_METHODS:
            continue

        dataset = str(
            config.get(
                "dataset",
                "cifar10",
            )
        )
        data_dir = str(
            config.get(
                "data_dir",
                "./data",
            )
        )
        saliency_dir = str(
            config.get(
                "saliency_dir",
                "./data",
            )
        )
        saliency_path = get_train_saliency_path(
            dataset_name=dataset,
            saliency_dir=saliency_dir,
        )
        jobs[(dataset, data_dir, saliency_dir)] = saliency_path

    for (dataset, data_dir, saliency_dir), saliency_path in jobs.items():
        if saliency_path.exists():
            print(
                f"Found saliency maps: {saliency_path}"
            )
            continue

        print(
            f"Missing saliency maps: {saliency_path}"
        )
        print(
            "Generating saliency maps before running saliency-based configs."
        )

        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "allthemix.data.saliency",
                "--dataset",
                dataset,
                "--data_dir",
                data_dir,
                "--output_dir",
                saliency_dir,
            ]
        )


def _stream_command_to_log(
    command: list[str],
    log_path: Path,
) -> int:
    """Run a command and stream combined stdout/stderr to console and log."""
    with log_path.open(
        mode="w",
        encoding="utf-8",
    ) as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        assert process.stdout is not None

        for line in process.stdout:
            print(
                line,
                end="",
            )
            log_file.write(
                line,
            )

        return process.wait()


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse suite-runner arguments and keep unknown train overrides."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_dir", type=str, required=True)
    parser.add_argument("--config_names", nargs="*", default=list(DEFAULT_CONFIG_NAMES))
    parser.add_argument("--all_configs", type=str2bool, default=False)
    parser.add_argument("--missing", choices=("error", "skip"), default="error")
    parser.add_argument("--output_root", type=str, default="outputs/experiment_run")
    parser.add_argument("--run_dir", type=str, default="")
    parser.add_argument("--resume", type=str2bool, default=True)
    parser.add_argument("--generate_saliency", type=str2bool, default=True)
    parser.add_argument("--zip_results", type=str2bool, default=False)

    return parser.parse_known_args()


def main() -> None:
    """Run a directory of train configs and summarize the results."""
    args, train_overrides = parse_args()

    config_dir = Path(
        args.config_dir,
    )
    config_names = [] if args.all_configs else args.config_names
    config_paths = _collect_config_paths(
        config_dir=config_dir,
        config_names=config_names,
        missing=args.missing,
    )

    if not config_paths:
        raise ValueError(
            f"No config files selected from: {config_dir}"
        )

    run_dir = (
        Path(
            args.run_dir,
        )
        if args.run_dir
        else Path(
            args.output_root,
        )
        / f"{_default_run_prefix(config_dir)}_{_timestamp()}"
    )

    metrics_dir = run_dir / "metrics"
    logs_dir = run_dir / "logs"
    configs_dir = run_dir / "configs"
    checkpoints_dir = run_dir / "checkpoints"
    status_csv = run_dir / "run_status.csv"
    summary_csv = run_dir / "summary.csv"

    for directory in (
        metrics_dir,
        logs_dir,
        configs_dir,
        checkpoints_dir,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    if not args.resume or not status_csv.exists():
        _write_status_header(
            status_csv=status_csv,
        )

    if args.generate_saliency:
        _ensure_saliency_maps(
            config_paths=config_paths,
        )

    print(
        f"Config dir: {config_dir}"
    )
    print(
        f"Run dir: {run_dir}"
    )
    print(
        f"Train overrides: {' '.join(train_overrides) if train_overrides else '(none)'}"
    )
    print()

    for config_path in config_paths:
        config_name = config_path.stem
        config = load_yaml_config(
            config_path,
        )
        metric_csv = metrics_dir / f"{config_name}.csv"
        final_test_csv = build_final_test_output_path(
            metric_csv,
        )
        log_path = logs_dir / f"{config_name}.log"
        raw_config_snapshot = configs_dir / f"{config_name}.yaml"
        resolved_config_snapshot = configs_dir / f"{config_name}.resolved.yaml"

        shutil.copy2(
            config_path,
            raw_config_snapshot,
        )

        with resolved_config_snapshot.open(
            mode="w",
            encoding="utf-8",
        ) as file:
            yaml.safe_dump(
                config,
                file,
                sort_keys=False,
            )

        if args.resume and _is_metric_complete(
            metric_csv=metric_csv,
            config=config,
        ):
            print(
                f"Skipping completed config: {config_path}"
            )
            continue

        started_at = datetime.now().isoformat(
            timespec="seconds",
        )
        print(
            "=" * 60
        )
        print(
            f"Running config: {config_path}"
        )
        print(
            f"Metric CSV: {metric_csv}"
        )
        print(
            f"Log: {log_path}"
        )
        print(
            f"Started: {started_at}"
        )
        print(
            "=" * 60
        )

        command = [
            sys.executable,
            "-u",
            "-m",
            "allthemix.cli.train",
            "--config",
            str(config_path),
            "--save_csv",
            "true",
            "--output_dir",
            str(metrics_dir),
            "--output_name",
            f"{config_name}.csv",
            "--checkpoint_dir",
            str(checkpoints_dir),
            *train_overrides,
        ]
        exit_code = _stream_command_to_log(
            command=command,
            log_path=log_path,
        )
        ended_at = datetime.now().isoformat(
            timespec="seconds",
        )
        status = "ok" if exit_code == 0 else "failed"

        _append_status(
            status_csv=status_csv,
            row={
                "config": str(
                    config_path,
                ),
                "metric_csv": str(
                    metric_csv,
                ),
                "final_test_csv": str(
                    final_test_csv,
                ),
                "log_path": str(
                    log_path,
                ),
                "status": status,
                "exit_code": exit_code,
                "started_at": started_at,
                "ended_at": ended_at,
            },
        )
        _write_summary(
            summary_csv=summary_csv,
            metrics_dir=metrics_dir,
            config_paths=config_paths,
        )

        print(
            f"Finished: {config_path} | status={status} | "
            f"exit_code={exit_code} | ended={ended_at}"
        )
        print()

        if exit_code != 0:
            raise SystemExit(
                exit_code,
            )

    _write_summary(
        summary_csv=summary_csv,
        metrics_dir=metrics_dir,
        config_paths=config_paths,
    )

    if args.zip_results:
        archive_path = shutil.make_archive(
            base_name=str(
                run_dir,
            ),
            format="zip",
            root_dir=run_dir,
        )
        print(
            f"Zipped results to: {archive_path}"
        )

    print(
        "All selected runs finished."
    )
    print(
        f"Run directory: {run_dir}"
    )
    print(
        f"Summary CSV: {summary_csv}"
    )


if __name__ == "__main__":
    main()
