from __future__ import annotations

import csv
from pathlib import Path


def test_collect_config_paths_uses_requested_order(
    tmp_path: Path,
) -> None:
    """Verify suite configs are collected in the requested order."""
    from allthemix.cli.run_suite import _collect_config_paths

    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    for name in (
        "cutmix",
        "baseline",
    ):
        (config_dir / f"{name}.yaml").write_text(
            f"method: {name}\n",
            encoding="utf-8",
        )

    config_paths = _collect_config_paths(
        config_dir=config_dir,
        config_names=[
            "baseline",
            "cutmix",
        ],
        missing="error",
    )

    assert [
        path.stem
        for path in config_paths
    ] == [
        "baseline",
        "cutmix",
    ]


def test_collect_config_paths_can_collect_all_yaml(
    tmp_path: Path,
) -> None:
    """Verify suite config collection can use every YAML in a directory."""
    from allthemix.cli.run_suite import _collect_config_paths

    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    for name in (
        "z_method",
        "a_method",
    ):
        (config_dir / f"{name}.yaml").write_text(
            f"method: {name}\n",
            encoding="utf-8",
        )

    config_paths = _collect_config_paths(
        config_dir=config_dir,
        config_names=[],
        missing="error",
    )

    assert [
        path.stem
        for path in config_paths
    ] == [
        "a_method",
        "z_method",
    ]


def test_metric_completion_uses_config_epochs(
    tmp_path: Path,
) -> None:
    """Verify resume completion checks the final epoch."""
    from allthemix.cli.run_suite import _is_metric_complete

    metric_csv = tmp_path / "baseline.csv"
    metric_csv.write_text(
        "\n".join(
            [
                "epoch,eval_top1_error,best_top1_error,best_epoch",
                "1,0.5,0.5,1",
                "2,0.4,0.4,2",
            ]
        ),
        encoding="utf-8",
    )

    assert _is_metric_complete(
        metric_csv=metric_csv,
        config={
            "epochs": 2,
        },
    )
    assert not _is_metric_complete(
        metric_csv=metric_csv,
        config={
            "epochs": 3,
        },
    )


def test_write_summary_reads_metric_and_final_test(
    tmp_path: Path,
) -> None:
    """Verify suite summary includes final eval, best eval, and final test."""
    from allthemix.cli.run_suite import _write_summary

    config_dir = tmp_path / "configs"
    metrics_dir = tmp_path / "metrics"
    config_dir.mkdir()
    metrics_dir.mkdir()

    config_path = config_dir / "baseline.yaml"
    config_path.write_text(
        "method: baseline\n",
        encoding="utf-8",
    )

    (metrics_dir / "baseline.csv").write_text(
        "\n".join(
            [
                "epoch,eval_top1_error,best_top1_error,best_epoch",
                "1,0.5,0.5,1",
                "2,0.4,0.3,1",
            ]
        ),
        encoding="utf-8",
    )
    (metrics_dir / "baseline_final_test.csv").write_text(
        "\n".join(
            [
                "test_top1_error",
                "0.35",
            ]
        ),
        encoding="utf-8",
    )

    summary_csv = tmp_path / "summary.csv"
    _write_summary(
        summary_csv=summary_csv,
        metrics_dir=metrics_dir,
        config_paths=[
            config_path,
        ],
    )

    with summary_csv.open(
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(
            csv.DictReader(
                file,
            )
        )

    assert rows == [
        {
            "config_name": "baseline",
            "method": "baseline",
            "epochs_done": "2",
            "final_eval_top1_error": "40.00",
            "best_eval_top1_error": "30.00",
            "best_epoch": "1",
            "final_test_top1_error": "35.00",
        }
    ]
