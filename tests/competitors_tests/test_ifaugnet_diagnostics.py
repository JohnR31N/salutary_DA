from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from allthemix.competitors.ifaugnet.diagnostics import (
    select_early_classifier_epoch,
    summarize_overnight_suite,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_classifier_metric(
    path: Path,
    *,
    final_error: float,
    best_error: float,
    best_epoch: int,
) -> None:
    _write_csv(
        path,
        [
            {
                "epoch": 100,
                "train_accuracy": 0.99,
                "eval_top1_error": final_error,
                "best_top1_error": best_error,
                "best_epoch": best_epoch,
            }
        ],
    )


def _write_influence_metric(path: Path, *, loss: float) -> None:
    _write_csv(
        path,
        [
            {
                "step": 10,
                "loss": loss,
                "estimated_val_loss_reduction": -loss,
                "replacement_influence_std": 0.03,
                "identity_l2": 0.02,
                "accuracy_retention": 0.95,
                "tau_saturation_fraction": 0.1,
                "augmented_l1": 0.04,
                "gradient_global_norm": 0.5,
                "policy_healthy": 1.0,
            }
        ],
    )


def test_select_early_classifier_epoch_uses_first_threshold_crossing(
    tmp_path: Path,
) -> None:
    metric = tmp_path / "classifier.csv"
    _write_csv(
        metric,
        [
            {"epoch": 1, "train_accuracy": 0.75},
            {"epoch": 2, "train_accuracy": 0.95},
            {"epoch": 3, "train_accuracy": 0.99},
        ],
    )

    assert select_early_classifier_epoch(metric, 0.95) == 2


def test_select_early_classifier_epoch_rejects_unreached_target(
    tmp_path: Path,
) -> None:
    metric = tmp_path / "classifier.csv"
    _write_csv(metric, [{"epoch": 1, "train_accuracy": 0.75}])

    with pytest.raises(ValueError, match="never reached"):
        select_early_classifier_epoch(metric, 0.95)


def test_summarize_overnight_suite_isolates_requested_factors(
    tmp_path: Path,
) -> None:
    base_run = tmp_path / "base"
    diagnostics = base_run / "overnight_diagnostics"
    base_metrics = base_run / "metrics"
    diagnostic_metrics = diagnostics / "metrics"
    diagnostics.mkdir(parents=True)
    (diagnostics / "ifaugnet_jax_stable.resolved.yaml").write_text(
        "ifaugnet_policy_classifier_checkpoint: best\n"
        "ifaugnet_learning_rate: 0.0001\n",
        encoding="utf-8",
    )

    _write_classifier_metric(
        base_metrics / "no_aug_p000.csv",
        final_error=0.265,
        best_error=0.2616,
        best_epoch=93,
    )
    _write_classifier_metric(
        base_metrics / "pretrain_p025.csv",
        final_error=0.280,
        best_error=0.2750,
        best_epoch=95,
    )
    _write_classifier_metric(
        base_metrics / "learned_p010.csv",
        final_error=0.272,
        best_error=0.2700,
        best_epoch=96,
    )
    _write_classifier_metric(
        base_metrics / "learned_p025.csv",
        final_error=0.271,
        best_error=0.2706,
        best_epoch=100,
    )
    existing_influence = base_metrics / "learned_policy_influence.csv"
    _write_influence_metric(existing_influence, loss=-0.1)

    policies = (
        ("best_nopre_lr1e4", "best", False, 0.0001),
        ("best_nopre_lr5e5", "best", False, 0.00005),
        ("early_nopre_lr1e4", "early", False, 0.0001),
        ("early_pretrain_lr1e4", "early", True, 0.0001),
    )
    policy_rows = []
    trial_rows = []
    trial_errors = {
        "best_nopre_lr1e4": 0.255,
        "best_nopre_lr5e5": 0.250,
        "early_nopre_lr1e4": 0.258,
        "early_pretrain_lr1e4": 0.263,
    }
    for policy_id, source, pretrain, learning_rate in policies:
        influence_metric = diagnostic_metrics / f"{policy_id}_influence.csv"
        _write_influence_metric(influence_metric, loss=-0.2)
        policy_rows.append(
            {
                "policy_id": policy_id,
                "classifier_source": source,
                "gan_pretrain": str(pretrain).lower(),
                "influence_lr": learning_rate,
                "min_lr": learning_rate / 10.0,
                "pretrain_steps": 2000 if pretrain else 0,
                "checkpoint_root": str(diagnostics / "checkpoints" / policy_id),
                "influence_metric_csv": str(influence_metric),
                "status": "ok",
                "exit_code": 0,
            }
        )
        for probability in (0.10, 0.25):
            arm = f"{policy_id}_p{int(probability * 100):03d}"
            metric = diagnostic_metrics / f"{arm}.csv"
            error = trial_errors[policy_id] + (0.002 if probability == 0.25 else 0)
            _write_classifier_metric(
                metric,
                final_error=error + 0.001,
                best_error=error,
                best_epoch=98,
            )
            trial_rows.append(
                {
                    "arm": arm,
                    "policy_id": policy_id,
                    "classifier_source": source,
                    "gan_pretrain": str(pretrain).lower(),
                    "influence_lr": learning_rate,
                    "aug_probability": probability,
                    "metric_csv": str(metric),
                    "influence_metric_csv": str(influence_metric),
                    "status": "ok",
                    "exit_code": 0,
                }
            )

    _write_csv(diagnostics / "policy_index.csv", policy_rows)
    _write_csv(diagnostics / "trial_index.csv", trial_rows)

    verdict = summarize_overnight_suite(base_run, diagnostics)

    assert verdict["official_test_used"] is False
    assert verdict["learned_policy_beats_no_augmentation"] is True
    assert verdict["best_learned_arm"] == "best_nopre_lr5e5_p010"
    with (diagnostics / "factor_effects.csv").open(newline="") as file:
        factors = {row["factor"] for row in csv.DictReader(file)}
    assert factors == {
        "application_probability",
        "classifier_saturation",
        "gan_pretrain",
        "influence_lr",
    }
    with (diagnostics / "influence_health_summary.csv").open(
        newline=""
    ) as file:
        health = list(csv.DictReader(file))
    assert len(health) == 5
    saved_verdict = json.loads(
        (diagnostics / "diagnostic_verdict.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved_verdict == verdict
