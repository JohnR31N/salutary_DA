from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from allthemix.config import load_yaml_config

_LEARNED_ARM_PATTERN = re.compile(r"learned_p(\d{3})")


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Metric CSV does not exist: {path}")
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Metric CSV contains no rows: {path}")
    return rows


def _float(row: dict[str, str], name: str, default: float = 0.0) -> float:
    value = row.get(name, "")
    if value in {"", "NA", "None", None}:
        return default
    return float(value)


def select_early_classifier_epoch(
    metric_path: str | Path,
    target_train_accuracy: float,
) -> int:
    """Select the first epoch that reaches a non-saturated train accuracy."""
    if not 0.0 < target_train_accuracy < 1.0:
        raise ValueError("target_train_accuracy must be in (0, 1).")
    rows = _read_rows(Path(metric_path))
    for row in rows:
        if _float(row, "train_accuracy") >= target_train_accuracy:
            return int(float(row["epoch"]))
    raise ValueError(
        "Classifier never reached target train accuracy "
        f"{target_train_accuracy:.4f} in {metric_path}."
    )


def _classifier_result(
    *,
    arm: str,
    policy_id: str,
    classifier_source: str,
    gan_pretrain: bool,
    influence_lr: float,
    probability: float,
    metric_path: Path,
    influence_metric_path: Path | None,
) -> dict[str, Any]:
    last = _read_rows(metric_path)[-1]
    return {
        "arm": arm,
        "policy_id": policy_id,
        "classifier_source": classifier_source,
        "gan_pretrain": gan_pretrain,
        "influence_lr": influence_lr,
        "aug_probability": probability,
        "epochs": int(float(last["epoch"])),
        "final_val_error": _float(last, "eval_top1_error") * 100.0,
        "best_val_error": _float(last, "best_top1_error") * 100.0,
        "best_epoch": int(float(last["best_epoch"])),
        "metric_csv": str(metric_path),
        "influence_metric_csv": (
            "" if influence_metric_path is None else str(influence_metric_path)
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _effect_row(
    *,
    factor: str,
    comparison: str,
    probability: float,
    left: dict[str, Any],
    right: dict[str, Any],
    favorable_when: str,
) -> dict[str, Any]:
    delta = right["best_val_error"] - left["best_val_error"]
    return {
        "factor": factor,
        "comparison": comparison,
        "probability": probability,
        "left_arm": left["arm"],
        "left_best_val_error": left["best_val_error"],
        "right_arm": right["arm"],
        "right_best_val_error": right["best_val_error"],
        "right_minus_left_error": delta,
        "favorable_when": favorable_when,
    }


def _factor_effects(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    learned = [
        row
        for row in results
        if row["policy_id"] not in {"control", "pretrain_control"}
    ]
    lookup = {
        (
            row["classifier_source"],
            bool(row["gan_pretrain"]),
            round(float(row["influence_lr"]), 10),
            round(float(row["aug_probability"]), 4),
        ): row
        for row in learned
    }
    effects: list[dict[str, Any]] = []

    sources = sorted({str(row["classifier_source"]) for row in learned})
    lrs = sorted({round(float(row["influence_lr"]), 10) for row in learned})
    probabilities = sorted(
        {round(float(row["aug_probability"]), 4) for row in learned}
    )

    for source in sources:
        for lr in lrs:
            for probability in probabilities:
                no_pretrain = lookup.get((source, False, lr, probability))
                pretrain = lookup.get((source, True, lr, probability))
                if no_pretrain is not None and pretrain is not None:
                    effects.append(
                        _effect_row(
                            factor="gan_pretrain",
                            comparison=f"no_pretrain_to_pretrain:{source}:lr={lr:g}",
                            probability=probability,
                            left=no_pretrain,
                            right=pretrain,
                            favorable_when=(
                                "negative means GAN pretraining improves validation"
                            ),
                        )
                    )

    for pretrain in (False, True):
        for lr in lrs:
            for probability in probabilities:
                best = lookup.get(("best", pretrain, lr, probability))
                early = lookup.get(("early", pretrain, lr, probability))
                if best is not None and early is not None:
                    effects.append(
                        _effect_row(
                            factor="classifier_saturation",
                            comparison=(
                                f"best_to_early:pretrain={pretrain}:lr={lr:g}"
                            ),
                            probability=probability,
                            left=best,
                            right=early,
                            favorable_when=(
                                "negative means the earlier classifier improves policy"
                            ),
                        )
                    )

    for source in sources:
        for pretrain in (False, True):
            for probability in probabilities:
                candidates = [
                    row
                    for row in learned
                    if row["classifier_source"] == source
                    and bool(row["gan_pretrain"]) == pretrain
                    and round(float(row["aug_probability"]), 4) == probability
                ]
                candidates.sort(key=lambda row: float(row["influence_lr"]))
                if len(candidates) < 2:
                    continue
                high = candidates[-1]
                for low in candidates[:-1]:
                    effects.append(
                        _effect_row(
                            factor="influence_lr",
                            comparison=(
                                f"high_to_low:{source}:pretrain={pretrain}:"
                                f"{high['influence_lr']:g}_to_{low['influence_lr']:g}"
                            ),
                            probability=probability,
                            left=high,
                            right=low,
                            favorable_when=(
                                "negative means the lower influence LR improves validation"
                            ),
                        )
                    )

    for source in sources:
        for pretrain in (False, True):
            for lr in lrs:
                candidates = [
                    row
                    for row in learned
                    if row["classifier_source"] == source
                    and bool(row["gan_pretrain"]) == pretrain
                    and round(float(row["influence_lr"]), 10) == lr
                ]
                candidates.sort(
                    key=lambda row: float(row["aug_probability"]),
                )
                if len(candidates) < 2:
                    continue
                high = candidates[-1]
                for low in candidates[:-1]:
                    effects.append(
                        _effect_row(
                            factor="application_probability",
                            comparison=(
                                f"high_to_low:{source}:pretrain={pretrain}:"
                                f"lr={lr:g}:{high['aug_probability']:g}_to_"
                                f"{low['aug_probability']:g}"
                            ),
                            probability=float(low["aug_probability"]),
                            left=high,
                            right=low,
                            favorable_when=(
                                "negative means the lower application "
                                "probability improves validation"
                            ),
                        )
                    )

    return effects


def _influence_health(
    policy_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    summary = []
    for policy in policy_rows:
        if policy.get("status") != "ok":
            continue
        metric_path = Path(policy["influence_metric_csv"])
        if not metric_path.is_file():
            continue
        rows = _read_rows(metric_path)
        healthy = [row for row in rows if _float(row, "policy_healthy") >= 0.5]
        selected = min(
            healthy,
            key=lambda row: _float(row, "loss", float("inf")),
        ) if healthy else rows[-1]
        last = rows[-1]
        summary.append(
            {
                "policy_id": policy["policy_id"],
                "classifier_source": policy["classifier_source"],
                "gan_pretrain": policy["gan_pretrain"],
                "influence_lr": policy["influence_lr"],
                "logged_checks": len(rows),
                "healthy_checks": len(healthy),
                "selected_step": int(float(selected["step"])),
                "selected_loss": _float(selected, "loss"),
                "selected_estimated_val_reduction": _float(
                    selected,
                    "estimated_val_loss_reduction",
                ),
                "selected_replacement_influence_std": _float(
                    selected,
                    "replacement_influence_std",
                ),
                "selected_identity_l2": _float(selected, "identity_l2"),
                "selected_accuracy_retention": _float(
                    selected,
                    "accuracy_retention",
                ),
                "selected_tau_saturation": _float(
                    selected,
                    "tau_saturation_fraction",
                ),
                "selected_augmented_l1": _float(selected, "augmented_l1"),
                "selected_gradient_norm": _float(
                    selected,
                    "gradient_global_norm",
                ),
                "last_step": int(float(last["step"])),
                "last_policy_healthy": _float(last, "policy_healthy"),
            }
        )
    return summary


def summarize_overnight_suite(
    base_run: str | Path,
    diagnostic_dir: str | Path,
) -> dict[str, Any]:
    """Summarize validation-only arms and isolate policy-learning factors."""
    base_run = Path(base_run)
    diagnostic_dir = Path(diagnostic_dir)
    base_metrics = base_run / "metrics"
    config_candidates = (
        diagnostic_dir / "ifaugnet_jax_stable.resolved.yaml",
        base_run / "ifaugnet_jax_stable.resolved.yaml",
        base_run / "ifaugnet_jax_stable.yaml",
    )
    config_path = next(
        (path for path in config_candidates if path.is_file()),
        None,
    )
    if config_path is None:
        raise FileNotFoundError(
            "No frozen IF-AugNet config was found under the base or "
            "diagnostic run directory."
        )
    config = load_yaml_config(config_path)
    existing_source = config.get("ifaugnet_policy_classifier_checkpoint", "final")
    existing_lr = float(config["ifaugnet_learning_rate"])
    existing_influence_metric = base_metrics / "learned_policy_influence.csv"

    results = [
        _classifier_result(
            arm="no_aug_p000",
            policy_id="control",
            classifier_source="none",
            gan_pretrain=False,
            influence_lr=0.0,
            probability=0.0,
            metric_path=base_metrics / "no_aug_p000.csv",
            influence_metric_path=None,
        ),
        _classifier_result(
            arm="pretrain_p025",
            policy_id="pretrain_control",
            classifier_source=existing_source,
            gan_pretrain=True,
            influence_lr=0.0,
            probability=0.25,
            metric_path=base_metrics / "pretrain_p025.csv",
            influence_metric_path=None,
        ),
    ]

    policy_rows: list[dict[str, str]] = []
    if existing_influence_metric.is_file():
        policy_rows.append(
            {
                "policy_id": "existing",
                "classifier_source": str(existing_source),
                "gan_pretrain": "true",
                "influence_lr": str(existing_lr),
                "influence_metric_csv": str(existing_influence_metric),
                "status": "ok",
            }
        )
    for path in sorted(base_metrics.glob("learned_p*.csv")):
        match = _LEARNED_ARM_PATTERN.fullmatch(path.stem)
        if match is None:
            continue
        results.append(
            _classifier_result(
                arm=path.stem,
                policy_id="existing",
                classifier_source=str(existing_source),
                gan_pretrain=True,
                influence_lr=existing_lr,
                probability=int(match.group(1)) / 100.0,
                metric_path=path,
                influence_metric_path=existing_influence_metric,
            )
        )

    policy_index = diagnostic_dir / "policy_index.csv"
    if policy_index.is_file():
        with policy_index.open(newline="") as file:
            policy_rows.extend(csv.DictReader(file))

    trial_index = diagnostic_dir / "trial_index.csv"
    if trial_index.is_file():
        with trial_index.open(newline="") as file:
            for trial in csv.DictReader(file):
                if trial.get("status") != "ok":
                    continue
                metric_path = Path(trial["metric_csv"])
                if not metric_path.is_file():
                    continue
                results.append(
                    _classifier_result(
                        arm=trial["arm"],
                        policy_id=trial["policy_id"],
                        classifier_source=trial["classifier_source"],
                        gan_pretrain=trial["gan_pretrain"].lower() == "true",
                        influence_lr=float(trial["influence_lr"]),
                        probability=float(trial["aug_probability"]),
                        metric_path=metric_path,
                        influence_metric_path=Path(trial["influence_metric_csv"]),
                    )
                )

    control = results[0]
    pretrain_control = results[1]
    for row in results:
        row["delta_vs_no_aug"] = (
            row["best_val_error"] - control["best_val_error"]
        )
        row["delta_vs_pretrain_control"] = (
            row["best_val_error"] - pretrain_control["best_val_error"]
        )

    learned = [
        row
        for row in results
        if row["policy_id"] not in {"control", "pretrain_control"}
    ]
    if not learned:
        raise ValueError("No completed learned-policy arms were found.")
    best = min(
        learned,
        key=lambda row: (
            row["best_val_error"],
            row["final_val_error"],
            row["arm"],
        ),
    )

    _write_csv(diagnostic_dir / "all_validation_results.csv", results)
    effects = _factor_effects(results)
    _write_csv(diagnostic_dir / "factor_effects.csv", effects)
    health = _influence_health(policy_rows)
    _write_csv(diagnostic_dir / "influence_health_summary.csv", health)

    factor_summary = {}
    for factor in sorted({row["factor"] for row in effects}):
        factor_rows = [row for row in effects if row["factor"] == factor]
        deltas = [float(row["right_minus_left_error"]) for row in factor_rows]
        factor_summary[factor] = {
            "comparisons": len(deltas),
            "favorable_comparisons": sum(delta < 0.0 for delta in deltas),
            "mean_right_minus_left_error": sum(deltas) / len(deltas),
            "best_right_minus_left_error": min(deltas),
            "worst_right_minus_left_error": max(deltas),
        }

    verdict = {
        "official_test_used": False,
        "best_learned_arm": best["arm"],
        "best_learned_policy_id": best["policy_id"],
        "best_learned_classifier_source": best["classifier_source"],
        "best_learned_gan_pretrain": best["gan_pretrain"],
        "best_learned_influence_lr": best["influence_lr"],
        "best_learned_aug_probability": best["aug_probability"],
        "best_learned_val_error": best["best_val_error"],
        "best_learned_delta_vs_no_aug": best["delta_vs_no_aug"],
        "best_learned_delta_vs_pretrain_control": (
            best["delta_vs_pretrain_control"]
        ),
        "learned_policy_beats_no_augmentation": (
            best["best_val_error"] < control["best_val_error"]
        ),
        "completed_learned_arms": len(learned),
        "completed_factor_comparisons": len(effects),
        "factor_summary": factor_summary,
    }
    (diagnostic_dir / "diagnostic_verdict.json").write_text(
        json.dumps(verdict, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return verdict


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IF-AugNet diagnostic helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    select = subparsers.add_parser("select-early-epoch")
    select.add_argument("--metric-csv", required=True)
    select.add_argument("--target-train-accuracy", type=float, default=0.95)

    summarize = subparsers.add_parser("summarize-overnight")
    summarize.add_argument("--base-run", required=True)
    summarize.add_argument("--diagnostic-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "select-early-epoch":
        print(
            select_early_classifier_epoch(
                args.metric_csv,
                args.target_train_accuracy,
            )
        )
        return
    verdict = summarize_overnight_suite(
        args.base_run,
        args.diagnostic_dir,
    )
    print(json.dumps(verdict, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
