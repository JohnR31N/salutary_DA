"""Post-hoc sealed-test evaluation of saved IF-AugNet retrain checkpoints.

Backfills official test numbers for runs that trained with --final_test
false: loads each arm's val-best retrain checkpoint from
``<run_dir>/checkpoints/trials/<arm>/`` and evaluates it exactly once on the
official test split, writing the same ``<arm>_final_test.csv`` companion that
--final_test true would have produced. Model selection stays on validation;
the test split is never used for any decision.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from allthemix.utils.backend_environment import validate_jax_environment

validate_jax_environment()

import jax

from allthemix.config import load_yaml_config
from allthemix.data.pipeline import build_test_pipeline
from allthemix.data.preprocessors.selector import get_metadata
from allthemix.networks.builder import build_model
from allthemix.training.engine.single.loop import evaluate
from allthemix.training.engine.single.train import create_train_state
from allthemix.utils.checkpoint import restore_model_state_file
from allthemix.utils.cli import str2bool
from allthemix.utils.experiment_logger import (
    format_final_test_message,
    write_final_test_result,
)

DEFAULT_CHECKPOINT_NAME = "ifaugnet_retrain_best.msgpack"
DEFAULT_CONFIG_NAME = "probe_resolved_config.yaml"


def discover_arm_checkpoints(
    run_dir: Path,
    checkpoint_name: str,
) -> dict[str, Path]:
    """Find every trial arm with a saved retrain checkpoint."""
    trials_root = run_dir / "checkpoints" / "trials"
    if not trials_root.is_dir():
        raise FileNotFoundError(
            f"Run directory has no checkpoints/trials: {trials_root}"
        )

    arms = {}
    for trial_dir in sorted(trials_root.iterdir()):
        checkpoint = trial_dir / checkpoint_name
        if checkpoint.is_file():
            arms[trial_dir.name] = checkpoint

    if not arms:
        raise FileNotFoundError(
            f"No {checkpoint_name} found under {trials_root}."
        )

    return arms


def build_template_state(
    config: dict[str, Any],
    batch_size: int,
):
    """Build an evaluation-only train state matching the run's model."""
    metadata = get_metadata(
        str(config["dataset"]),
    )
    model = build_model(
        name=str(config["model"]),
        num_classes=metadata.num_classes,
        resnet_stem_type=str(config.get("resnet_stem_type", "cifar")),
        preact_stem_bn_relu=bool(config.get("preact_stem_bn_relu", False)),
        preact_pytorch_default_init=bool(
            config.get("preact_pytorch_default_init", False)
        ),
    )

    # Optimizer hyperparameters are irrelevant for evaluation; the template
    # only supplies the parameter/batch-stats structure for restoration.
    return create_train_state(
        rng=jax.random.PRNGKey(0),
        model=model,
        learning_rate=float(config.get("learning_rate", 0.1)),
        momentum=float(config.get("momentum", 0.9)),
        weight_decay=float(config.get("weight_decay", 1.0e-4)),
        input_shape=(
            batch_size,
            metadata.image_size,
            metadata.image_size,
            metadata.channels,
        ),
        nesterov=bool(config.get("nesterov", False)),
    ), metadata


def evaluate_run(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Evaluate every requested arm checkpoint on the sealed test split."""
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"Run directory does not exist: {run_dir}"
        )

    config_path = (
        Path(args.config).expanduser().resolve()
        if args.config
        else run_dir / DEFAULT_CONFIG_NAME
    )
    config = load_yaml_config(config_path)
    if config.get("method") != "ifaugnet":
        raise ValueError(
            f"The run config must use method: ifaugnet ({config_path})."
        )

    data_dir = args.data_dir or str(config.get("data_dir", "./data"))
    batch_size = (
        args.batch_size
        if args.batch_size > 0
        else int(config.get("batch_size", 128))
    )

    arms = discover_arm_checkpoints(
        run_dir=run_dir,
        checkpoint_name=args.checkpoint_name,
    )
    if args.arms:
        requested = [
            arm.strip() for arm in args.arms.split(",") if arm.strip()
        ]
        missing = sorted(set(requested) - set(arms))
        if missing:
            raise ValueError(
                f"Requested arms have no checkpoint: {missing}. "
                f"Available: {sorted(arms)}."
            )
        arms = {arm: arms[arm] for arm in requested}

    template_state, metadata = build_template_state(
        config=config,
        batch_size=batch_size,
    )
    test_dataset = build_test_pipeline(
        name=str(config["dataset"]),
        data_dir=data_dir,
        batch_size=batch_size,
        tiny_imagenet_normalization=str(
            config.get("tiny_imagenet_normalization", "imagenet")
        ),
        val_source=str(config.get("val_source", "train")),
        validation_split=float(config.get("validation_split", 0.0)),
    )

    metrics_dir = run_dir / "metrics"
    results = {}
    for arm, checkpoint_path in arms.items():
        state, loaded = restore_model_state_file(
            state=template_state,
            checkpoint_path=checkpoint_path,
        )
        (
            test_loss,
            test_top1_accuracy,
            test_top5_accuracy,
            test_top1_error,
            test_top5_error,
        ) = evaluate(
            state=state,
            test_ds=test_dataset,
            num_classes=metadata.num_classes,
            max_eval_steps=args.max_eval_steps,
        )
        print(
            f"[{arm}] loaded fields={','.join(loaded)} | "
            + format_final_test_message(
                test_loss=test_loss,
                test_top1_accuracy=test_top1_accuracy,
                test_top5_accuracy=test_top5_accuracy,
                test_top1_error=test_top1_error,
                test_top5_error=test_top5_error,
            )
        )
        results[arm] = {
            "checkpoint": str(checkpoint_path),
            "test_loss": float(test_loss),
            "test_top1_accuracy": float(test_top1_accuracy),
            "test_top5_accuracy": float(test_top5_accuracy),
            "test_top1_error": float(test_top1_error),
            "test_top5_error": float(test_top5_error),
        }

        if args.write_csv:
            metrics_dir.mkdir(parents=True, exist_ok=True)
            # Same companion path --final_test would have written, so the
            # probe summary block picks these up on its next run.
            write_final_test_result(
                output_path=metrics_dir / f"{arm}.csv",
                test_loss=test_loss,
                test_top1_accuracy=test_top1_accuracy,
                test_top5_accuracy=test_top5_accuracy,
                test_top1_error=test_top1_error,
                test_top5_error=test_top5_error,
            )

    summary = {
        "run_dir": str(run_dir),
        "config": str(config_path),
        "checkpoint_name": args.checkpoint_name,
        "dataset": str(config["dataset"]),
        "selection": (
            "validation-best checkpoint per arm; sealed final evaluation "
            "used once"
        ),
        "results": results,
    }
    summary_path = run_dir / "final_test_from_best.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"SUMMARY={summary_path}")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate saved IF-AugNet retrain checkpoints on the sealed "
            "test split (post-hoc --final_test backfill)."
        ),
    )
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--config", default="")
    parser.add_argument("--data_dir", default="")
    parser.add_argument("--arms", default="")
    parser.add_argument(
        "--checkpoint_name",
        default=DEFAULT_CHECKPOINT_NAME,
    )
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--max_eval_steps", type=int, default=-1)
    parser.add_argument("--write_csv", type=str2bool, default=True)
    return parser.parse_args()


def main() -> None:
    evaluate_run(_parse_args())


if __name__ == "__main__":
    main()
