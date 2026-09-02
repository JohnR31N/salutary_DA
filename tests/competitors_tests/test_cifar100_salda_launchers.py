"""Regression tests for the single CIFAR-100 instantaneous-GA launcher."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPOSITORY_ROOT / "scripts/experiment_run/run_cifar100_salda_ga.sh"
SCORER = REPOSITORY_ROOT / "salutary_da/scorers/gradient_alignment.py"
CONFIG = REPOSITORY_ROOT / "configs/cifar100/preact_resnet18/salda_ga.yaml"


def _smoke_gate_source() -> str:
    """Extract the launcher's Python smoke-consumer gate verbatim."""

    source = LAUNCHER.read_text(encoding="utf-8")
    return source.split("<<'PY'\n", maxsplit=1)[1].split("\nPY\n", maxsplit=1)[0]


def _run_batch_smoke_gate(
    tmp_path: Path,
    *,
    mode: str,
    execution_overrides: dict[str, object] | None = None,
    completion_overrides: dict[str, object] | None = None,
    protocol_overrides: dict[str, object] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the exact embedded gate against one synthetic batch smoke artifact."""

    direction_active = mode not in {"baseline", "noop"}
    strategy_present = mode != "baseline"
    pool_sha = "a" * 64
    schedule_sha = "b" * 64 if strategy_present else None
    protocol = {
        "validation_pool_sha256": pool_sha,
        "validation_batch_schedule_sha256": schedule_sha,
    }
    protocol.update(protocol_overrides or {})
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    execution = {
        "train_steps": 2,
        "direction_refreshes": 2 if direction_active else 0,
        "validation_gradient_evaluations": 11 if direction_active else 0,
        "validation_exact_reanchors": 1 if direction_active else 0,
        "direction_validation_example_visits": 5_500 if direction_active else 0,
        "validation_pool_examples": 5_000,
        "validation_batch_schedule_sha256": schedule_sha,
    }
    execution.update(execution_overrides or {})
    completion = {
        "status": "SUCCESS",
        "git_commit": "f" * 40,
        "completed_epochs": 1,
        "vtest_loaded": False,
        "vtest_batches": 0,
        "endpoint_builder_calls": 0,
        "policy_mode": mode,
        "parameter_scope": "classifier_head",
        "method": "mixup",
        "seed": 0,
        "validation_direction_mode": "batch_aggregate",
        "validation_direction_main_table_eligible": False,
        "validation_examples_per_gradient_evaluation": 500,
        "validation_reanchor_interval": 50,
        "validation_pool_sha256": pool_sha,
        "validation_batch_schedule_sha256": schedule_sha,
        "protocol_artifact": str(protocol_path),
        "train_updates": 2,
        "execution": execution,
    }
    completion.update(completion_overrides or {})
    completion_path = tmp_path / "completion.json"
    completion_path.write_text(json.dumps(completion), encoding="utf-8")
    return subprocess.run(
        [
            sys.executable,
            "-",
            str(completion_path),
            "f" * 40,
            mode,
            "classifier_head",
            "mixup",
            "0",
            "batch_aggregate",
            "500",
            "50",
        ],
        input=_smoke_gate_source(),
        text=True,
        capture_output=True,
        check=False,
    )


def test_only_instantaneous_ga_entry_point_remains() -> None:
    """Keep one launcher and remove the retired diagnostic pipeline."""

    source = LAUNCHER.read_text(encoding="utf-8")
    assert "python -u -m allthemix.cli.train" in source
    assert "run_cifar100_salda_diagnostics.sh" not in source
    assert "SALDA_GATE_ARTIFACT" not in source
    assert "CORRECTNESS_ARTIFACT" not in source
    assert "SALDA_VALIDATION_CHUNK_SIZE" not in source
    assert "salda_ga_validation_chunk_size" not in source
    assert "trajectory" not in source


def test_launcher_exposes_current_runtime_modes() -> None:
    """Retain scoring, immediate actions, controls, and full-score timing."""

    source = LAUNCHER.read_text(encoding="utf-8")
    for arm in (
        "baseline)",
        "noop)",
        "last_score)",
        "last_action)",
        "last_shuffled_action)",
        "full_score)",
    ):
        assert arm in source
    assert 'SALDA_BASE_METHOD:-mixup' in source
    assert 'SALDA_VALIDATION_DIRECTION_MODE:-full' in source
    assert 'SALDA_VALIDATION_BATCH_SIZE:-500' in source
    assert 'SALDA_VALIDATION_REANCHOR_INTERVAL:-50' in source
    assert '"batch_aggregate"' in source
    assert "batch_aggregate requires the classifier_head scope" in source
    assert "--salda_ga_validation_direction_mode" in source
    assert "--salda_ga_validation_batch_size" in source
    assert "--salda_ga_validation_reanchor_interval" in source
    assert 'MAX_TRAIN_STEPS=2' in source
    for smoke_gate_field in (
        '"train_updates": expected_train_updates',
        '"validation_gradient_evaluations": expected_gradient_evaluations',
        '"validation_exact_reanchors": expected_exact_reanchors',
        '"direction_validation_example_visits": expected_validation_visits',
        '"validation_pool_examples": 5_000',
        'execution.get("validation_batch_schedule_sha256")',
        'protocol.get("validation_pool_sha256")',
        'protocol.get("validation_batch_schedule_sha256")',
    ):
        assert smoke_gate_field in source
    assert "expected_gradient_evaluations = 11" in source
    assert "expected_validation_visits = 5_500" in source


@pytest.mark.parametrize("mode", ["score_only", "noop", "baseline"])
def test_batch_smoke_gate_accepts_exact_arm_workload(
    tmp_path: Path,
    mode: str,
) -> None:
    """Accept active 11/1/5500 work or inactive zero direction work."""

    result = _run_batch_smoke_gate(tmp_path, mode=mode)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("execution_overrides", "completion_overrides", "protocol_overrides"),
    [
        ({"validation_gradient_evaluations": 10}, {}, {}),
        ({}, {"train_updates": 1}, {}),
        ({}, {"validation_batch_schedule_sha256": "c" * 64}, {}),
        ({}, {}, {"validation_pool_sha256": "d" * 64}),
    ],
)
def test_batch_smoke_gate_rejects_workload_or_fingerprint_mismatch(
    tmp_path: Path,
    execution_overrides: dict[str, object],
    completion_overrides: dict[str, object],
    protocol_overrides: dict[str, object],
) -> None:
    """Reject incomplete batch execution and unbound protocol fingerprints."""

    result = _run_batch_smoke_gate(
        tmp_path,
        mode="score_only",
        execution_overrides=execution_overrides,
        completion_overrides=completion_overrides,
        protocol_overrides=protocol_overrides,
    )
    assert result.returncode != 0
    assert "smoke artifact mismatch" in result.stderr


def test_full_vdev_direction_retains_the_vanilla_batch_path() -> None:
    """Keep full mode free of the retired chunk, padding, mask, or scan path."""

    scorer = SCORER.read_text(encoding="utf-8")
    config = CONFIG.read_text(encoding="utf-8")
    assert "prepare_validation_batch" in scorer
    assert "prepare_stratified_validation_batch_cycle" in scorer
    assert "jax.lax.pmean(local_gradient" in scorer
    assert "salda_ga_validation_direction_mode: full" in config
    assert "salda_ga_validation_batch_size: 500" in config
    assert "salda_ga_validation_reanchor_interval: 50" in config
    for retired in (
        "validation_chunk_size",
        "global_chunk_size",
        "valid_mask",
        "jax.lax.scan",
    ):
        assert retired not in scorer
        assert retired not in config
