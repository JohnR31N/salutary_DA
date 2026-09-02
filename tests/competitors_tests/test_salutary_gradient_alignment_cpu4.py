from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_validation_direction_regressions_run_on_four_cpu_devices() -> None:
    """Run the complete-Vdev PMAP checks in an isolated four-device process."""

    required_modules = ("jax", "flax", "optax")
    missing_modules = [
        name for name in required_modules if importlib.util.find_spec(name) is None
    ]
    if missing_modules:
        pytest.skip(f"requires {', '.join(missing_modules)}")

    repository_root = Path(__file__).resolve().parents[2]
    test_path = repository_root / "tests" / "competitors_tests" / (
        "test_salutary_gradient_alignment.py"
    )
    strategy_test_path = repository_root / "tests" / "competitors_tests" / (
        "test_salutary_gradient_alignment_strategy.py"
    )
    test_names = (
        "test_prepare_validation_batch_shards_all_5000_examples_once",
        "test_prepare_validation_batch_shards_all_stl10_4000_examples_once",
        "test_stratified_validation_batch_cycle_matches_external_gradient",
        "test_full_ga_matches_external_validation_gradient_and_training_jvp",
        "test_classifier_head_ga_matches_external_head_and_full_component",
    )
    strategy_test_names = (
        "test_origin_action_warmup_is_exact_standard_update_without_ga",
        "test_four_device_multistep_noop_preserves_recipe_rng_metrics_and_state",
        "test_validation_batch_aggregate_reanchors_and_wraps_from_step_zero",
    )
    command = [
        sys.executable,
        "-m",
        "pytest",
        *(f"{test_path}::{test_name}" for test_name in test_names),
        *(f"{strategy_test_path}::{test_name}" for test_name in strategy_test_names),
        "-q",
    ]
    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu"
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    environment["ALLTHEMIX_CPU4_GA_TEST_CHILD"] = "1"

    result = subprocess.run(
        command,
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )

    assert result.returncode == 0, (
        "isolated four-device GA regression failed\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
