#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

LEGACY_STAGE="${1:-}"
LEGACY_ARM="${2:-}"
if [[ -n "$LEGACY_STAGE" || -n "$LEGACY_ARM" ]]; then
  if [[ "$LEGACY_ARM" != "last_score" ]]; then
    echo "The compatibility entry point supports only arm=last_score." >&2
    exit 2
  fi
  case "$LEGACY_STAGE" in
    smoke)
      LEGACY_SOURCE_CONFIG=configs/stl10/preact_resnet18/salda_ga_smoke.yaml
      ;;
    timing)
      LEGACY_SOURCE_CONFIG=configs/stl10/preact_resnet18/salda_ga_timing10.yaml
      ;;
    *)
      echo "The compatibility entry point supports smoke or timing." >&2
      exit 2
      ;;
  esac
  if [[ -n "${SOURCE_CONFIG:-}" && "$SOURCE_CONFIG" != "$LEGACY_SOURCE_CONFIG" ]]; then
    echo "Positional stage and SOURCE_CONFIG select different workloads." >&2
    exit 2
  fi
  SOURCE_CONFIG="$LEGACY_SOURCE_CONFIG"
fi

SOURCE_CONFIG="${SOURCE_CONFIG:-configs/stl10/preact_resnet18/salda_ga_timing20.yaml}"
if [[ "${ALLTHEMIX_SALDA_CONFIG_ROUTE_ONLY:-false}" == "true" ]]; then
  printf '%s\n' "$SOURCE_CONFIG"
  exit 0
fi

source scripts/environment/use_backend.sh jax --runtime

: "${RUN_DIR:?set RUN_DIR to a new timing output directory}"
: "${DIRECTION_SMOKE_ARTIFACT:?set the reviewed direction artifact path}"
: "${DIRECTION_SMOKE_ARTIFACT_SHA256:?set the direction artifact file SHA}"
SMOKE_COMPLETION="${SMOKE_COMPLETION:-}"
SMOKE_COMPLETION_SHA256="${SMOKE_COMPLETION_SHA256:-}"
WRAPPER_STARTED="$(date +%s.%N)"

DATA_DIR="${DATA_DIR:-/mnt/disks/allthemix/data}"
HOST_TAG="${HOSTNAME%%.*}"
TPU_LOCK_FILE="${ALLTHEMIX_TPU_LOCK_FILE:-/mnt/disks/allthemix/locks/tpu-${HOST_TAG}.lock}"
GIT_COMMIT="$(git rev-parse HEAD)"
SHORT_COMMIT="${GIT_COMMIT:0:12}"
RESOLVED_CONFIG="${RUN_DIR}/resolved_run_config.yaml"
CACHE_DIR="${RUN_DIR}/jax_cache"
CONFIG_STEM="$(basename "$SOURCE_CONFIG" .yaml)"
RUN_NAME="stl10_${CONFIG_STEM}_last_score_seed0_${SHORT_COMMIT}"

case "$SOURCE_CONFIG" in
  configs/stl10/preact_resnet18/salda_ga_smoke.yaml|\
  configs/stl10/preact_resnet18/salda_ga_timing10.yaml|\
  configs/stl10/preact_resnet18/salda_ga_timing20.yaml)
    ;;
  *)
    echo "SOURCE_CONFIG must be one of the registered STL-10 GA configs." >&2
    exit 2
    ;;
esac

if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "STL-10 timing requires a clean committed checkout." >&2
  exit 2
fi
if [[ -e "$RUN_DIR" ]]; then
  echo "RUN_DIR must not exist: $RUN_DIR" >&2
  exit 2
fi
if [[ -e "$CACHE_DIR" ]]; then
  echo "The timing compilation cache must start absent: $CACHE_DIR" >&2
  exit 2
fi
if [[ ! -f "$SOURCE_CONFIG" ]]; then
  echo "Tracked timing config is missing: $SOURCE_CONFIG" >&2
  exit 2
fi
if [[ ! -f "$DIRECTION_SMOKE_ARTIFACT" ]]; then
  echo "The direction prerequisite must exist." >&2
  exit 2
fi
if [[ ! "$DIRECTION_SMOKE_ARTIFACT_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "The direction file SHA must be an exact lowercase SHA-256 value." >&2
  exit 2
fi
if ! command -v flock >/dev/null 2>&1; then
  echo "Atomic TPU locking requires flock." >&2
  exit 2
fi

mkdir -p "$(dirname "$TPU_LOCK_FILE")"
exec {TPU_LOCK_FD}>"$TPU_LOCK_FILE"
if ! flock -n "$TPU_LOCK_FD"; then
  echo "TPU lock is held at $TPU_LOCK_FILE" >&2
  exit 2
fi
cleanup_lock() {
  flock -u "$TPU_LOCK_FD" 2>/dev/null || true
  exec {TPU_LOCK_FD}>&-
}
RUN_STATUS="FAILED"
finish_run() {
  local process_status="$?"
  if [[ -d "$RUN_DIR" ]]; then
    printf '%s\n' "$process_status" >"$RUN_DIR/.wrapper_exit_code.tmp"
    mv "$RUN_DIR/.wrapper_exit_code.tmp" "$RUN_DIR/wrapper_exit_code.txt"
    printf '%s\n' "$RUN_STATUS" >"$RUN_DIR/.status.tmp"
    mv "$RUN_DIR/.status.tmp" "$RUN_DIR/status.txt"
    date -u +%Y-%m-%dT%H:%M:%SZ >"$RUN_DIR/.ended_at.tmp"
    mv "$RUN_DIR/.ended_at.tmp" "$RUN_DIR/ended_at.txt"
  fi
  cleanup_lock
}
trap finish_run EXIT

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/wandb" "$CACHE_DIR"
cp "$SOURCE_CONFIG" "$RUN_DIR/source_config.yaml"

python - "$SOURCE_CONFIG" "$RESOLVED_CONFIG" "$DATA_DIR" "$RUN_DIR" \
  "$RUN_NAME" "$GIT_COMMIT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import yaml

from allthemix.config import load_yaml_config

source, destination, data_dir, run_dir, run_name, git_commit = sys.argv[1:]
config = load_yaml_config(Path(source))
required = {
    "dataset": "stl10",
    "model": "preact_resnet18",
    "method": "mixup",
    "batch_size": 128,
    "epochs": 200,
    "max_eval_steps": -1,
    "seed": 0,
    "data_seed": -1,
    "resnet_stem_type": "cifar",
    "preact_stem_bn_relu": False,
    "preact_pytorch_default_init": False,
    "validation_split": 0.5,
    "val_source": "test",
    "eval_on_test_each_epoch": False,
    "val_select_split_fraction": 0.0,
    "final_test": False,
    "save_checkpoint": False,
    "save_best_only": False,
    "distributed": True,
    "sync_batch_stats": True,
    "cross_device_shuffle": False,
    "basic_aug": True,
    "aug_recipe": "basic",
    "shuffle_buffer_size": 10_000,
    "deterministic_data": True,
    "strict_determinism": False,
    "train_subset_fraction": 1.0,
    "debug_train_source": "none",
    "early_stop_enabled": False,
    "save_csv": True,
    "log_time": True,
    "wandb": True,
    "wandb_project": "allthemix-probes",
    "wandb_mode": "online",
    "learning_rate": 0.1,
    "momentum": 0.9,
    "nesterov": False,
    "weight_decay": 0.0005,
    "lr_schedule": "cosine",
    "min_learning_rate": 0.0,
    "warmup_epochs": 0,
    "lr_decay_epochs": [100, 150],
    "lr_decay_rate": 0.1,
    "mixup_alpha": 1.0,
    "salda_ga_mode": "score_only",
    "salda_ga_parameter_scope": "classifier_head",
    "salda_ga_validation_direction_mode": "full",
    "salda_ga_validation_batch_size": 400,
    "salda_ga_validation_reanchor_interval": 50,
    "salda_ga_maximum_rows": 128,
    "salda_ga_soft_label_dose": 0.01,
    "salda_ga_max_weight_deviation": 0.05,
    "salda_ga_weight_temperature": 1.0,
    "salda_ga_minimum_relative_ess": 0.9,
    "salda_ga_minimum_gain": 0.0,
    "salda_ga_minimum_label_margin": 0.0,
    "salda_ga_minimum_relative_label_margin": 0.0,
    "salda_ga_fallback_enabled": False,
    "salda_ga_fallback_soft_label_dose": 0.01,
}
mismatches = {
    key: {"observed": config.get(key), "expected": expected}
    for key, expected in required.items()
    if config.get(key) != expected
}
if mismatches:
    raise SystemExit("timing source config mismatch: " + json.dumps(mismatches))
workloads = {
    1: {
        "max_train_steps": 1,
        "salda_ga_audit_mode": True,
        "salda_ga_profile_components": True,
    },
    20: {
        "max_train_steps": -1,
        "salda_ga_audit_mode": False,
        "salda_ga_profile_components": False,
    },
}
workloads[10] = dict(workloads[20])
stop_epoch = config.get("salda_ga_stop_epoch")
if stop_epoch not in workloads:
    raise SystemExit("config must select the registered smoke or timing workload")
workload_mismatches = {
    key: {"observed": config.get(key), "expected": expected}
    for key, expected in workloads[stop_epoch].items()
    if config.get(key) != expected
}
if workload_mismatches:
    raise SystemExit(
        "config workload mismatch: " + json.dumps(workload_mismatches)
    )

# Only operational paths, unique run naming, and exact source provenance vary.
injected = {
    "data_dir": data_dir,
    "output_dir": run_dir,
    "output_name": "metrics.csv",
    "run_name": run_name,
    "wandb_run_name": run_name,
    "checkpoint_dir": str(Path(run_dir) / "checkpoints"),
    "salda_ga_git_commit": git_commit,
}
config.update(injected)
Path(destination).write_text(
    yaml.safe_dump(config, sort_keys=True),
    encoding="utf-8",
)
print(hashlib.sha256(Path(source).read_bytes()).hexdigest())
print(hashlib.sha256(Path(destination).read_bytes()).hexdigest())
PY

STOP_EPOCH="$(python - "$RESOLVED_CONFIG" <<'PY'
import sys
from pathlib import Path
from allthemix.config import load_yaml_config
print(load_yaml_config(Path(sys.argv[1]))["salda_ga_stop_epoch"])
PY
)"

if [[ "$STOP_EPOCH" == "10" || "$STOP_EPOCH" == "20" ]]; then
  if [[ ! -f "$SMOKE_COMPLETION" ]]; then
    echo "A timing run requires the reviewed smoke completion." >&2
    exit 2
  fi
  if [[ ! "$SMOKE_COMPLETION_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
    echo "The smoke file SHA must be an exact lowercase SHA-256 value." >&2
    exit 2
  fi
  python - "$DIRECTION_SMOKE_ARTIFACT" "$DIRECTION_SMOKE_ARTIFACT_SHA256" \
    "$SMOKE_COMPLETION" "$SMOKE_COMPLETION_SHA256" "$RESOLVED_CONFIG" \
    "$GIT_COMMIT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from allthemix.config import load_yaml_config
from salutary_da.protocol import (
    build_data_protocol,
    build_run_protocol_data_fields,
    build_runtime_config,
    build_training_recipe,
    canonical_protocol_sha256,
    get_instantaneous_ga_dataset_protocol,
)

direction_path, direction_sha, smoke_path, smoke_sha, config_path, commit = sys.argv[1:]

def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def self_hash(value, field):
    unhashed = dict(value)
    declared = unhashed.pop(field, None)
    data = json.dumps(
        unhashed,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    observed = hashlib.sha256(data).hexdigest()
    if declared != observed:
        raise SystemExit(f"{field} mismatch")

if file_sha(direction_path) != direction_sha:
    raise SystemExit("direction prerequisite file SHA mismatch")
if file_sha(smoke_path) != smoke_sha:
    raise SystemExit("smoke prerequisite file SHA mismatch")
direction = json.loads(Path(direction_path).read_text(encoding="utf-8"))
smoke = json.loads(Path(smoke_path).read_text(encoding="utf-8"))
self_hash(direction, "payload_sha256")
self_hash(smoke, "completion_sha256")
protocol_path = Path(smoke["protocol_artifact"])
if not protocol_path.is_file():
    raise SystemExit("smoke protocol artifact is missing")
if file_sha(protocol_path) != smoke["protocol_artifact_sha256"]:
    raise SystemExit("smoke protocol file SHA mismatch")
protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
config = load_yaml_config(Path(config_path))
dataset_protocol = get_instantaneous_ga_dataset_protocol(config["dataset"])
direction_binding = smoke.get("direction_prerequisite", {})
required_direction = {
    "status": "SUCCESS",
    "dataset": "stl10",
    "git_commit": commit,
    "backend": "tpu",
    "device_count": 4,
    "validation_examples": 4_000,
    "validation_class_counts": [400] * 10,
    "local_validation_examples": 1_000,
    "validation_direction_mode": "full",
    "validation_direction_layout": "single_complete_batch",
    "parameter_scope": "classifier_head",
    "direction_leaf_shapes": [[4, 10], [4, 512, 10]],
    "finite": True,
    "distributed": True,
    "sync_batch_stats": True,
    "main_table_eligible": False,
}
required_smoke = {
    "status": "SUCCESS",
    "dataset": "stl10",
    "git_commit": commit,
    "completed_epochs": 1,
    "optimizer_horizon_epochs": 200,
    "policy_mode": "score_only",
    "seed": 0,
    "resolved_data_seed": 0,
    "method": "mixup",
    "parameter_scope": "classifier_head",
    "validation_direction_mode": "full",
    "validation_examples_per_gradient_evaluation": 4_000,
    "train_updates": 1,
    "vdev_evaluations": 1,
    "vdev_batches": 32,
    "vdev_example_visits_for_epoch_readout": 4_000,
    "vtest_loaded": False,
    "vtest_batches": 0,
    "vtest_examples": 0,
    "endpoint_builder_calls": 0,
    "endpoint_evaluations": 0,
    "initial_optimizer_step": 0,
    "terminal_optimizer_step": 1,
}
errors = [
    f"direction.{key}"
    for key, expected in required_direction.items()
    if direction.get(key) != expected
]
errors.extend(
    f"smoke.{key}"
    for key, expected in required_smoke.items()
    if smoke.get(key) != expected
)
if direction_binding.get("artifact_file_sha256") != direction_sha:
    errors.append("smoke.direction_prerequisite.artifact_file_sha256")
if direction_binding.get("payload_sha256") != direction.get("payload_sha256"):
    errors.append("smoke.direction_prerequisite.payload_sha256")
if direction_binding.get("validation_pool_sha256") != direction.get(
    "validation_pool_sha256"
):
    errors.append("smoke.direction_prerequisite.validation_pool_sha256")
if smoke.get("validation_pool_sha256") != direction.get("validation_pool_sha256"):
    errors.append("smoke.validation_pool_sha256")
if protocol.get("direction_prerequisite") != direction_binding:
    errors.append("protocol.direction_prerequisite")
runtime_config = build_runtime_config(
    config,
    method_name=config["method"],
    protocol=dataset_protocol,
)
training_recipe = build_training_recipe(runtime_config)
data_protocol = build_data_protocol(
    method_name=config["method"],
    validation_fingerprint=direction["validation_pool_sha256"],
    protocol=dataset_protocol,
)
expected_hashes = {
    "runtime_config_sha256": canonical_protocol_sha256(runtime_config),
    "training_recipe_sha256": canonical_protocol_sha256(training_recipe),
    "data_protocol_sha256": canonical_protocol_sha256(data_protocol),
}
for key, expected in expected_hashes.items():
    if protocol.get(key) != expected:
        errors.append(f"protocol.{key}")
    if smoke.get(key) != expected:
        errors.append(f"smoke.{key}")
if protocol.get("runtime_config") != runtime_config:
    errors.append("protocol.runtime_config")
if protocol.get("training_recipe") != training_recipe:
    errors.append("protocol.training_recipe")
for key, expected in build_run_protocol_data_fields(data_protocol).items():
    if protocol.get(key) != expected:
        errors.append(f"protocol.data_protocol.{key}")
runtime = protocol.get("runtime_config", {})
ga = runtime.get("gradient_alignment", {})
runtime_required = {
    "dataset": config["dataset"],
    "model": config["model"],
    "base_method": config["method"],
    "global_batch_size": config["batch_size"],
    "optimizer_horizon_epochs": config["epochs"],
    "distributed": config["distributed"],
    "sync_batch_stats": config["sync_batch_stats"],
}
errors.extend(
    f"smoke.runtime_config.{key}"
    for key, expected in runtime_required.items()
    if runtime.get(key) != expected
)
ga_required = {
    "parameter_scope": config["salda_ga_parameter_scope"],
    "validation_direction_mode": config["salda_ga_validation_direction_mode"],
    "validation_examples_per_gradient_evaluation": 4_000,
    "policy_mode": config["salda_ga_mode"],
}
errors.extend(
    f"smoke.runtime_config.gradient_alignment.{key}"
    for key, expected in ga_required.items()
    if ga.get(key) != expected
)
execution = smoke.get("execution", {})
for key, expected in {
    "action_enabled": True,
    "train_steps": 1,
    "direction_refreshes": 1,
    "validation_gradient_evaluations": 1,
    "validation_exact_reanchors": 0,
    "direction_validation_example_visits": 4_000,
    "validation_pool_examples": 4_000,
}.items():
    if execution.get(key) != expected:
        errors.append(f"smoke.execution.{key}")
actions = smoke.get("action_summary", {})
for key, expected in {
    "scored_batches": 1,
    "scored_rows": 128,
    "applied_rows": 0,
    "batches_with_actions": 0,
    "fallback_batches": 0,
    "invalid_score_rows": 0,
}.items():
    if actions.get(key) != expected:
        errors.append(f"smoke.action_summary.{key}")
if smoke.get("best_vdev_checkpoint") is not None:
    errors.append("smoke.best_vdev_checkpoint")
wandb = smoke.get("wandb", {})
if not (
    wandb.get("enabled")
    and wandb.get("mode") == "online"
    and wandb.get("run_id")
    and wandb.get("url")
    and wandb.get("finish_completed")
):
    errors.append("smoke.wandb")
if errors:
    raise SystemExit("timing prerequisite mismatch: " + ", ".join(errors))
PY
fi

export JAX_COMPILATION_CACHE_DIR="$CACHE_DIR"
export WANDB_DIR="$RUN_DIR/wandb"
export ALLTHEMIX_SALDA_DIRECTION_ARTIFACT="$DIRECTION_SMOKE_ARTIFACT"
export ALLTHEMIX_SALDA_DIRECTION_ARTIFACT_SHA256="$DIRECTION_SMOKE_ARTIFACT_SHA256"

python - "$RUN_DIR/runtime_environment.json" "$CACHE_DIR" "$RUN_DIR/wandb" \
  "$TPU_LOCK_FILE" "$DATA_DIR" "$SOURCE_CONFIG" "$RESOLVED_CONFIG" <<'PY'
import json
import sys
from pathlib import Path

(
    output,
    cache_dir,
    wandb_dir,
    lock_file,
    data_dir,
    source_config,
    resolved_config,
) = sys.argv[1:]
payload = {
    "jax_compilation_cache_dir": str(Path(cache_dir).resolve()),
    "wandb_dir": str(Path(wandb_dir).resolve()),
    "tpu_lock_file": str(Path(lock_file).resolve()),
    "data_dir": str(Path(data_dir).resolve()),
    "source_config": str(Path(source_config).resolve()),
    "resolved_config": str(Path(resolved_config).resolve()),
}
Path(output).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

printf '%s\n' "$GIT_COMMIT" >"$RUN_DIR/commit.txt"
printf '%s\n' "$SOURCE_CONFIG" >"$RUN_DIR/source_config_path.txt"
printf '%s  %s\n' "$DIRECTION_SMOKE_ARTIFACT_SHA256" \
  "$DIRECTION_SMOKE_ARTIFACT" >"$RUN_DIR/prerequisites.sha256"
if [[ "$STOP_EPOCH" == "10" || "$STOP_EPOCH" == "20" ]]; then
  printf '%s  %s\n' "$SMOKE_COMPLETION_SHA256" "$SMOKE_COMPLETION" \
    >>"$RUN_DIR/prerequisites.sha256"
fi
sha256sum "$RUN_DIR/source_config.yaml" "$RESOLVED_CONFIG" \
  >"$RUN_DIR/config_files.sha256"
printf '%q ' python -u -m allthemix.cli.train --config "$RESOLVED_CONFIG" \
  >"$RUN_DIR/command.txt"
printf '\n' >>"$RUN_DIR/command.txt"
date -u +%Y-%m-%dT%H:%M:%SZ >"$RUN_DIR/started_at.txt"

set +e
python -u -m allthemix.cli.train --config "$RESOLVED_CONFIG" \
  2>&1 | tee "$RUN_DIR/logs/train.log"
PIPELINE_CODES=("${PIPESTATUS[@]}")
set -e
TRAINER_CODE="${PIPELINE_CODES[0]}"
TEE_CODE="${PIPELINE_CODES[1]}"
printf '%s\n' "$TRAINER_CODE" >"$RUN_DIR/trainer_exit_code.txt"
printf '%s\n' "$TEE_CODE" >"$RUN_DIR/tee_exit_code.txt"
if [[ "$TRAINER_CODE" -ne 0 || "$TEE_CODE" -ne 0 ]]; then
  printf 'FAILED\n' >"$RUN_DIR/status.txt"
  exit 2
fi

COMPLETION="$RUN_DIR/${RUN_NAME}_salda_training_complete.json"
python - "$COMPLETION" "$RESOLVED_CONFIG" "$DIRECTION_SMOKE_ARTIFACT_SHA256" \
  "$SMOKE_COMPLETION" "$SMOKE_COMPLETION_SHA256" "$WRAPPER_STARTED" \
  "$RUN_DIR/postflight.json" \
  <<'PY'
import hashlib
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from allthemix.config import load_yaml_config
from salutary_da.protocol import get_instantaneous_ga_dataset_protocol

(
    completion_path,
    config_path,
    direction_sha,
    smoke_path,
    smoke_sha,
    started,
    output,
) = sys.argv[1:]
value = json.loads(Path(completion_path).read_text(encoding="utf-8"))
config = load_yaml_config(Path(config_path))
protocol = get_instantaneous_ga_dataset_protocol(config["dataset"])
epochs = config["salda_ga_stop_epoch"]
updates_per_epoch = (
    config["max_train_steps"]
    if config["max_train_steps"] > 0
    else protocol.steps_per_epoch
)
updates = epochs * updates_per_epoch
vdev_batches = epochs * protocol.validation_batches_per_epoch
readout_visits = epochs * protocol.validation_examples
direction_visits = updates * protocol.validation_examples
scored_rows = updates * protocol.global_batch_size
errors = []

unhashed = dict(value)
declared_hash = unhashed.pop("completion_sha256", None)
observed_hash = hashlib.sha256(
    json.dumps(
        unhashed,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()
if declared_hash != observed_hash:
    errors.append("completion_sha256")
required = {
    "status": "SUCCESS",
    "dataset": protocol.dataset,
    "git_commit": config["salda_ga_git_commit"],
    "completed_epochs": epochs,
    "optimizer_horizon_epochs": config["epochs"],
    "policy_mode": config["salda_ga_mode"],
    "seed": config["seed"],
    "resolved_data_seed": config["seed"],
    "method": config["method"],
    "parameter_scope": config["salda_ga_parameter_scope"],
    "validation_direction_mode": config["salda_ga_validation_direction_mode"],
    "validation_examples_per_gradient_evaluation": protocol.validation_examples,
    "train_updates": updates,
    "vdev_evaluations": epochs,
    "vdev_batches": vdev_batches,
    "vdev_example_visits_for_epoch_readout": readout_visits,
    "vtest_loaded": False,
    "vtest_batches": 0,
    "vtest_examples": 0,
    "vtest_result": None,
    "endpoint_builder_calls": 0,
    "endpoint_evaluations": 0,
    "initial_optimizer_step": 0,
    "terminal_optimizer_step": updates,
    "best_vdev_checkpoint": None,
}
errors.extend(
    key for key, expected in required.items() if value.get(key) != expected
)
protocol_path = Path(value.get("protocol_artifact", ""))
if not protocol_path.is_file():
    errors.append("protocol_artifact")
    run_protocol = {}
elif hashlib.sha256(protocol_path.read_bytes()).hexdigest() != value.get(
    "protocol_artifact_sha256"
):
    errors.append("protocol_artifact_sha256")
    run_protocol = {}
else:
    run_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
for key in (
    "runtime_config_sha256",
    "training_recipe_sha256",
    "data_protocol_sha256",
):
    if run_protocol.get(key) != value.get(key):
        errors.append(f"protocol.{key}")
if epochs in (10, 20):
    smoke = json.loads(Path(smoke_path).read_text(encoding="utf-8"))
    for key in (
        "runtime_config_sha256",
        "training_recipe_sha256",
        "data_protocol_sha256",
        "validation_pool_sha256",
    ):
        if value.get(key) != smoke.get(key):
            errors.append(f"smoke_binding.{key}")
workload = value.get("workload_closure", {})
if epochs in (10, 20):
    if not (
        workload.get("workload")
        == ("ten_epoch_timing" if epochs == 10 else "bounded_epoch_timing")
        and workload.get("required") is True
        and workload.get("passed") is True
        and workload.get("observed") == workload.get("expected")
    ):
        errors.append("workload_closure")
elif not (
    workload.get("workload") == "unregistered_short_run"
    and workload.get("required") is False
    and workload.get("passed") is True
):
    errors.append("workload_closure")
pre_endpoint = value.get("pre_endpoint_workload_closure", {})
for key, expected in {
    "passed": True,
    "registered": epochs in (10, 20),
    "completed_epochs": epochs,
    "train_batches_per_epoch": updates_per_epoch,
    "train_updates": updates,
    "vdev_evaluations": epochs,
    "vdev_batches": vdev_batches,
    "vdev_examples": readout_visits,
    "initial_optimizer_step": 0,
    "terminal_optimizer_step": updates,
    "endpoint_builder_calls_before_closure": 0,
    "endpoint_evaluations_before_closure": 0,
}.items():
    if pre_endpoint.get(key) != expected:
        errors.append(f"pre_endpoint_workload_closure.{key}")
execution = value.get("execution", {})
for key, expected in {
    "action_enabled": True,
    "parameter_scope": "classifier_head",
    "validation_direction_mode": "full",
    "validation_pool_examples": protocol.validation_examples,
    "validation_examples_per_gradient_evaluation": protocol.validation_examples,
    "train_steps": updates,
    "direction_refreshes": updates,
    "validation_gradient_evaluations": updates,
    "validation_exact_reanchors": 0,
    "validation_anchor_drift_comparisons": 0,
    "direction_validation_example_visits": direction_visits,
}.items():
    if execution.get(key) != expected:
        errors.append(f"execution.{key}")
actions = value.get("action_summary", {})
for key, expected in {
    "scored_batches": updates,
    "scored_rows": scored_rows,
    "applied_rows": 0,
    "batches_with_actions": 0,
    "fallback_batches": 0,
    "invalid_score_rows": 0,
    "mean_dose_over_all_scored_rows": 0.0,
}.items():
    if actions.get(key) != expected:
        errors.append(f"action_summary.{key}")
timing = value.get("epoch_timing_summary", {})
rows = value.get("epochs", [])
wall = np.asarray(
    [row["component_timing_seconds"]["end_to_end_wall"] for row in rows],
    dtype=np.float64,
)
if len(rows) != epochs or [row.get("epoch") for row in rows] != list(
    range(1, epochs + 1)
):
    errors.append("epochs")
stable = wall[1:]
summary_wall = timing.get("components", {}).get("end_to_end_wall", {})
recomputed = (
    {
        "mean": float(np.mean(stable)),
        "median": float(np.median(stable)),
        "p90": float(np.quantile(stable, 0.9)),
        "count": int(stable.size),
    }
    if stable.size
    else {"mean": None, "median": None, "p90": None, "count": 0}
)
if timing.get("compile_epoch_1_seconds") != float(wall[0]):
    errors.append("epoch_timing_summary.compile_epoch_1_seconds")
if timing.get("stable_epoch_range") != list(range(2, epochs + 1)):
    errors.append("epoch_timing_summary.stable_epoch_range")
for key, expected in recomputed.items():
    observed = summary_wall.get(key)
    if isinstance(expected, float):
        if observed is None or not math.isclose(observed, expected, rel_tol=1e-12):
            errors.append(f"epoch_timing_summary.end_to_end_wall.{key}")
    elif observed != expected:
        errors.append(f"epoch_timing_summary.end_to_end_wall.{key}")
csv_path = Path(config["output_dir"]) / config["output_name"]
if not csv_path.is_file():
    errors.append("metrics_csv")
else:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    if [int(row["epoch"]) for row in csv_rows] != list(range(1, epochs + 1)):
        errors.append("metrics_csv.epochs")
    expected_epoch_updates = updates_per_epoch
    for index, row in enumerate(csv_rows, start=1):
        if int(row["salda_train_updates"]) != expected_epoch_updates:
            errors.append(f"metrics_csv.{index}.salda_train_updates")
        if int(row["salda_vdev_batches"]) != protocol.validation_batches_per_epoch:
            errors.append(f"metrics_csv.{index}.salda_vdev_batches")
        if int(row["salda_vtest_batches"]) != 0:
            errors.append(f"metrics_csv.{index}.salda_vtest_batches")
if Path(config["checkpoint_dir"]).exists():
    errors.append("checkpoint_dir")
if list(Path(config["output_dir"]).glob("*_final_test.csv")):
    errors.append("final_test_csv")
target = value.get("timing_target", {})
if not (
    target.get("registered") is False
    and target.get("passed") is None
    and target.get("reason") == "dataset_timing_target_not_registered"
):
    errors.append("timing_target")
wandb = value.get("wandb", {})
if not (
    wandb.get("enabled")
    and wandb.get("mode") == "online"
    and wandb.get("run_id")
    and wandb.get("url")
    and wandb.get("finish_completed")
):
    errors.append("wandb")
if errors:
    raise SystemExit("timing postflight mismatch: " + ", ".join(errors))
result = {
    "status": "SUCCESS",
    "completion": str(Path(completion_path).resolve()),
    "completion_file_sha256": hashlib.sha256(
        Path(completion_path).read_bytes()
    ).hexdigest(),
    "direction_prerequisite_file_sha256": direction_sha,
    "smoke_prerequisite_file_sha256": smoke_sha or None,
    "tracked_source_config_file_sha256": hashlib.sha256(
        Path(config_path).parent.joinpath("source_config.yaml").read_bytes()
    ).hexdigest(),
    "expanded_run_config_file_sha256": hashlib.sha256(
        Path(config_path).read_bytes()
    ).hexdigest(),
    "runtime_environment_file_sha256": hashlib.sha256(
        Path(config_path).parent.joinpath("runtime_environment.json").read_bytes()
    ).hexdigest(),
    "trainer_resolved_config_sha256": value["resolved_config_sha256"],
    "runtime_config_sha256": value["runtime_config_sha256"],
    "training_recipe_sha256": value["training_recipe_sha256"],
    "data_protocol_sha256": value["data_protocol_sha256"],
    "completed_epochs": epochs,
    "train_updates": updates,
    "vdev_batches": vdev_batches,
    "vdev_readout_example_visits": readout_visits,
    "direction_evaluations": updates,
    "direction_example_visits": direction_visits,
    "scored_rows": scored_rows,
    "epoch_1_loop_wall_seconds": float(wall[0]),
    "stable_epoch_range": list(range(2, epochs + 1)),
    "stable_epoch_wall_seconds": stable.tolist(),
    "stable_epoch_wall_summary": recomputed,
    "wrapper_wall_seconds": time.time() - float(started),
    "wandb": wandb,
}
Path(output).write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

printf '%s\n' "$COMPLETION" >"$RUN_DIR/completion_path.txt"
RUN_STATUS="SUCCESS"
printf '%s\n' "$RUN_DIR/postflight.json"
