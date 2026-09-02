#!/usr/bin/env bash

set -euo pipefail

STAGE="${1:-smoke}"
ARM="${2:-noop}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

source scripts/environment/use_backend.sh jax --runtime

if ! command -v flock >/dev/null 2>&1; then
  echo "Atomic TPU locking requires flock." >&2
  exit 2
fi

GIT_COMMIT="$(git rev-parse HEAD)"
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "CIFAR-100 instantaneous-GA runs require a clean committed worktree." >&2
  exit 2
fi

HOST_TAG="${HOSTNAME%%.*}"
TPU_LOCK_FILE="${ALLTHEMIX_TPU_LOCK_FILE:-/mnt/disks/allthemix/locks/tpu-${HOST_TAG}.lock}"
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
trap cleanup_lock EXIT
printf 'pid=%s started=%s host=%s repo=%s commit=%s\n' \
  "$$" "$(date -u +%FT%TZ)" "$HOST_TAG" "$ROOT" "$GIT_COMMIT" \
  >&"$TPU_LOCK_FD"

DATA_DIR="${DATA_DIR:-/mnt/disks/allthemix/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/disks/allthemix/outputs}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SEED="${SEED:-0}"
BASE_METHOD="${SALDA_BASE_METHOD:-mixup}"
VALIDATION_DIRECTION_MODE="${SALDA_VALIDATION_DIRECTION_MODE:-full}"
VALIDATION_BATCH_SIZE="${SALDA_VALIDATION_BATCH_SIZE:-500}"
VALIDATION_REANCHOR_INTERVAL="${SALDA_VALIDATION_REANCHOR_INTERVAL:-50}"
if [[ "$SEED" != "0" && "$SEED" != "1" && "$SEED" != "2" ]]; then
  echo "SEED must be exactly 0, 1, or 2." >&2
  exit 2
fi
if [[ "$BASE_METHOD" != "baseline" && "$BASE_METHOD" != "mixup" ]]; then
  echo "SALDA_BASE_METHOD must be baseline or mixup." >&2
  exit 2
fi
if [[ "$VALIDATION_DIRECTION_MODE" != "full" && \
      "$VALIDATION_DIRECTION_MODE" != "batch_aggregate" ]]; then
  echo "SALDA_VALIDATION_DIRECTION_MODE must be full or batch_aggregate." >&2
  exit 2
fi
if [[ "$VALIDATION_BATCH_SIZE" != "500" ]]; then
  echo "SALDA_VALIDATION_BATCH_SIZE must be exactly 500." >&2
  exit 2
fi
if [[ "$VALIDATION_REANCHOR_INTERVAL" != "50" ]]; then
  echo "SALDA_VALIDATION_REANCHOR_INTERVAL must be exactly 50." >&2
  exit 2
fi

MODE=""
SCOPE="classifier_head"
case "$ARM" in
  baseline)
    MODE=baseline
    ;;
  noop)
    MODE=noop
    ;;
  last_score)
    MODE=score_only
    ;;
  last_action)
    MODE="${SALDA_ACTION_MODE:-soft_label}"
    if [[ "$MODE" != "soft_label" && "$MODE" != "reweight" ]]; then
      echo "SALDA_ACTION_MODE must be soft_label or reweight." >&2
      exit 2
    fi
    ;;
  last_shuffled_action)
    MODE="${SALDA_ACTION_MODE:-shuffled_reweight}"
    if [[ "$MODE" != "shuffled_soft_label" && \
          "$MODE" != "shuffled_reweight" ]]; then
      echo "SALDA_ACTION_MODE must be shuffled_soft_label or shuffled_reweight." >&2
      exit 2
    fi
    ;;
  full_score)
    MODE=score_only
    SCOPE=full
    ;;
  *)
    echo "Unknown arm '$ARM'." >&2
    exit 2
    ;;
esac
if [[ "$VALIDATION_DIRECTION_MODE" == "batch_aggregate" && \
      "$SCOPE" != "classifier_head" ]]; then
  echo "batch_aggregate requires the classifier_head scope." >&2
  exit 2
fi

STOP_EPOCH=""
MAX_TRAIN_STEPS=-1
FINAL_TEST=false
SAVE_CHECKPOINT=false
PROFILE_COMPONENTS=false
AUDIT_MODE=false
case "$STAGE" in
  smoke)
    STOP_EPOCH=1
    if [[ "$VALIDATION_DIRECTION_MODE" == "batch_aggregate" ]]; then
      MAX_TRAIN_STEPS=2
    else
      MAX_TRAIN_STEPS=1
    fi
    AUDIT_MODE=true
    PROFILE_COMPONENTS=true
    ;;
  timing)
    STOP_EPOCH=10
    ;;
  profile)
    STOP_EPOCH=2
    PROFILE_COMPONENTS=true
    ;;
  formal)
    STOP_EPOCH=-1
    FINAL_TEST=true
    SAVE_CHECKPOINT=true
    ;;
  *)
    echo "Unknown stage '$STAGE'. Use smoke|timing|profile|formal." >&2
    exit 2
    ;;
esac

if [[ "$STAGE" != "smoke" ]]; then
  SMOKE_COMPLETION="${SMOKE_COMPLETION:-}"
  if [[ -z "$SMOKE_COMPLETION" || ! -f "$SMOKE_COMPLETION" ]]; then
    echo "Set SMOKE_COMPLETION to the matching successful smoke JSON." >&2
    exit 2
  fi
  python - "$SMOKE_COMPLETION" "$GIT_COMMIT" "$MODE" "$SCOPE" \
    "$BASE_METHOD" "$SEED" "$VALIDATION_DIRECTION_MODE" \
    "$VALIDATION_BATCH_SIZE" "$VALIDATION_REANCHOR_INTERVAL" <<'PY'
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
mode = sys.argv[3]
direction_mode = sys.argv[7]
direction_active = mode not in {"baseline", "noop"}
strategy_present = mode != "baseline"
expected_train_updates = 2 if direction_mode == "batch_aggregate" else 1
if not direction_active:
    expected_direction_refreshes = 0
    expected_gradient_evaluations = 0
    expected_exact_reanchors = 0
    expected_validation_visits = 0
elif direction_mode == "batch_aggregate":
    expected_direction_refreshes = 2
    expected_gradient_evaluations = 11
    expected_exact_reanchors = 1
    expected_validation_visits = 5_500
else:
    expected_direction_refreshes = 1
    expected_gradient_evaluations = 1
    expected_exact_reanchors = 0
    expected_validation_visits = 5_000
required = {
    "status": "SUCCESS",
    "git_commit": sys.argv[2],
    "completed_epochs": 1,
    "vtest_loaded": False,
    "vtest_batches": 0,
    "endpoint_builder_calls": 0,
    "policy_mode": sys.argv[3],
    "parameter_scope": sys.argv[4],
    "method": sys.argv[5],
    "seed": int(sys.argv[6]),
    "validation_direction_mode": direction_mode,
    "validation_direction_main_table_eligible": direction_mode == "full",
    "validation_examples_per_gradient_evaluation": (
        5000 if direction_mode == "full" else int(sys.argv[8])
    ),
    "validation_reanchor_interval": (
        None if direction_mode == "full" else int(sys.argv[9])
    ),
    "train_updates": expected_train_updates,
}
mismatches = [key for key, expected in required.items() if value.get(key) != expected]
execution = value.get("execution")
if not isinstance(execution, dict):
    raise SystemExit("smoke artifact execution must be an object")
execution_required = {
    "train_steps": expected_train_updates,
    "direction_refreshes": expected_direction_refreshes,
    "validation_gradient_evaluations": expected_gradient_evaluations,
    "validation_exact_reanchors": expected_exact_reanchors,
    "direction_validation_example_visits": expected_validation_visits,
    "validation_pool_examples": 5_000,
}
mismatches.extend(
    f"execution.{key}"
    for key, expected in execution_required.items()
    if execution.get(key) != expected
)
protocol_path = Path(value.get("protocol_artifact", ""))
if not protocol_path.is_file():
    raise SystemExit("smoke protocol artifact is missing")
protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
pool_sha = value.get("validation_pool_sha256")
schedule_sha = value.get("validation_batch_schedule_sha256")
if not isinstance(pool_sha, str) or len(pool_sha) != 64:
    mismatches.append("validation_pool_sha256")
if protocol.get("validation_pool_sha256") != pool_sha:
    mismatches.append("protocol.validation_pool_sha256")
if strategy_present:
    if not isinstance(schedule_sha, str) or len(schedule_sha) != 64:
        mismatches.append("validation_batch_schedule_sha256")
    if execution.get("validation_batch_schedule_sha256") != schedule_sha:
        mismatches.append("execution.validation_batch_schedule_sha256")
    if protocol.get("validation_batch_schedule_sha256") != schedule_sha:
        mismatches.append("protocol.validation_batch_schedule_sha256")
elif schedule_sha is not None:
    mismatches.append("validation_batch_schedule_sha256")
if mismatches:
    raise SystemExit(f"smoke artifact mismatch: {mismatches}")
PY
fi

DIRECTION_TAG="$VALIDATION_DIRECTION_MODE"
if [[ "$VALIDATION_DIRECTION_MODE" == "batch_aggregate" ]]; then
  DIRECTION_TAG="${DIRECTION_TAG}${VALIDATION_BATCH_SIZE}"
fi
RUN_NAME="cifar100_salda_${STAGE}_${ARM}_${BASE_METHOD}_${DIRECTION_TAG}_seed${SEED}_${GIT_COMMIT:0:12}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/${RUN_NAME}_${TIMESTAMP}}"
RUN_CHECKPOINT_DIR="${RUN_CHECKPOINT_DIR:-${RUN_DIR}/checkpoints}"
mkdir -p "$RUN_DIR/logs"

POLICY_ARGS=(
  --salda_ga_maximum_rows "${SALDA_MAXIMUM_ROWS:-128}"
  --salda_ga_soft_label_dose "${SALDA_SOFT_LABEL_DOSE:-0.01}"
  --salda_ga_max_weight_deviation "${SALDA_MAX_WEIGHT_DEVIATION:-0.05}"
  --salda_ga_weight_temperature "${SALDA_WEIGHT_TEMPERATURE:-1.0}"
  --salda_ga_minimum_relative_ess "${SALDA_MINIMUM_RELATIVE_ESS:-0.9}"
  --salda_ga_minimum_gain "${SALDA_MINIMUM_GAIN:-0.0}"
  --salda_ga_minimum_label_margin "${SALDA_MINIMUM_LABEL_MARGIN:-0.0}"
  --salda_ga_minimum_relative_label_margin \
    "${SALDA_MINIMUM_RELATIVE_LABEL_MARGIN:-0.0}"
  --salda_ga_fallback_enabled "${SALDA_FALLBACK_ENABLED:-false}"
  --salda_ga_fallback_soft_label_dose \
    "${SALDA_FALLBACK_SOFT_LABEL_DOSE:-0.01}"
)

export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/mnt/disks/allthemix/xla/cifar100_salda}"
export WANDB_DIR="$RUN_DIR"

COMMAND=(
  python -u -m allthemix.cli.train
  --config configs/cifar100/preact_resnet18/salda_ga.yaml
  --method "$BASE_METHOD"
  --seed "$SEED"
  --data_seed -1
  --data_dir "$DATA_DIR"
  --output_dir "$RUN_DIR"
  --output_name metrics.csv
  --run_name "$RUN_NAME"
  --checkpoint_dir "$RUN_CHECKPOINT_DIR"
  --salda_ga_mode "$MODE"
  --salda_ga_parameter_scope "$SCOPE"
  --salda_ga_validation_direction_mode "$VALIDATION_DIRECTION_MODE"
  --salda_ga_validation_batch_size "$VALIDATION_BATCH_SIZE"
  --salda_ga_validation_reanchor_interval "$VALIDATION_REANCHOR_INTERVAL"
  --salda_ga_stop_epoch "$STOP_EPOCH"
  --salda_ga_git_commit "$GIT_COMMIT"
  --salda_ga_audit_mode "$AUDIT_MODE"
  --salda_ga_profile_components "$PROFILE_COMPONENTS"
  --max_train_steps "$MAX_TRAIN_STEPS"
  --final_test "$FINAL_TEST"
  --save_checkpoint "$SAVE_CHECKPOINT"
  --save_best_only true
  --wandb true
  --wandb_project "${WANDB_PROJECT:-allthemix-probes}"
  --wandb_run_name "$RUN_NAME"
  --wandb_mode "${WANDB_MODE:-online}"
  "${POLICY_ARGS[@]}"
)

printf 'stage=%s\narm=%s\nmethod=%s\nvalidation_direction=%s\nrun_dir=%s\ncommit=%s\n' \
  "$STAGE" "$ARM" "$BASE_METHOD" "$DIRECTION_TAG" "$RUN_DIR" \
  "$GIT_COMMIT" \
  | tee "$RUN_DIR/launch.txt"
printf '%q ' "${COMMAND[@]}" >"$RUN_DIR/command.txt"
printf '\n' >>"$RUN_DIR/command.txt"
"${COMMAND[@]}" 2>&1 | tee "$RUN_DIR/logs/run.log"
code=${PIPESTATUS[0]}
if [[ "$code" -eq 0 && "$STAGE" == "smoke" ]]; then
  completion="$RUN_DIR/${RUN_NAME}_salda_training_complete.json"
  if [[ ! -f "$completion" ]]; then
    echo "Smoke completion artifact was not produced." >&2
    exit 2
  fi
fi
printf 'exit_code=%s\nrun_dir=%s\n' "$code" "$RUN_DIR" \
  | tee -a "$RUN_DIR/launch.txt"
exit "$code"
