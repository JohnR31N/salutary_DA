#!/bin/bash
# CIFAR-100 GA production arms on the boot-overlap pools.
# usage: run_vote_opt.sh <arm> [...]   arms: b0 d1024 full chunks50 hvoter
#                                            lastga hlast
set -euo pipefail

readonly ROOT=/mnt/disks/allthemix
readonly REPO=/home/bluesoulreborn_gmail_com/AllTheMix-valbatch-sweep
readonly PY="$ROOT/venvs/allthemix-jax/bin/python"
readonly SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
readonly TRAINER="$SCRIPTS/train_cifar100_ga_dual_guidance_v12.py"
readonly OUT="$ROOT/outputs/c100_ga_vote_study_v1"
readonly VMASK="$OUT/boot_vdev_mask_seed0.npz"
readonly TMASK="$OUT/boot_vtest_mask_seed0.npz"
readonly LOCK="$ROOT/locks/tpu-vote-study.lock"

mkdir -p "$OUT" "$ROOT/locks" "$ROOT/xla/jax_c100ga"
cd "$REPO"
export PYTHONPATH=.
export JAX_COMPILATION_CACHE_DIR="$ROOT/xla/jax_c100ga"

exec 9>"$LOCK"
if ! flock -n 9; then echo "LOCK_BUSY"; exit 97; fi

test -s "$VMASK"
test -s "$TMASK"
sha256sum "$TRAINER" "$VMASK" "$TMASK" >> "$OUT/input_hashes.sha256"

arm_flags () {
  case "$1" in
    b0) echo "--budget_frac 0.0 --direction_batch 1024" ;;
    d1024) echo "--budget_frac 0.10 --direction_batch 1024" ;;
    full) echo "--budget_frac 0.10 --direction_batch 5000" ;;
    chunks50) echo "--budget_frac 0.10 --direction_batch 5000 --direction_vote_chunks 50 --vote_mode rank" ;;
    hvoter) echo "--budget_frac 0.10 --direction_batch 5000 --head_vote rank" ;;
    lastga) echo "--budget_frac 0.10 --parameter_scope head" ;;
    hlast) echo "--budget_frac 0.10 --parameter_scope head --head_vote rank" ;;
    *) echo "UNKNOWN" ;;
  esac
}

for name in "$@"; do
  flags=$(arm_flags "$name")
  if [[ "$flags" == UNKNOWN ]]; then echo "BAD_ARM $name"; exit 96; fi
  result="$OUT/${name}.json"
  if [[ -f "$result" ]] && grep -q '"status": "SUCCESS"' "$result"; then
    echo "ARM_SKIP $name"
    continue
  fi
  echo "ARM_START $name [$flags] $(date -u +%H:%M:%S)"
  # shellcheck disable=SC2086
  "$PY" -u "$TRAINER" \
    --data_dir "$ROOT/data" \
    --dataset cifar100 \
    --seed 0 \
    --split_mask "$VMASK" \
    --sealed_mask "$TMASK" \
    --per_row_stats \
    --track_sealed_each_epoch \
    $flags \
    --out "$result" \
    > "$OUT/${name}.log" 2>&1
  echo "ARM_DONE $name $(date -u +%H:%M:%S)"
done

sha256sum "$OUT"/*.json >> "$OUT/result_hashes.sha256" 2>/dev/null || true
echo "VOTE_STUDY_OK $(date -u +%H:%M:%S)"
