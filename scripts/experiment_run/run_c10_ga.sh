#!/bin/bash
# CIFAR-10 boot-overlap GA arms (v12 --dataset cifar10).
# usage: run_c10_ga.sh name:budget:dbatch [...]
set -euo pipefail

readonly ROOT=/mnt/disks/allthemix
readonly REPO=/home/bluesoulreborn_gmail_com/AllTheMix-valbatch-sweep
readonly PY="$ROOT/venvs/allthemix-jax/bin/python"
readonly SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
readonly TRAINER="$SCRIPTS/train_cifar100_ga_dual_guidance_v12.py"
readonly OUT="$ROOT/outputs/c10_ga_vote_study_v1"
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

for spec in "$@"; do
  IFS=':' read -r name budget dbatch chunks vmode hvote lin rot rk aes <<< "$spec"
  dbatch="${dbatch:-1024}"
  extra=()
  if [[ -n "${chunks:-}" && "$chunks" != 0 ]]; then
    extra+=(--direction_vote_chunks "$chunks" --vote_mode "${vmode:-rank}")
  fi
  if [[ -n "${hvote:-}" && "$hvote" != off ]]; then
    extra+=(--head_vote "$hvote")
  fi
  if [[ "${lin:-0}" == 1 ]]; then
    extra+=(--vote_linearize)
  fi
  if [[ -n "${rot:-}" && "$rot" != 0 ]]; then
    extra+=(--vote_rotate "$rot")
  fi
  if [[ -n "${rk:-}" && "$rk" != 0 ]]; then
    extra+=(--direction_refresh_k "$rk")
  fi
  if [[ -n "${aes:-}" && "$aes" != 0 ]]; then
    extra+=(--action_every_steps "$aes")
  fi
  result="$OUT/${name}.json"
  if [[ -f "$result" ]] && grep -q '"status": "SUCCESS"' "$result"; then
    echo "ARM_SKIP $name"
    continue
  fi
  echo "ARM_START $name budget=$budget dbatch=$dbatch $(date -u +%H:%M:%S)"
  "$PY" -u "$TRAINER" \
    --data_dir "$ROOT/data" \
    --dataset cifar10 \
    --parameter_scope full \
    --budget_frac "$budget" \
    --seed 0 \
    --direction_batch "$dbatch" \
    --split_mask "$VMASK" \
    --sealed_mask "$TMASK" \
    --per_row_stats \
    --track_sealed_each_epoch \
    "${extra[@]}" \
    --out "$result" \
    > "$OUT/${name}.log" 2>&1
  echo "ARM_DONE $name $(date -u +%H:%M:%S)"
done

sha256sum "$OUT"/*.json >> "$OUT/result_hashes.sha256" 2>/dev/null || true
echo "C10_GA_OK $(date -u +%H:%M:%S)"
