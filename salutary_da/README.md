# Instantaneous gradient alignment

`salutary_da` contains one training method: every optimizer step obtains a
validation direction from the complete registered Vdev pool,
scores the current original-image or MixUp batch, applies the configured
continuous policy, and performs the update without moving action tensors to
the host. `full` mode uses one vanilla distributed batch of 1,250 examples per
device. `batch_aggregate` mode caches a ten-component, class-balanced gradient
mean. It refreshes one 500-row component per ordinary step and all components
at one common state every 50 steps. The exact `full` mode contains no chunk,
padding, mask, or scan path. Only the common-state anchor is mathematically the
complete-pool gradient; the cached direction between anchors must pass the
registered equivalence experiment before use as a precision-preserving arm.

The runtime entry point is `allthemix.cli.train`. Registered configurations and
runners are `configs/cifar100/preact_resnet18/salda_ga.yaml` with
`scripts/experiment_run/run_cifar100_salda_ga.sh`, and
`configs/stl10/preact_resnet18/salda_ga.yaml` with
`scripts/experiment_run/run_stl10_salda_ga.sh`.

The production path is split into three modules:

- `gradient_alignment_strategy.py`: standard batch recipe, fresh direction,
  score, policy, and update orchestration.
- `scorers/gradient_alignment.py`: full-parameter and classifier-head
  validation gradients and training-mode logit tangents.
- `policies/per_row_continuous.py`: score-only, soft-label, and mean-one
  reweight decisions, including equal-dose shuffled controls.

## Search landmarks

Important algorithm blocks use paired comments of the form
`# #### <LANDMARK>: START/END ####`. Find every boundary with:

```bash
rg -n "# #### .*: (START|END) ####" salutary_da allthemix/cli/train.py
```

The stable landmark names are:

- `GA HARD-LABEL GAIN PROJECTION`
- `GA FULL-PARAMETER VALIDATION DIRECTION`
- `GA FULL-PARAMETER JVP`
- `GA CLASSIFIER-HEAD VALIDATION DIRECTION`
- `GA CLASSIFIER-HEAD DIRECTIONAL DERIVATIVE`
- `GA VALIDATION DIRECTION SCHEDULE`
- `GA STEP DIRECTION REFRESH`
- `GA STEP SCORE DISPATCH`
- `GA STEP POLICY DECISION`
- `GA STEP ACTION UPDATE`
- `GA POLICY ROW SELECTION`
- `GA POLICY ACTION MATERIALIZATION`
- `SALDA VTEST PRELOAD EXCLUSION`
- `SALDA PRE-ENDPOINT WORKLOAD CLOSURE`
- `SALDA BEST-VDEV CHECKPOINT GATE`
- `SALDA SEALED VTEST DATASET GATE`
- `SALDA BEST-VDEV RESTORE`
- `SALDA SEALED VTEST DATASET OPEN`
- `SHARED FINAL-TEST EVALUATION`

The markers are navigation contracts, not alternate control flow. Keep each
pair around the concrete implementation named by the landmark and update the
structural landmark test when a block is deliberately moved or renamed.

Every registered run uses four devices with distributed SyncBN, global batch
128, the complete 5,000-row Vdev pool, strict Vdev checkpoint selection, and
a sealed Vtest split. Vtest is read only for the formal final evaluation of
the best Vdev checkpoint.

## Guided hard-label stamping (experiment-line adoption, 2026-09-03)

Two additional modules adopted from the GA experiment line (regression-twin
validated bit-identical against the archived harness):

- `guided_step.py`: `make_ga_step` — the fused pmapped step for hard-label
  top-K relabeling: full-parameter or classifier-head guidance direction,
  optional chunk-consensus votes (partition the direction draw into M
  voters) and per-sample head-kernel votes, budgeted selection by relabel
  gain, stamping, SGD update. `make_mixup_step` is the matching control.
- `scorers/hard_label_gain.py`: first-order utility and per-class relabel
  gains (`scores_from_tangent`, `ga_row_scores`, `head_direction_shard`).

Harness: `scripts/experiment_run/train_cifar100_ga_dual_guidance_v12.py`
(runners `run_vote_opt.sh` for CIFAR-100, `run_c10_ga.sh` for CIFAR-10).
The pre-slim research registry (legacy gates, bootstrap/augmented/liveness
voters, probes) is archived under
`.artifacts/ga_val_test/legacy_registry_v1/`.
