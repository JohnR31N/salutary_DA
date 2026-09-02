"""GA-guided MixUp trainer (production harness).

Every step: a guidance direction (full-parameter gradient of the dev pool,
or the closed-form last-layer direction) scores all candidate hard-label
relabelings of the mixed batch; the budgeted top-K by relabel gain are
stamped, optionally filtered by consensus votes (chunk voters over the
direction draw, or per-sample head-kernel voters); one SGD step trains on
the repaired targets.

Engineering: train and guidance pools live on device (replicated) for the
whole run; per step the host uploads only index vectors and scalars; the
whole step is ONE pmapped function (salutary_da.guided_step).

Slimmed 2026-09-03 from the research harness. The full registry harness is
archived at .artifacts/ga_val_test/legacy_registry_v1/trainers/. Host rng
draw order for retained configurations is unchanged, so kept arms
reproduce archived runs bit-for-bit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from allthemix.data.pipeline import (
    build_dataset_pipeline,
    build_test_pipeline,
    build_validation_pipeline,
)
from allthemix.networks.builder import build_model
from allthemix.training.engine.single.train import create_train_state
from salutary_da.guided_step import make_ga_step, make_mixup_step
from allthemix.training.utils.lr_scheduler import build_lr_schedule
from allthemix.utils.parallel import replicate_state

NUM_CLASSES = 100
TRAIN_BATCH = 128
NUM_DEVICES = 4
PER_DEV = TRAIN_BATCH // NUM_DEVICES
EPOCHS = 200
STEPS_PER_EPOCH = 390
DIRECTION_BATCH = 1024
V_PER_DEV = DIRECTION_BATCH // NUM_DEVICES
MIXUP_ALPHA = 0.2
TAU_GRID = (0.05, 0.1, 0.2, 0.3, 0.5, 0.8)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="/mnt/disks/allthemix/data")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument(
        "--dataset", type=str, default="cifar100",
        choices=("cifar100", "cifar10"),
        help="dataset name for all pipelines; sets the class count "
             "(100 or 10). Masks must be built for the matching pool.")
    p.add_argument(
        "--stop_after_epoch", type=int, default=EPOCHS,
        help="stop after this epoch while preserving the original "
             "200-epoch learning-rate schedule")
    p.add_argument(
        "--parameter_scope", type=str, default="full",
        choices=("head", "full"),
        help="GA direction/tangent scope: classifier head (closed form) "
             "or all parameters (grad + jvp)")
    p.add_argument(
        "--stamp_alpha", type=float, default=1.0,
        help="stamped target = soft + alpha*(onehot - soft); 1.0 = hard")
    p.add_argument(
        "--direction_batch", type=int, default=None,
        help="per-step validation direction batch (default 1024; set to "
             "the dev pool size for the full-dev direction)")
    p.add_argument(
        "--budget_frac", type=float, default=0.10,
        help="cap stamped rows at this fraction of the batch; harmful "
             "rows are ranked by best-gain and only the global top-K "
             "stamped")
    p.add_argument(
        "--action_stop_epoch", type=int, default=-1,
        help="stop stamping after this epoch (-1 = never); the run "
             "continues as plain mixup")
    p.add_argument(
        "--action_every_steps", type=int, default=1,
        help="query validation and apply GA once every N train steps")
    p.add_argument(
        "--split_mask", type=str, default="",
        help="npz with a boolean vdev_mask over the [validation-half; "
             "test-half] protocol pool order; True rows form Vdev, the "
             "complement is sealed")
    p.add_argument(
        "--sealed_mask", type=str, default="",
        help="optional npz (vdev_mask key) selecting the sealed rows "
             "directly instead of the split_mask complement; may overlap "
             "the Vdev rows (independently sampled val/test design). "
             "Requires --split_mask.")
    p.add_argument(
        "--direction_vote_chunks", type=int, default=0,
        help="chunk-consensus voting: partition the per-step direction "
             "draw into this many equal voters; an action needs broad "
             "positive-gain support to be stamped. 0 = off.")
    p.add_argument(
        "--vote_mode", type=str, default="rank", choices=("gate", "rank"),
        help="gate = eligibility needs >= ceil(threshold*voters) votes; "
             "rank = lexicographic (votes, pooled gain)")
    p.add_argument(
        "--vote_threshold", type=float, default=0.6,
        help="gate modes: minimum approving voter fraction")
    p.add_argument(
        "--head_vote", type=str, default="off",
        choices=("off", "gate", "rank"),
        help="per-sample voting via the closed-form head-gradient kernel; "
             "every direction-pool row votes on each candidate action")
    p.add_argument(
        "--direction_refresh_k", type=int, default=0,
        help="stale-direction mode: recompute the full-dev guidance "
             "direction only every k optimizer steps and reuse the "
             "device-resident direction in between (scoring at the "
             "current theta stays per-step). 0 = fresh every step. "
             "Plain full-scope direction only.")
    p.add_argument(
        "--vote_linearize", action="store_true",
        help="chunk voters: linearize the training forward once per step "
             "and apply only the tangent map per voter (drops M-1 primal "
             "recomputes; same math, negligible fp difference)")
    p.add_argument(
        "--vote_rotate", type=int, default=0,
        help="evaluate only this many rotating chunk voters per step "
             "(window advances with the optimizer step so the full jury "
             "is covered over consecutive steps); 0 = all voters")
    p.add_argument(
        "--per_row_stats", action="store_true",
        help="record per-row correctness of the Vdev and sealed pools "
             "every epoch plus direction-batch inclusion counts; saved "
             "as <out>_per_row.npz")
    p.add_argument(
        "--track_sealed_each_epoch", action="store_true",
        help="evaluate the sealed pool every epoch so the trajectory "
             "minimum can be reported")
    return p.parse_args()


def _collect(ds):
    xs, ys = [], []
    for a, b in ds:
        xs.append(np.asarray(a, dtype=np.float32))
        ys.append(np.asarray(b, dtype=np.int32))
    return np.concatenate(xs), np.concatenate(ys)


def _replicated(array):
    return jax.device_put_replicated(jnp.asarray(array), jax.local_devices())


def _sharded(array, rows_per_dev):
    array = np.asarray(array)
    usable = rows_per_dev * NUM_DEVICES
    return jax.device_put_sharded(
        [jnp.asarray(array[i * rows_per_dev:(i + 1) * rows_per_dev]) for i in range(NUM_DEVICES)],
        jax.local_devices(),
    ), usable


def make_eval(apply_fn):
    @partial(jax.pmap, axis_name="batch")
    def eval_all(state, images, labels):
        logits = apply_fn(
            {"params": state.params, "batch_stats": state.batch_stats},
            images, training=False)
        logp = jax.nn.log_softmax(logits, axis=-1)
        loss_sum = -jnp.sum(logp[jnp.arange(labels.shape[0]), labels])
        correct = jnp.sum(jnp.argmax(logits, axis=-1) == labels)
        return jax.lax.psum(loss_sum, "batch"), jax.lax.psum(correct, "batch")

    return eval_all


def main() -> None:
    args = _parse_args()
    split_mask_sha256 = _sha256(args.split_mask) if args.split_mask else None
    sealed_mask_sha256 = (
        _sha256(args.sealed_mask) if args.sealed_mask else None)
    if args.action_every_steps <= 0:
        raise ValueError("--action_every_steps must be positive")
    global DIRECTION_BATCH, V_PER_DEV, NUM_CLASSES
    NUM_CLASSES = 10 if args.dataset == "cifar10" else 100
    if args.direction_batch:
        DIRECTION_BATCH = int(args.direction_batch)
        if DIRECTION_BATCH % NUM_DEVICES:
            raise ValueError("--direction_batch must be divisible by device count")
        V_PER_DEV = DIRECTION_BATCH // NUM_DEVICES
    if jax.default_backend() != "tpu" or jax.local_device_count() != NUM_DEVICES:
        raise RuntimeError("requires the TPU backend and four devices")
    if not 1 <= args.stop_after_epoch <= EPOCHS:
        raise ValueError(f"--stop_after_epoch must lie in 1..{EPOCHS}")
    epochs = 2 if args.smoke else args.stop_after_epoch
    steps = 6 if args.smoke else STEPS_PER_EPOCH

    vdev_x, vdev_y = _collect(build_validation_pipeline(
        name=args.dataset, data_dir=args.data_dir, batch_size=TRAIN_BATCH,
        validation_split=0.5, val_source="test"))
    sealed_x, sealed_y = _collect(build_test_pipeline(
        name=args.dataset, data_dir=args.data_dir, batch_size=TRAIN_BATCH,
        val_source="test", validation_split=0.5))
    if args.split_mask:
        full_x = np.concatenate([vdev_x, sealed_x])
        full_y = np.concatenate([vdev_y, sealed_y])
        mask = np.load(args.split_mask)["vdev_mask"].astype(bool)
        if mask.shape[0] != full_y.shape[0]:
            raise ValueError("split_mask length does not match the pool")
        if not (np.bincount(full_y[mask], minlength=NUM_CLASSES)
                == np.bincount(full_y[~mask], minlength=NUM_CLASSES)).all():
            raise ValueError("split_mask is not class-balanced")
        vdev_x, vdev_y = full_x[mask], full_y[mask]
        sealed_x, sealed_y = full_x[~mask], full_y[~mask]
        if args.sealed_mask:
            # Independently sampled sealed side; overlap with Vdev allowed.
            smask = np.load(args.sealed_mask)["vdev_mask"].astype(bool)
            if smask.shape[0] != full_y.shape[0]:
                raise ValueError("sealed_mask length does not match the pool")
            if not (np.bincount(full_y[smask], minlength=NUM_CLASSES)
                    == np.bincount(full_y[mask],
                                   minlength=NUM_CLASSES)).all():
                raise ValueError("sealed_mask is not class-balanced")
            sealed_x, sealed_y = full_x[smask], full_y[smask]
            print("sealed_mask overlap rows with vdev:",
                  int(np.sum(mask & smask)), flush=True)
            _shared_dbg = np.flatnonzero(mask & smask)[:5]
            for _r in _shared_dbg:
                _pvi = int(mask[:_r].sum())
                _pti = int(smask[:_r].sum())
                print("IDCHK row %d pixdiff %.6f labdiff %d" % (
                    _r,
                    float(np.abs(vdev_x[_pvi] - sealed_x[_pti]).max()),
                    int(vdev_y[_pvi]) - int(sealed_y[_pti])), flush=True)
    elif args.sealed_mask:
        raise ValueError("--sealed_mask requires --split_mask")
    # Pool is collected UNAUGMENTED; reflect-pad4 random-crop + flip are
    # applied on device per step (protocol semantics: fresh draw each use).
    train_ds, _ = build_dataset_pipeline(
        name=args.dataset, data_dir=args.data_dir, batch_size=TRAIN_BATCH,
        shuffle_buffer_size=10_000, drop_remainder=True, use_basic_augmentation=False,
        validation_split=0.5, eval_on_test=False, val_source="test",
        seed=args.seed, deterministic_data=True)
    train_x, train_y = _collect(train_ds)
    pool_size = train_x.shape[0]
    vdev_rows = vdev_x.shape[0]
    sealed_rows = sealed_x.shape[0]
    direction_x, direction_y = vdev_x, vdev_y
    if DIRECTION_BATCH > vdev_rows:
        DIRECTION_BATCH = (vdev_rows // NUM_DEVICES) * NUM_DEVICES
        V_PER_DEV = DIRECTION_BATCH // NUM_DEVICES
    if DIRECTION_BATCH <= 0:
        raise ValueError("guidance pool is too small for four TPU devices")

    if args.direction_vote_chunks or args.head_vote != "off":
        if (args.direction_vote_chunks
                and args.parameter_scope != "full"):
            raise ValueError(
                "chunk voters require --parameter_scope full")
        if (args.head_vote != "off"
                and args.parameter_scope == "full"
                and DIRECTION_BATCH != int(direction_y.shape[0])):
            raise ValueError(
                "full-scope head voting requires --direction_batch == "
                "the vdev pool size")
        if (args.direction_vote_chunks
                and V_PER_DEV % args.direction_vote_chunks):
            raise ValueError(
                "--direction_vote_chunks must divide the per-device "
                f"direction rows ({V_PER_DEV})")

    sched = build_lr_schedule(
        schedule_name="cosine", base_learning_rate=0.1, steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS, decay_epochs=[100, 150], decay_rate=0.1,
        min_learning_rate=0.0, warmup_epochs=0)
    model = build_model(name="preact_resnet18", num_classes=NUM_CLASSES)
    init = create_train_state(
        rng=jax.random.PRNGKey(0), model=model, learning_rate=sched,
        momentum=0.9, weight_decay=0.0001, input_shape=(TRAIN_BATCH, 32, 32, 3))
    state = replicate_state(init)

    # resident pools
    pool_x = _replicated(train_x)
    pool_y = _replicated(train_y)
    vpool_x = _replicated(direction_x)
    vpool_y = _replicated(direction_y)
    vdev_sh_x, vdev_usable = _sharded(vdev_x, vdev_rows // NUM_DEVICES)
    vdev_sh_y, _ = _sharded(vdev_y, vdev_rows // NUM_DEVICES)
    sealed_per_dev = sealed_x.shape[0] // NUM_DEVICES
    sealed_sh_x, sealed_usable = _sharded(sealed_x, sealed_per_dev)
    sealed_sh_y, _ = _sharded(sealed_y, sealed_per_dev)

    budget_rows = int(round(TRAIN_BATCH * args.budget_frac))
    if args.vote_rotate and (not args.direction_vote_chunks
                             or args.direction_vote_chunks
                             % args.vote_rotate):
        raise ValueError(
            "--vote_rotate must divide --direction_vote_chunks")
    if args.direction_refresh_k and (
            args.parameter_scope != "full" or args.direction_vote_chunks
            or args.head_vote != "off"):
        raise ValueError(
            "--direction_refresh_k applies to the plain full-scope "
            "direction only")
    ga_step = make_ga_step(
        model.apply, budget_rows, args.parameter_scope, args.stamp_alpha,
        args.direction_vote_chunks, args.vote_mode, args.vote_threshold,
        args.head_vote, args.vote_linearize, args.vote_rotate,
        bool(args.direction_refresh_k),
        num_classes=NUM_CLASSES, train_batch=TRAIN_BATCH,
        direction_batch=DIRECTION_BATCH, tau_grid=TAU_GRID)
    plain_mixup_step = make_mixup_step(
        model.apply, num_classes=NUM_CLASSES, tau_grid=TAU_GRID)
    eval_all = make_eval(model.apply)
    rng = np.random.default_rng(args.seed)

    per_row = None
    if args.per_row_stats:
        @jax.jit
        def _row_correct(params_1, bstats_1, images, labels):
            logits = model.apply(
                {"params": params_1, "batch_stats": bstats_1},
                images, training=False)
            return (jnp.argmax(logits, axis=-1) == labels)

        def row_correct_pool(state_repl, xs, ys, chunk=500):
            p1 = jax.tree_util.tree_map(lambda x: x[0], state_repl.params)
            b1 = jax.tree_util.tree_map(
                lambda x: x[0], state_repl.batch_stats)
            out = np.zeros(xs.shape[0], dtype=np.uint8)
            for start in range(0, xs.shape[0], chunk):
                out[start:start + chunk] = np.asarray(_row_correct(
                    p1, b1, jnp.asarray(xs[start:start + chunk]),
                    jnp.asarray(ys[start:start + chunk])))
            return out

        per_row = {"vdev": [], "sealed": [],
                   "vidx_counts": np.zeros(vdev_rows, dtype=np.int64)}

    best = {"vdev_err": 1.0, "vdev_loss": 9.9, "epoch": -1, "state": None}
    best_sealed = {
        "sealed_err": 1.0,
        "sealed_loss": 9.9,
        "vdev_err": None,
        "epoch": -1,
    }
    refresh_k = args.direction_refresh_k
    refresh_dummy = jnp.zeros((NUM_DEVICES,), dtype=jnp.int32)
    dir_dummy = jnp.zeros((NUM_DEVICES, 1), dtype=jnp.int32)
    dir_carry = (jax.tree_util.tree_map(jnp.zeros_like, state.params)
                 if refresh_k else dir_dummy)
    history = []
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wall = time.perf_counter()

    for epoch in range(1, epochs + 1):
        ep_start = time.perf_counter()
        mod_counts, losses, grid_counts, diagnostic_counts = [], [], [], []
        guidance_queries = 0
        for step_in_epoch in range(steps):
            idx = rng.choice(pool_size, TRAIN_BATCH, replace=False).astype(np.int32)
            partner = idx[rng.permutation(TRAIN_BATCH)]
            vidx = rng.choice(
                vdev_rows, DIRECTION_BATCH,
                replace=False).astype(np.int32)
            if per_row is not None:
                per_row["vidx_counts"] += np.bincount(
                    vidx, minlength=vdev_rows)
            lam = np.float32(rng.beta(MIXUP_ALPHA, MIXUP_ALPHA))
            tidx = jnp.asarray(idx.reshape(NUM_DEVICES, PER_DEV))
            pidx = jnp.asarray(partner.reshape(NUM_DEVICES, PER_DEV))
            vidx_d = jnp.asarray(vidx.reshape(NUM_DEVICES, V_PER_DEV))
            vw = np.full(
                DIRECTION_BATCH, 1.0 / DIRECTION_BATCH,
                dtype=np.float32)
            vw_d = jnp.asarray(vw.reshape(NUM_DEVICES, V_PER_DEV))
            lam_d = jnp.full((NUM_DEVICES,), lam)
            dkey = jax.random.split(
                jax.random.PRNGKey(int(rng.integers(0, 2**31 - 1))), NUM_DEVICES)
            aug = np.concatenate([
                rng.integers(0, 9, (TRAIN_BATCH, 4)),
                rng.integers(0, 2, (TRAIN_BATCH, 2)),
            ], axis=1).astype(np.int32)
            aug_d = jnp.asarray(aug.reshape(NUM_DEVICES, PER_DEV, 6))
            query_guidance = (
                step_in_epoch % args.action_every_steps == 0
                and (args.action_stop_epoch < 0
                     or epoch <= args.action_stop_epoch)
            )
            if query_guidance:
                guidance_queries += 1
                if refresh_k:
                    gstep = (epoch - 1) * steps + step_in_epoch
                    refresh_d = jnp.full(
                        (NUM_DEVICES,),
                        1 if gstep % refresh_k == 0 else 0,
                        dtype=jnp.int32)
                    (state, mod, loss, grid, diagnostics,
                     dir_carry) = ga_step(
                        state, pool_x, pool_y, vpool_x, vpool_y,
                        tidx, pidx, vidx_d, vw_d, lam_d, dkey, aug_d,
                        refresh_d, dir_carry)
                else:
                    state, mod, loss, grid, diagnostics = ga_step(
                        state, pool_x, pool_y, vpool_x, vpool_y,
                        tidx, pidx, vidx_d, vw_d, lam_d, dkey, aug_d,
                        refresh_dummy, dir_dummy)
            else:
                state, mod, loss, grid, diagnostics = plain_mixup_step(
                    state, pool_x, pool_y, tidx, pidx, lam_d, dkey, aug_d)
            mod_counts.append(mod)
            losses.append(loss)
            grid_counts.append(grid)
            diagnostic_counts.append(diagnostics)

        mod_frac = float(np.mean([np.asarray(m)[0] for m in mod_counts])) / TRAIN_BATCH
        grid_frac = (np.mean(
            [np.asarray(g)[0] for g in grid_counts], axis=0) / TRAIN_BATCH).tolist()
        diagnostic_frac = (np.mean(
            [np.asarray(d)[0] for d in diagnostic_counts], axis=0)
            / TRAIN_BATCH).tolist()
        if args.direction_vote_chunks or args.head_vote != "off":
            diagnostic_names = (
                "votes_x_selected_over_B",
                "selected_frac_dup",
                "votes_sum_over_B",
                "vote_unused3",
                "vote_unused4",
                "vote_unused5",
            )
        else:
            diagnostic_names = tuple(f"unused{i}" for i in range(6))
        action_diagnostics = {
            name: float(value)
            for name, value in zip(diagnostic_names, diagnostic_frac)
        }
        vloss_sum, vcorrect = eval_all(state, vdev_sh_x, vdev_sh_y)
        vl = float(vloss_sum[0]) / vdev_usable
        ve = 1.0 - float(vcorrect[0]) / vdev_usable
        if args.track_sealed_each_epoch:
            sealed_loss_sum, sealed_correct = eval_all(
                state, sealed_sh_x, sealed_sh_y
            )
            epoch_sealed_loss = float(sealed_loss_sum[0]) / sealed_usable
            epoch_sealed_err = 1.0 - float(sealed_correct[0]) / sealed_usable
            if epoch_sealed_err < best_sealed["sealed_err"]:
                best_sealed.update(
                    sealed_err=epoch_sealed_err,
                    sealed_loss=epoch_sealed_loss,
                    vdev_err=ve,
                    epoch=epoch,
                )
        else:
            epoch_sealed_loss = None
            epoch_sealed_err = None
        ep_seconds = time.perf_counter() - ep_start
        if ve < best["vdev_err"]:
            best.update(
                vdev_err=ve, vdev_loss=vl, epoch=epoch,
                state=jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state)))
        rec = {"epoch": epoch, "modified_frac": mod_frac,
               "guidance_query_frac": guidance_queries / steps,
               "vdev_loss": vl, "vdev_err": ve, "epoch_seconds": ep_seconds,
               "sealed_loss": epoch_sealed_loss,
               "sealed_err": epoch_sealed_err,
               "tau_grid_frac": {str(t): round(f, 4) for t, f in zip(TAU_GRID, grid_frac)},
               "action_diagnostics": action_diagnostics}
        history.append(rec)
        if per_row is not None:
            per_row["vdev"].append(row_correct_pool(state, vdev_x, vdev_y))
            per_row["sealed"].append(
                row_correct_pool(state, sealed_x, sealed_y))
            if epoch == 1 and args.sealed_mask:
                _m = np.load(args.split_mask)["vdev_mask"].astype(bool)
                _s = np.load(args.sealed_mask)["vdev_mask"].astype(bool)
                _pv = np.cumsum(_m) - 1
                _pt = np.cumsum(_s) - 1
                _sh = np.flatnonzero(_m & _s)
                _va = per_row["vdev"][-1]
                _sa = per_row["sealed"][-1]
                _bad = [int(r) for r in _sh
                        if _va[_pv[r]] != _sa[_pt[r]]]
                print("IDCHK2 epoch1 shared=%d disagree=%d" % (
                    len(_sh), len(_bad)), flush=True)
                for _r in _bad[:3]:
                    print("IDCHK2 row %d pixdiff %.6f labdiff %d "
                          "v=%d s=%d" % (
                              _r,
                              float(np.abs(vdev_x[_pv[_r]]
                                           - sealed_x[_pt[_r]]).max()),
                              int(vdev_y[_pv[_r]]) - int(sealed_y[_pt[_r]]),
                              int(_va[_pv[_r]]), int(_sa[_pt[_r]])),
                          flush=True)
        sealed_text = (
            f" test {epoch_sealed_err*100:.2f}%"
            if epoch_sealed_err is not None else ""
        )
        print(
            f"ep{epoch:3d} mod={mod_frac:.1%} vloss {vl:.4f} "
            f"val {ve*100:.2f}%{sealed_text} "
            f"| {ep_seconds:.1f}s | grid " +
            " ".join(f"{t}:{f:.0%}" for t, f in zip(TAU_GRID, grid_frac)),
            flush=True)
        out_path.write_text(json.dumps({
            "status": "RUNNING", "dataset": args.dataset,
            "budget_rows": budget_rows, "budget_frac": args.budget_frac,
            "seed": args.seed,
            "stop_after_epoch": int(epochs),
            "vdev_rows": int(vdev_rows),
            "split_mask": args.split_mask,
            "split_mask_sha256": split_mask_sha256,
            "sealed_mask": args.sealed_mask,
            "sealed_mask_sha256": sealed_mask_sha256,
            "direction_pool_rows": int(direction_y.shape[0]),
            "sealed_rows": int(sealed_x.shape[0]),
            "direction_batch": DIRECTION_BATCH,
            "parameter_scope": args.parameter_scope,
            "action_stop_epoch": args.action_stop_epoch,
            "action_every_steps": args.action_every_steps,
            "direction_vote_chunks": args.direction_vote_chunks,
            "vote_mode": args.vote_mode,
            "vote_threshold": args.vote_threshold,
            "head_vote": args.head_vote,
            "vote_linearize": args.vote_linearize,
            "vote_rotate": args.vote_rotate,
            "direction_refresh_k": args.direction_refresh_k,
            "stamp_alpha": args.stamp_alpha,
            "elapsed_seconds": time.perf_counter() - wall,
            "best": {k: v for k, v in best.items() if k != "state"},
            "best_sealed": best_sealed if args.track_sealed_each_epoch else None,
            "track_sealed_each_epoch": args.track_sealed_each_epoch,
            "history": history,
        }, indent=2, sort_keys=True), encoding="utf-8")

    st = replicate_state(best["state"]) if best["state"] is not None else state
    sloss_sum, scorrect = eval_all(st, sealed_sh_x, sealed_sh_y)
    floss_sum, fcorrect = eval_all(state, sealed_sh_x, sealed_sh_y)
    sealed = {
        "sealed_loss": float(sloss_sum[0]) / sealed_usable,
        "sealed_err": 1.0 - float(scorrect[0]) / sealed_usable,
        "at_epoch": best["epoch"],
        "sealed_final_err": 1.0 - float(fcorrect[0]) / sealed_usable,
        "sealed_final_loss": float(floss_sum[0]) / sealed_usable,
    }
    print(f"SEALED ga_hard: err {sealed['sealed_err']*100:.2f}% "
          f"loss {sealed['sealed_loss']:.4f} (best ep {best['epoch']})", flush=True)

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    if per_row is not None:
        np.savez_compressed(
            str(out_path) + "_per_row.npz",
            vdev=np.stack(per_row["vdev"]).astype(np.uint8),
            sealed=np.stack(per_row["sealed"]).astype(np.uint8),
            vidx_counts=per_row["vidx_counts"])
        print("PER_ROW_SAVED", str(out_path) + "_per_row.npz", flush=True)
    payload["status"] = "SUCCESS"
    payload["sealed"] = sealed
    payload["elapsed_seconds"] = time.perf_counter() - wall
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
