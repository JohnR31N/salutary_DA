"""Gradient-alignment guided training steps (production set).

make_ga_step builds the pmapped guided update for the retained method
family: full-parameter or last-layer guidance direction, optional
chunk-consensus votes (partition the direction draw into M voters) and
per-sample head-kernel votes, budgeted top-K selection by relabel gain,
hard-label stamping, and the SGD update. make_mixup_step is the matching
plain-MixUp update.

Slimmed 2026-09-03 from the research registry (legacy consistency gates,
bootstrap/augmented/liveness/median voters, confidence stamping, band and
margin filters, probes). The full registry is archived verbatim at
.artifacts/ga_val_test/legacy_registry_v1/guidance/step_registry.py (package relocated to salutary_da 2026-09-03).
Retained code paths are textually identical to the registry, so kept
configurations reproduce archived runs bit-for-bit.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from salutary_da.scorers.hard_label_gain import (
    ga_row_scores,
    head_direction_shard,
    scores_from_tangent,
)


def augment_crop_flip(images, dy, dx, flip):
    """Reflect-pad4 random crop + horizontal flip, one draw per row."""
    padded = jnp.pad(images, ((0, 0), (4, 4), (4, 4), (0, 0)), mode="reflect")
    cropped = jax.vmap(
        lambda img, y, x: jax.lax.dynamic_slice(img, (y, x, 0), (32, 32, 3))
    )(padded, dy, dx)
    return jnp.where(
        flip.astype(jnp.bool_)[:, None, None, None],
        cropped[:, :, ::-1, :], cropped)



def make_ga_step(apply_fn, budget_rows, parameter_scope="full",
                 stamp_alpha=1.0, vote_chunks=0, vote_mode="rank",
                 vote_threshold=0.6, head_vote="off",
                 vote_linearize=False, vote_rotate=0,
                 direction_refresh=False, *,
                 num_classes=100, train_batch=128,
                 direction_batch=1024,
                 tau_grid=(0.05, 0.1, 0.2, 0.3, 0.5, 0.8)):

    @partial(jax.pmap, axis_name="batch")
    def ga_step(state, pool_x, pool_y, vpool_x, vpool_y, tidx, pidx, vidx,
                vw, lam, dkey, aug, refresh_flag, dir_carry):
        xa = augment_crop_flip(pool_x[tidx], aug[:, 0], aug[:, 1], aug[:, 4])
        xb = augment_crop_flip(pool_x[pidx], aug[:, 2], aug[:, 3], aug[:, 5])
        images = lam * xa + (1.0 - lam) * xb
        soft = (lam * jax.nn.one_hot(pool_y[tidx], num_classes)
                + (1.0 - lam) * jax.nn.one_hot(pool_y[pidx], num_classes))
        hvotes = None

        if parameter_scope == "head":
            vx = vpool_x[vidx]
            vy = vpool_y[vidx]
            _, vfeats = apply_fn(
                {"params": state.params, "batch_stats": state.batch_stats},
                vx, training=False, return_features=True)
            head = state.params["head"]["Dense_0"]
            vlogits = vfeats @ head["kernel"] + head["bias"]
            dk_shard, db_shard = head_direction_shard(
                vfeats, vlogits, vy, num_classes, direction_batch)
            d_kernel = jax.lax.psum(dk_shard, "batch")
            d_bias = jax.lax.psum(db_shard, "batch")

            # score all rows (training-mode features, frozen batch stats)
            (logits, feats), _ = apply_fn(
                {"params": state.params, "batch_stats": state.batch_stats},
                images, training=True, return_features=True,
                mutable=["batch_stats"], rngs={"dropout": dkey},
                sync_batch_stats=True)
            _tangent, utility, gains = ga_row_scores(
                feats, logits, soft, d_kernel, d_bias)
            if head_vote != "off":
                # B-scheme votes in the all-head path: every tensor is
                # already in scope (eval-mode val feats/logits, train-mode
                # mixed feats) — zero extra forwards.
                g_hv = jax.nn.softmax(vlogits, axis=-1) - jax.nn.one_hot(
                    vy, num_classes)                      # (V, C)
                s_dot = soft @ g_hv.T                     # (B, V)
                f_dot = feats @ vfeats.T + 1.0            # (B, V)
                signed = (s_dot[:, :, None]
                          - g_hv[None, :, :]) * f_dot[:, :, None]
                hvotes = jax.lax.psum(
                    jnp.sum(signed > 0.0, axis=1).astype(jnp.int32),
                    "batch")
        else:
            # full-parameter direction: u = grad over ALL params of the mean
            # val CE (global mean via /direction_batch + psum).
            vx = vpool_x[vidx]
            vy = vpool_y[vidx]

            def train_logits_fn(params):
                (out, _f), _ = apply_fn(
                    {"params": params, "batch_stats": state.batch_stats},
                    images, training=True, return_features=True,
                    mutable=["batch_stats"], rngs={"dropout": dkey},
                    sync_batch_stats=True)
                return out

            if vote_chunks:
                # Chunk consensus: contiguous per-device slices of the
                # (permuted) direction draw form an equal random partition
                # into `vote_chunks` global voters. mean_m(direction_m)
                # equals the pooled direction and gains are linear in the
                # direction, so mean_m(gains_m) reproduces the pooled
                # gains EXACTLY while the per-chunk signs supply the
                # consensus votes.
                local_rows = vidx.shape[0] // vote_chunks
                util_acc = None
                gains_acc = None
                votes = None
                if vote_linearize:
                    # One primal linearization; each voter applies only
                    # the linear tangent map (drops M-1 primal recomputes).
                    logits, f_lin = jax.linearize(
                        train_logits_fn, state.params)
                if vote_rotate:
                    # Evaluate only `vote_rotate` voters per step, the
                    # window rotating with the optimizer step so the full
                    # jury is covered across consecutive steps.
                    n_windows = vote_chunks // vote_rotate
                    win = jax.lax.rem(
                        state.step, jnp.asarray(n_windows, state.step.dtype))
                    active = [(win * vote_rotate + r) % vote_chunks
                              for r in range(vote_rotate)]
                else:
                    active = list(range(vote_chunks))
                for m in active:
                    if vote_rotate:
                        start = m * local_rows
                        cvx = jax.lax.dynamic_slice_in_dim(
                            vx, start, local_rows, 0)
                        cvy = jax.lax.dynamic_slice_in_dim(
                            vy, start, local_rows, 0)
                        cvw = jax.lax.dynamic_slice_in_dim(
                            vw, start, local_rows, 0)
                    else:
                        sl = slice(m * local_rows, (m + 1) * local_rows)
                        cvx, cvy, cvw = vx[sl], vy[sl], vw[sl]

                    def chunk_loss_fn(params, cvx=cvx, cvy=cvy, cvw=cvw):
                        vlogits = apply_fn(
                            {"params": params,
                             "batch_stats": state.batch_stats},
                            cvx, training=False)
                        vlogp = jax.nn.log_softmax(vlogits, axis=-1)
                        picked = vlogp[jnp.arange(cvy.shape[0]), cvy]
                        # global chunk weights sum to 1 (uniform vw slice
                        # scaled by the chunk count)
                        return -jnp.sum(picked * cvw) * float(vote_chunks)

                    dir_m = jax.lax.psum(
                        jax.grad(chunk_loss_fn)(state.params), "batch")
                    if vote_linearize:
                        tangent_m = f_lin(dir_m)
                    else:
                        logits, tangent_m = jax.jvp(
                            train_logits_fn, (state.params,), (dir_m,))
                    util_m, gains_m = scores_from_tangent(
                        logits, soft, tangent_m)
                    vote_m = (gains_m > 0.0).astype(jnp.int32)
                    votes = vote_m if votes is None else votes + vote_m
                    util_acc = util_m if util_acc is None else util_acc + util_m
                    gains_acc = (gains_m if gains_acc is None
                                 else gains_acc + gains_m)
                utility = util_acc / float(len(active))
                gains = gains_acc / float(len(active))
            else:
                votes = None

                def val_loss_fn(params):
                    vlogits = apply_fn(
                        {"params": params, "batch_stats": state.batch_stats},
                        vx, training=False)
                    vlogp = jax.nn.log_softmax(vlogits, axis=-1)
                    picked = vlogp[jnp.arange(vy.shape[0]), vy]
                    # per-row weights vw sum to 1 globally
                    return -jnp.sum(picked * vw)

                def _fresh_direction(_):
                    return jax.lax.psum(
                        jax.grad(val_loss_fn)(state.params), "batch")

                if direction_refresh:
                    # Stale-direction mode: recompute the dev direction
                    # only when the host raises refresh_flag; otherwise
                    # reuse the device-resident carry. Scoring (JVP at the
                    # CURRENT theta) stays fresh every step - only the
                    # direction ages.
                    direction = jax.lax.cond(
                        refresh_flag > 0, _fresh_direction,
                        lambda _: dir_carry, operand=None)
                else:
                    direction = _fresh_direction(None)
                # tangent J_theta logits @ u via one jvp through the same
                # training-mode forward the head path scores with.
                logits, tangent = jax.jvp(
                    train_logits_fn, (state.params,), (direction,))
                utility, gains = scores_from_tangent(logits, soft, tangent)

            if head_vote != "off":
                # Per-sample head-kernel votes: gain_head[i,c,j] =
                # <soft_i - e_c, p_j - e_{y_j}> * (<f_i, f_j> + 1); each
                # device counts votes from its own vidx shard and psum
                # yields the global tally.
                (_tl, tfeats), _ = apply_fn(
                    {"params": state.params,
                     "batch_stats": state.batch_stats},
                    images, training=True, return_features=True,
                    mutable=["batch_stats"], rngs={"dropout": dkey},
                    sync_batch_stats=True)
                vlogits_hv, vfeats = apply_fn(
                    {"params": state.params,
                     "batch_stats": state.batch_stats},
                    vx, training=False, return_features=True)
                g = jax.nn.softmax(vlogits_hv, axis=-1) - jax.nn.one_hot(
                    vy, num_classes)                      # (V, C)
                s_dot = soft @ g.T                        # (B, V)
                f_dot = tfeats @ vfeats.T + 1.0           # (B, V)
                signed = (s_dot[:, :, None]
                          - g[None, :, :]) * f_dot[:, :, None]  # (B, V, C)
                hvotes = jax.lax.psum(
                    jnp.sum(signed > 0.0, axis=1).astype(jnp.int32),
                    "batch")

        harmful = utility < 0.0
        votes_best = None
        best_hard = jnp.argmax(gains, axis=-1)
        best_gain = jnp.max(gains, axis=-1)
        if budget_rows == 0:
            selected = jnp.zeros_like(harmful)
        elif budget_rows < train_batch:
            # Global top-K: among harmful rows, keep only the batch-wide
            # top-`budget_rows` by repair value (best-gain); consensus
            # votes gate eligibility or lead the ranking lexicographically.
            rank_value = best_gain
            n_voters = vote_chunks
            if n_voters or head_vote != "off":
                rid = jnp.arange(best_hard.shape[0])
                all_g = jax.lax.all_gather(best_gain, "batch").reshape(-1)
                gmax = jnp.max(jnp.abs(all_g)) + 1e-12
                if head_vote != "off":
                    hbest = hvotes[rid, best_hard]
                    if head_vote == "gate":
                        hmin = int(np.ceil(
                            float(vote_threshold) * int(direction_batch)))
                        harmful = harmful & (hbest >= hmin)
                if n_voters:
                    votes_best = votes[rid, best_hard]
                    if vote_mode == "gate":
                        vote_min = int(np.ceil(
                            float(vote_threshold) * n_voters))
                        harmful = harmful & (votes_best >= vote_min)
                    else:
                        # exact lexicographic (votes, pooled gain): the
                        # vote quantum 2*gmax exceeds the +-gmax range.
                        rank_value = (votes_best.astype(jnp.float32)
                                      * 2.0 * gmax + best_gain)
                elif head_vote == "rank":
                    votes_best = hbest
                    rank_value = (votes_best.astype(jnp.float32)
                                  * 2.0 * gmax + best_gain)
            score = jnp.where(harmful, rank_value, -jnp.inf)
            all_scores = jax.lax.all_gather(score, "batch").reshape(-1)
            kth = jnp.sort(all_scores)[train_batch - budget_rows]
            selected = harmful & (score >= kth)
        else:
            selected = harmful
        stamped = soft + float(stamp_alpha) * (
            jax.nn.one_hot(best_hard, num_classes) - soft)
        targets = jnp.where(selected[:, None], stamped, soft)

        if votes_best is not None:
            # Vote observability, psum'd: [sum votes of stamped rows,
            # stamped count, sum votes of all rows, 0, 0, 0].
            zero = jnp.zeros((), dtype=jnp.int32)
            diagnostics = jax.lax.psum(jnp.stack([
                jnp.sum(jnp.where(selected, votes_best, 0)
                        ).astype(jnp.int32),
                jnp.sum(selected.astype(jnp.int32)),
                jnp.sum(votes_best).astype(jnp.int32),
                zero, zero, zero,
            ]), "batch")
        else:
            diagnostics = jnp.zeros((6,), dtype=jnp.int32)

        # --- train update on the repaired targets ---
        def loss_fn(params):
            # sync_batch_stats stays False here: protocol trains without
            # cross-device BN sync (the scoring forward above uses True only
            # to match the scorer's tangent path).
            (out, _f), new_vars = apply_fn(
                {"params": params, "batch_stats": state.batch_stats},
                images, training=True, return_features=True,
                mutable=["batch_stats"], rngs={"dropout": dkey})
            logp = jax.nn.log_softmax(out, axis=-1)
            return jnp.mean(-jnp.sum(targets * logp, axis=-1)), new_vars["batch_stats"]

        (loss, new_bs), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        grads = jax.lax.pmean(grads, "batch")
        new_state = state.apply_gradients(grads=grads, batch_stats=new_bs)
        grid = jnp.stack([
            jax.lax.psum(jnp.sum(harmful & (best_gain > t)), "batch")
            for t in tau_grid])
        outs = (new_state, jax.lax.psum(jnp.sum(selected), "batch"),
                jax.lax.pmean(loss, "batch"), grid, diagnostics)
        if direction_refresh:
            return outs + (direction,)
        return outs

    return ga_step


def make_mixup_step(apply_fn, *, num_classes=100,
                    tau_grid=(0.05, 0.1, 0.2, 0.3, 0.5, 0.8)):
    """Plain MixUp update used on steps that do not query validation."""

    @partial(jax.pmap, axis_name="batch")
    def mixup_step(state, pool_x, pool_y, tidx, pidx, lam, dkey, aug):
        xa = augment_crop_flip(pool_x[tidx], aug[:, 0], aug[:, 1], aug[:, 4])
        xb = augment_crop_flip(pool_x[pidx], aug[:, 2], aug[:, 3], aug[:, 5])
        images = lam * xa + (1.0 - lam) * xb
        soft = (
            lam * jax.nn.one_hot(pool_y[tidx], num_classes)
            + (1.0 - lam) * jax.nn.one_hot(pool_y[pidx], num_classes)
        )

        def loss_fn(params):
            (out, _features), new_vars = apply_fn(
                {"params": params, "batch_stats": state.batch_stats},
                images, training=True, return_features=True,
                mutable=["batch_stats"], rngs={"dropout": dkey},
            )
            logp = jax.nn.log_softmax(out, axis=-1)
            return jnp.mean(-jnp.sum(soft * logp, axis=-1)), new_vars[
                "batch_stats"
            ]

        (loss, new_bs), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(state.params)
        grads = jax.lax.pmean(grads, "batch")
        new_state = state.apply_gradients(grads=grads, batch_stats=new_bs)
        zero = jax.lax.psum(jnp.asarray(0, dtype=jnp.int32), "batch")
        return (
            new_state,
            zero,
            jax.lax.pmean(loss, "batch"),
            jnp.zeros((len(tau_grid),), dtype=jnp.int32),
            jnp.zeros((6,), dtype=jnp.int32),
        )

    return mixup_step
