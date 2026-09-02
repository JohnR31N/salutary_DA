"""Hard-label relabel-gain scoring for Salutary DA guided steps.

First-order utilities and per-class relabel gains against a
guidance direction: the exact classifier-head closed form and the
tangent-based form shared by the full-parameter JVP path.
"""

import jax
import jax.numpy as jnp

def head_direction_shard(vfeats, vlogits, vy, num_classes, total_count):
    """Per-shard contribution to u = grad_{K,b} of the mean val CE loss.

    For the affine head the gradient has the closed form
    dK = feats^T (softmax - onehot)/N and db = sum over rows; the caller
    psums shard contributions so N is the GLOBAL example count.
    """
    vg = (jax.nn.softmax(vlogits, axis=-1)
          - jax.nn.one_hot(vy, num_classes)) / float(total_count)
    return vfeats.T @ vg, jnp.sum(vg, axis=0)


def ga_row_scores(feats, logits, soft, d_kernel, d_bias):
    """Tangent, per-row soft-label utility, and hard-label gains.

    tangent[i] = J_{K,b} logits(x_i) @ u (exact for the affine head);
    utility[i] = <grad_{K,b} CE(x_i, soft_i), u> (utility>0: training on the
    soft target helps val); gains[i, c] = utility(e_c) - utility(soft_i),
    the alignment change from switching row i's target to hard label c.
    """
    tangent = feats @ d_kernel + d_bias
    utility, gains = scores_from_tangent(logits, soft, tangent)
    return tangent, utility, gains


def scores_from_tangent(logits, soft, tangent):
    """Utility and hard-label gains given a per-row logit tangent J@u."""
    utility = jnp.sum((jax.nn.softmax(logits, axis=-1) - soft) * tangent, axis=-1)
    gains = jnp.sum(soft * tangent, axis=-1, keepdims=True) - tangent
    return utility, gains

