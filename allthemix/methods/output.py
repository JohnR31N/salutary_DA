from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class MixOutput(NamedTuple):
    """The single output contract every mixing method returns.

    All mixers registered in ``allthemix.methods.selector`` return this
    container (CatchUpMix returns a named superset with the same five
    fields), and all consumers access fields by NAME — never by position.
    ``tests/methods_tests/test_mix_output_contract.py`` enforces both sides.

    - ``images``    mixed batch, same shape as the input batch
    - ``labels_a``  anchor labels
    - ``labels_b``  partner labels (equal to ``labels_a`` when unmixed)
    - ``lam``       mixing coefficient(s): scalar or per-sample, method-defined
    - ``perm``      pairing indices actually used (identity when unmixed;
                    cross-device pairing passes the local relative permutation)
    """

    images: jnp.ndarray
    labels_a: jnp.ndarray
    labels_b: jnp.ndarray
    lam: jnp.ndarray
    perm: jnp.ndarray
