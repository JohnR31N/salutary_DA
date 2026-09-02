from __future__ import annotations

import inspect
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from allthemix.methods.guidedmixup import (
    _box_blur_2d_single_channel,
    _build_pairing,
    _compute_l2_distance_matrix,
    _compute_spectral_residual_saliency_maps,
    _greedy_pairing_matrix,
    _make_gaussian_kernel_1d,
    _normalize_saliency_maps,
    _pair_by_greedy_max_distance,
    _pair_by_random,
    guided_sr,
    guidedmixup,
)


def _assert_allclose(
    actual,
    expected,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> None:
    """Assert allclose."""
    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        atol=atol,
        rtol=rtol,
    )


def _assert_valid_permutation(
    permutation,
    batch_size: int,
) -> None:
    """Assert valid permutation."""
    permutation_np = np.asarray(
        permutation,
        dtype=np.int32,
    )

    assert permutation_np.shape == (batch_size,)
    assert sorted(permutation_np.tolist()) == list(range(batch_size))


def _make_toy_batch(
    batch_size: int = 8,
    height: int = 4,
    width: int = 4,
    channels: int = 3,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Support make toy batch."""
    images = np.arange(
        batch_size * height * width * channels,
        dtype=np.float32,
    ).reshape(
        batch_size,
        height,
        width,
        channels,
    )

    images = images / images.max()

    labels = np.arange(
        batch_size,
        dtype=np.int32,
    )

    saliency_maps = np.zeros(
        (
            batch_size,
            height,
            width,
        ),
        dtype=np.float32,
    )

    for i in range(batch_size):
        y1 = i % height
        x1 = (i * 2 + 1) % width
        y2 = (i + 1) % height
        x2 = (i * 3 + 2) % width

        saliency_maps[i] += 0.01
        saliency_maps[i, y1, x1] += 1.0
        saliency_maps[i, y2, x2] += 0.5

    return (
        jnp.asarray(images),
        jnp.asarray(labels),
        jnp.asarray(saliency_maps),
    )


def _preprocess_saliency_like_guidedmixup(
    saliency_maps: jnp.ndarray,
    blur_kernel: int = 3,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """Support preprocess saliency like guidedmixup."""
    saliency_maps = _normalize_saliency_maps(
        saliency_maps,
        eps=eps,
    )

    saliency_maps = _box_blur_2d_single_channel(
        saliency_maps,
        kernel_size=blur_kernel,
    )

    saliency_maps = _normalize_saliency_maps(
        saliency_maps,
        eps=eps,
    )

    return saliency_maps


def test_normalize_saliency_maps_adds_channel_and_sum_to_one() -> None:
    """Verify that normalize saliency maps adds channel and sum to one."""
    saliency_maps = jnp.asarray(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            [
                [-1.0, 0.0],
                [0.0, 3.0],
            ],
        ],
        dtype=jnp.float32,
    )

    normalized = _normalize_saliency_maps(
        saliency_maps,
    )

    assert normalized.shape == (2, 2, 2, 1)

    sums = jnp.sum(
        normalized,
        axis=(1, 2, 3),
    )

    _assert_allclose(
        sums,
        jnp.ones(
            (
                2,
            ),
            dtype=jnp.float32,
        ),
    )

    assert float(jnp.min(normalized)) >= 0.0


def test_normalize_saliency_maps_handles_zero_map_without_nan() -> None:
    """Verify that normalize saliency maps handles zero map without nan."""
    saliency_maps = jnp.zeros(
        (
            2,
            4,
            4,
        ),
        dtype=jnp.float32,
    )

    normalized = _normalize_saliency_maps(
        saliency_maps,
    )

    assert normalized.shape == (2, 4, 4, 1)
    assert bool(jnp.all(jnp.isfinite(normalized)))

    _assert_allclose(
        normalized,
        jnp.zeros_like(normalized),
    )


def test_box_blur_kernel_one_is_identity() -> None:
    """Verify that box blur kernel one is identity."""
    saliency_maps = jnp.arange(
        2 * 4 * 4,
        dtype=jnp.float32,
    ).reshape(
        2,
        4,
        4,
        1,
    )

    blurred = _box_blur_2d_single_channel(
        saliency_maps,
        kernel_size=1,
    )

    _assert_allclose(
        blurred,
        saliency_maps,
    )


def test_box_blur_rejects_even_kernel() -> None:
    """Verify that box blur rejects even kernel."""
    saliency_maps = jnp.ones(
        (
            2,
            4,
            4,
            1,
        ),
        dtype=jnp.float32,
    )

    with pytest.raises(ValueError):
        _box_blur_2d_single_channel(
            saliency_maps,
            kernel_size=2,
        )


def test_box_blur_constant_map_stays_constant() -> None:
    """Verify that box blur constant map stays constant."""
    saliency_maps = jnp.ones(
        (
            3,
            5,
            5,
            1,
        ),
        dtype=jnp.float32,
    ) * 7.0

    blurred = _box_blur_2d_single_channel(
        saliency_maps,
        kernel_size=3,
    )

    _assert_allclose(
        blurred,
        saliency_maps,
    )


def test_gaussian_blur_uses_reflect_padding_like_torchvision() -> None:
    """Verify that Guided-SR Gaussian blur follows torchvision padding."""
    saliency_maps = jnp.zeros(
        (
            1,
            3,
            3,
            1,
        ),
        dtype=jnp.float32,
    )
    saliency_maps = saliency_maps.at[
        0,
        0,
        0,
        0,
    ].set(
        1.0,
    )

    blurred = _box_blur_2d_single_channel(
        saliency_maps,
        kernel_size=3,
    )

    kernel = _make_gaussian_kernel_1d(
        kernel_size=3,
    )
    expected_corner = kernel[1] * kernel[1]

    _assert_allclose(
        blurred[0, 0, 0, 0],
        expected_corner,
        atol=1e-6,
    )


def test_compute_l2_distance_matrix_has_correct_pairwise_distances() -> None:
    """Verify that compute l2 distance matrix has correct pairwise distances."""
    saliency_maps = jnp.asarray(
        [
            [
                [
                    [0.0],
                    [0.0],
                ],
            ],
            [
                [
                    [3.0],
                    [4.0],
                ],
            ],
            [
                [
                    [6.0],
                    [8.0],
                ],
            ],
        ],
        dtype=jnp.float32,
    )

    distance_matrix = _compute_l2_distance_matrix(
        saliency_maps,
    )

    assert distance_matrix.shape == (3, 3)

    _assert_allclose(
        distance_matrix[0, 1],
        5.0,
        atol=1e-5,
    )

    _assert_allclose(
        distance_matrix[1, 2],
        5.0,
        atol=1e-5,
    )

    _assert_allclose(
        distance_matrix[0, 2],
        10.0,
        atol=1e-5,
    )

    _assert_allclose(
        distance_matrix,
        distance_matrix.T,
        atol=1e-6,
    )


def test_greedy_pairing_matrix_selects_maximum_remaining_entries() -> None:
    """Verify that greedy pairing matrix selects maximum remaining entries."""
    distance_matrix = jnp.asarray(
        [
            [0.0, 9.0, 1.0, 2.0],
            [3.0, 0.0, 8.0, 4.0],
            [5.0, 6.0, 0.0, 7.0],
            [4.0, 3.0, 2.0, 0.0],
        ],
        dtype=jnp.float32,
    )

    permutation = _greedy_pairing_matrix(
        distance_matrix,
    )

    expected = jnp.asarray(
        [
            1,
            2,
            3,
            0,
        ],
        dtype=jnp.int32,
    )

    np.testing.assert_array_equal(
        np.asarray(permutation),
        np.asarray(expected),
    )

    _assert_valid_permutation(
        permutation,
        batch_size=4,
    )

    permutation_np = np.asarray(
        permutation,
        dtype=np.int32,
    )

    assert np.all(
        permutation_np != np.arange(4),
    )

    chosen_distances = distance_matrix[
        jnp.arange(4),
        permutation,
    ]

    _assert_allclose(
        chosen_distances,
        jnp.asarray(
            [
                9.0,
                8.0,
                7.0,
                4.0,
            ],
            dtype=jnp.float32,
        ),
    )


def test_pair_by_random_is_valid_permutation_and_deterministic_for_same_rng() -> None:
    """Verify that pair by random is valid permutation and deterministic for same rng."""
    batch_size = 16

    rng = jax.random.PRNGKey(
        0,
    )

    permutation_1 = _pair_by_random(
        rng,
        batch_size=batch_size,
    )

    permutation_2 = _pair_by_random(
        rng,
        batch_size=batch_size,
    )

    _assert_valid_permutation(
        permutation_1,
        batch_size=batch_size,
    )

    _assert_valid_permutation(
        permutation_2,
        batch_size=batch_size,
    )

    _assert_allclose(
        permutation_1,
        permutation_2,
    )


def test_pair_by_greedy_max_distance_is_valid_nonself_permutation() -> None:
    """Verify that pair by greedy max distance is valid nonself permutation."""
    _, _, saliency_maps = _make_toy_batch(
        batch_size=8,
    )

    processed_saliency = _preprocess_saliency_like_guidedmixup(
        saliency_maps,
        blur_kernel=3,
    )

    permutation = _pair_by_greedy_max_distance(
        processed_saliency,
    )

    _assert_valid_permutation(
        permutation,
        batch_size=8,
    )

    permutation_np = np.asarray(
        permutation,
        dtype=np.int32,
    )

    assert np.all(
        permutation_np != np.arange(8),
    )


def test_pair_by_greedy_max_distance_matches_greedy_pairing_matrix() -> None:
    """Verify that pair by greedy max distance matches greedy pairing matrix."""
    _, _, saliency_maps = _make_toy_batch(
        batch_size=8,
    )

    processed_saliency = _preprocess_saliency_like_guidedmixup(
        saliency_maps,
        blur_kernel=3,
    )

    distance_matrix = _compute_l2_distance_matrix(
        processed_saliency,
    )

    expected = _greedy_pairing_matrix(
        distance_matrix,
    )

    actual = _pair_by_greedy_max_distance(
        processed_saliency,
    )

    np.testing.assert_array_equal(
        np.asarray(actual),
        np.asarray(expected),
    )


def test_build_pairing_random_and_greedy_are_valid() -> None:
    """Verify that build pairing random and greedy are valid."""
    _, _, saliency_maps = _make_toy_batch(
        batch_size=8,
    )

    processed_saliency = _preprocess_saliency_like_guidedmixup(
        saliency_maps,
        blur_kernel=3,
    )

    random_permutation = _build_pairing(
        rng=jax.random.PRNGKey(0),
        saliency_maps=processed_saliency,
        condition="random",
    )

    greedy_permutation = _build_pairing(
        rng=jax.random.PRNGKey(0),
        saliency_maps=processed_saliency,
        condition="greedy",
    )

    _assert_valid_permutation(
        random_permutation,
        batch_size=8,
    )

    _assert_valid_permutation(
        greedy_permutation,
        batch_size=8,
    )

    greedy_np = np.asarray(
        greedy_permutation,
        dtype=np.int32,
    )

    assert np.all(
        greedy_np != np.arange(8),
    )


def test_build_pairing_rejects_unknown_condition() -> None:
    """Verify that build pairing rejects unknown condition."""
    _, _, saliency_maps = _make_toy_batch(
        batch_size=8,
    )

    processed_saliency = _preprocess_saliency_like_guidedmixup(
        saliency_maps,
        blur_kernel=3,
    )

    with pytest.raises(ValueError):
        _build_pairing(
            rng=jax.random.PRNGKey(0),
            saliency_maps=processed_saliency,
            condition="not_a_condition",
        )


def test_spectral_residual_saliency_maps_are_finite_nonnegative() -> None:
    """Verify that online Guided-SR saliency maps are usable."""
    images, _, _ = _make_toy_batch(
        batch_size=4,
        height=8,
        width=8,
        channels=3,
    )

    saliency_maps = _compute_spectral_residual_saliency_maps(
        images=images,
        blur_kernel=3,
    )

    assert saliency_maps.shape == (4, 8, 8, 1)
    assert bool(
        jnp.all(
            jnp.isfinite(
                saliency_maps,
            )
        )
    )
    assert float(
        jnp.min(
            saliency_maps,
        )
    ) >= 0.0


@pytest.mark.parametrize(
    "condition",
    [
        "random",
        "greedy",
    ],
)
def test_guidedmixup_matches_manual_formula_for_each_condition(
    condition: str,
) -> None:
    """Verify that guidedmixup matches manual formula for each condition."""
    images, labels, saliency_maps = _make_toy_batch(
        batch_size=8,
        height=4,
        width=4,
        channels=3,
    )

    rng = jax.random.PRNGKey(
        123,
    )

    mixed_images, labels_a, labels_b, lam, _perm = guidedmixup(
        rng=rng,
        images=images,
        labels=labels,
        saliency_maps=saliency_maps,
        num_classes=100,
        alpha=1.0,
        prob=1.0,
        blur_kernel=3,
        condition=condition,
    )

    processed_saliency = _preprocess_saliency_like_guidedmixup(
        saliency_maps,
        blur_kernel=3,
    )

    _, pairing_rng = jax.random.split(
        rng,
    )

    permutation = _build_pairing(
        rng=pairing_rng,
        saliency_maps=processed_saliency,
        condition=condition,
    )

    paired_images = images[permutation]
    paired_labels = labels[permutation]
    paired_saliency = processed_saliency[permutation]

    expected_mask = processed_saliency / (
        processed_saliency + paired_saliency + 1e-8
    )

    expected_mixed_images = expected_mask * images + (
        1.0 - expected_mask
    ) * paired_images

    expected_lam = jnp.mean(
        expected_mask,
        axis=(1, 2, 3),
    )

    _assert_allclose(
        mixed_images,
        expected_mixed_images,
        atol=1e-6,
    )

    _assert_allclose(
        labels_a,
        labels,
    )

    _assert_allclose(
        labels_b,
        paired_labels,
    )

    _assert_allclose(
        lam,
        expected_lam,
        atol=1e-6,
    )

    assert mixed_images.shape == images.shape
    assert labels_a.shape == labels.shape
    assert labels_b.shape == labels.shape
    assert lam.shape == (images.shape[0],)
    assert bool(jnp.all(jnp.isfinite(mixed_images)))
    assert bool(jnp.all(jnp.isfinite(lam)))


def test_guided_sr_matches_guidedmixup_with_online_sr_maps() -> None:
    """Verify that guided_sr uses online SR maps then the GuidedMixup formula."""
    images, labels, _ = _make_toy_batch(
        batch_size=8,
        height=8,
        width=8,
        channels=3,
    )

    rng = jax.random.PRNGKey(
        1234,
    )

    saliency_maps = _compute_spectral_residual_saliency_maps(
        images=images,
    )

    guided_sr_output = guided_sr(
        rng=rng,
        images=images,
        labels=labels,
        num_classes=100,
        alpha=1.0,
        prob=1.0,
        blur_kernel=3,
        condition="random",
    )

    guidedmixup_output = guidedmixup(
        rng=rng,
        images=images,
        labels=labels,
        saliency_maps=saliency_maps,
        num_classes=100,
        alpha=1.0,
        prob=1.0,
        blur_kernel=3,
        condition="random",
    )

    for actual, expected in zip(
        guided_sr_output,
        guidedmixup_output,
    ):
        _assert_allclose(
            actual,
            expected,
            atol=5e-4,
            rtol=5e-4,
        )


def test_guided_sr_prob_zero_returns_vanilla_batch() -> None:
    """Verify that guided_sr skips the batch when mix_prob is zero."""
    images, labels, _ = _make_toy_batch(
        batch_size=8,
        height=8,
        width=8,
        channels=3,
    )

    mixed_images, labels_a, labels_b, lam, _perm = guided_sr(
        rng=jax.random.PRNGKey(0),
        images=images,
        labels=labels,
        num_classes=100,
        alpha=1.0,
        prob=0.0,
        blur_kernel=3,
        condition="random",
    )

    _assert_allclose(
        mixed_images,
        images,
    )

    _assert_allclose(
        labels_a,
        labels,
    )

    _assert_allclose(
        labels_b,
        labels,
    )

    _assert_allclose(
        lam,
        jnp.ones_like(
            lam,
        ),
    )


def test_guidedmixup_accepts_4d_saliency_maps() -> None:
    """Verify that guidedmixup accepts 4d saliency maps."""
    images, labels, saliency_maps = _make_toy_batch(
        batch_size=8,
        height=4,
        width=4,
        channels=3,
    )

    saliency_maps_4d = saliency_maps[..., None]

    mixed_images, labels_a, labels_b, lam, _perm = guidedmixup(
        rng=jax.random.PRNGKey(0),
        images=images,
        labels=labels,
        saliency_maps=saliency_maps_4d,
        num_classes=100,
        alpha=1.0,
        prob=1.0,
        blur_kernel=3,
        condition="random",
    )

    assert mixed_images.shape == images.shape
    assert labels_a.shape == labels.shape
    assert labels_b.shape == labels.shape
    assert lam.shape == (images.shape[0],)


def test_guidedmixup_alpha_is_currently_not_used() -> None:
    """Verify that guidedmixup alpha is currently not used."""
    images, labels, saliency_maps = _make_toy_batch(
        batch_size=8,
        height=4,
        width=4,
        channels=3,
    )

    rng = jax.random.PRNGKey(
        0,
    )

    output_alpha_small = guidedmixup(
        rng=rng,
        images=images,
        labels=labels,
        saliency_maps=saliency_maps,
        num_classes=100,
        alpha=0.1,
        prob=1.0,
        blur_kernel=3,
        condition="random",
    )

    output_alpha_large = guidedmixup(
        rng=rng,
        images=images,
        labels=labels,
        saliency_maps=saliency_maps,
        num_classes=100,
        alpha=100.0,
        prob=1.0,
        blur_kernel=3,
        condition="random",
    )

    for actual, expected in zip(
        output_alpha_small,
        output_alpha_large,
    ):
        _assert_allclose(
            actual,
            expected,
            atol=1e-6,
        )


def test_guidedmixup_prob_zero_returns_vanilla_batch() -> None:
    """Verify that guidedmixup can skip mixing like official mix_prob."""
    images, labels, saliency_maps = _make_toy_batch(
        batch_size=8,
        height=4,
        width=4,
        channels=3,
    )

    mixed_images, labels_a, labels_b, lam, _perm = guidedmixup(
        rng=jax.random.PRNGKey(0),
        images=images,
        labels=labels,
        saliency_maps=saliency_maps,
        num_classes=100,
        alpha=1.0,
        prob=0.0,
        blur_kernel=3,
        condition="random",
    )

    _assert_allclose(
        mixed_images,
        images,
    )

    _assert_allclose(
        labels_a,
        labels,
    )

    _assert_allclose(
        labels_b,
        labels,
    )

    _assert_allclose(
        lam,
        jnp.ones_like(
            lam,
        ),
    )


def test_guidedmixup_reconstruction_error_is_zero_for_random_condition() -> None:
    """Verify that guidedmixup reconstruction error is zero for random condition."""
    images, labels, saliency_maps = _make_toy_batch(
        batch_size=8,
        height=4,
        width=4,
        channels=3,
    )

    rng = jax.random.PRNGKey(
        7,
    )

    mixed_images, labels_a, labels_b, lam, _perm = guidedmixup(
        rng=rng,
        images=images,
        labels=labels,
        saliency_maps=saliency_maps,
        num_classes=100,
        alpha=1.0,
        prob=1.0,
        blur_kernel=3,
        condition="random",
    )

    del labels_a
    del labels_b
    del lam

    processed_saliency = _preprocess_saliency_like_guidedmixup(
        saliency_maps,
        blur_kernel=3,
    )

    _, pairing_rng = jax.random.split(
        rng,
    )

    permutation = _build_pairing(
        rng=pairing_rng,
        saliency_maps=processed_saliency,
        condition="random",
    )

    paired_images = images[permutation]
    paired_saliency = processed_saliency[permutation]

    pixel_mask = processed_saliency / (
        processed_saliency + paired_saliency + 1e-8
    )

    reconstructed = pixel_mask * images + (
        1.0 - pixel_mask
    ) * paired_images

    reconstruction_error = jnp.mean(
        (
            mixed_images - reconstructed
        )
        ** 2
    )

    _assert_allclose(
        reconstruction_error,
        0.0,
        atol=1e-8,
    )


def test_guidedmixup_reconstruction_error_is_zero_for_greedy_condition() -> None:
    """Verify that guidedmixup reconstruction error is zero for greedy condition."""
    images, labels, saliency_maps = _make_toy_batch(
        batch_size=8,
        height=4,
        width=4,
        channels=3,
    )

    rng = jax.random.PRNGKey(
        7,
    )

    mixed_images, labels_a, labels_b, lam, _perm = guidedmixup(
        rng=rng,
        images=images,
        labels=labels,
        saliency_maps=saliency_maps,
        num_classes=100,
        alpha=1.0,
        prob=1.0,
        blur_kernel=3,
        condition="greedy",
    )

    del labels_a
    del labels_b
    del lam

    processed_saliency = _preprocess_saliency_like_guidedmixup(
        saliency_maps,
        blur_kernel=3,
    )

    _, pairing_rng = jax.random.split(
        rng,
    )

    permutation = _build_pairing(
        rng=pairing_rng,
        saliency_maps=processed_saliency,
        condition="greedy",
    )

    paired_images = images[permutation]
    paired_saliency = processed_saliency[permutation]

    pixel_mask = processed_saliency / (
        processed_saliency + paired_saliency + 1e-8
    )

    reconstructed = pixel_mask * images + (
        1.0 - pixel_mask
    ) * paired_images

    reconstruction_error = jnp.mean(
        (
            mixed_images - reconstructed
        )
        ** 2
    )

    _assert_allclose(
        reconstruction_error,
        0.0,
        atol=1e-8,
    )


def test_selector_exposes_guidedmixup_condition_argument() -> None:
    """Verify that selector exposes guidedmixup condition argument."""
    from allthemix.methods.selector import get_mixer

    signature = inspect.signature(
        get_mixer,
    )

    assert "guidedmixup_condition" in signature.parameters


def test_selector_guidedmixup_matches_direct_guidedmixup_call() -> None:
    """Verify that selector guidedmixup matches direct guidedmixup call."""
    from allthemix.methods.selector import get_mixer

    images, labels, saliency_maps = _make_toy_batch(
        batch_size=8,
        height=4,
        width=4,
        channels=3,
    )

    rng = jax.random.PRNGKey(
        11,
    )

    mixer = get_mixer(
        name="guidedmixup",
        num_classes=100,
        guidedmixup_alpha=1.0,
        guidedmixup_prob=1.0,
        guidedmixup_blur_kernel=3,
        guidedmixup_condition="random",
    )

    mixer_output = mixer(
        rng,
        images,
        labels,
        {
            "saliency_maps": saliency_maps,
        },
    )

    direct_output = guidedmixup(
        rng=rng,
        images=images,
        labels=labels,
        saliency_maps=saliency_maps,
        num_classes=100,
        alpha=1.0,
        prob=1.0,
        blur_kernel=3,
        condition="random",
    )

    for actual, expected in zip(
        mixer_output,
        direct_output,
    ):
        _assert_allclose(
            actual,
            expected,
            atol=1e-6,
        )


def test_selector_guided_sr_matches_direct_guided_sr_call_without_aux() -> None:
    """Verify that selector guided_sr does not require precomputed saliency."""
    from allthemix.methods.selector import get_mixer

    images, labels, _ = _make_toy_batch(
        batch_size=8,
        height=8,
        width=8,
        channels=3,
    )

    rng = jax.random.PRNGKey(
        17,
    )

    mixer = get_mixer(
        name="guided_sr",
        num_classes=100,
        guidedmixup_alpha=1.0,
        guidedmixup_prob=1.0,
        guidedmixup_blur_kernel=3,
        guidedmixup_condition="random",
    )

    mixer_output = mixer(
        rng,
        images,
        labels,
        None,
    )

    direct_output = guided_sr(
        rng=rng,
        images=images,
        labels=labels,
        num_classes=100,
        alpha=1.0,
        prob=1.0,
        blur_kernel=3,
        condition="random",
    )

    for actual, expected in zip(
        mixer_output,
        direct_output,
    ):
        _assert_allclose(
            actual,
            expected,
            atol=1e-6,
        )


def test_parse_args_accepts_guidedmixup_condition_from_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that parse args accepts guidedmixup condition from config."""
    from allthemix.cli.args import parse_args

    config_path = tmp_path / "guidedmixup_config.yaml"

    config_path.write_text(
        "dataset: cifar100\n"
        "model: preact_resnet18\n"
        "method: guidedmixup\n"
        "guidedmixup_condition: random\n"
        "guidedmixup_blur_kernel: 7\n"
        "batch_size: 100\n"
        "epochs: 300"
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(config_path),
        ],
    )

    args = parse_args()

    assert args.dataset == "cifar100"
    assert args.model == "preact_resnet18"
    assert args.method == "guidedmixup"
    assert args.guidedmixup_condition == "random"
    assert args.guidedmixup_blur_kernel == 7
    assert args.batch_size == 100
    assert args.epochs == 300
