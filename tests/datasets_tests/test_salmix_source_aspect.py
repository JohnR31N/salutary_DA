from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_datasets as tfds

from allthemix.data.preprocessors.cifar import get_normalization_stats
from allthemix.data.preprocessors.simple_sal_aug import (
    apply_sal_augmentation_recipe,
)
from allthemix.data.salmix_pipeline import (
    _preprocess_salmix_train_example,
    _resize_image,
    _resize_saliency_map,
    build_salmix_dataset_pipeline,
    build_salmix_test_pipeline,
)


def _make_example(
    image: tf.Tensor,
    label: int = 3,
) -> dict[str, tf.Tensor]:
    return {
        "image": image,
        "label": tf.constant(
            label,
            dtype=tf.int64,
        ),
    }


def _make_gradient_image(
    height: int,
    width: int,
    low: int = 100,
    high: int = 155,
) -> tf.Tensor:
    """Build a mid-range uint8 gradient image, equal across channels."""
    ramp = np.linspace(
        low,
        high,
        num=height * width,
        dtype=np.float32,
    ).reshape(
        height,
        width,
    )
    image = np.stack(
        [
            ramp,
            ramp,
            ramp,
        ],
        axis=-1,
    )
    return tf.constant(
        np.round(image).astype(np.uint8),
    )


def test_imagenet100_salmix_train_preprocess_skips_square_squash() -> None:
    """Source-aspect datasets must crop original geometry, not a squashed square."""
    seed = tf.constant(
        [3, 7],
        dtype=tf.int32,
    )
    image = _make_gradient_image(
        height=448,
        width=224,
    )
    saliency_map = tf.constant(
        np.random.RandomState(0).randint(
            0,
            256,
            size=(224, 224),
            dtype=np.int64,
        ).astype(np.uint8),
    )

    out_image, out_label, out_map = _preprocess_salmix_train_example(
        example=_make_example(
            image,
        ),
        saliency_map=saliency_map,
        dataset_name="imagenet100",
        use_sal_basic_augmentation=False,
        image_size=224,
        saliency_augmentation_recipe="imagenet",
        seed=seed,
    )

    assert out_image.shape == (224, 224, 3)
    assert out_map.shape == (224, 224)
    assert int(out_label) == 3

    # Replicate the legacy squash-first path with the same stateless seed. If
    # the new source-aspect branch were not active, outputs would be identical.
    squashed_image = _resize_image(
        image=tf.cast(
            image,
            tf.float32,
        )
        / 255.0,
        image_size=224,
    )
    squashed_map = _resize_saliency_map(
        saliency_map=tf.cast(
            saliency_map,
            tf.float32,
        )
        / 255.0,
        image_size=224,
    )
    legacy_image, _legacy_map = apply_sal_augmentation_recipe(
        image=squashed_image,
        saliency_map=squashed_map,
        image_size=224,
        use_sal_basic_augmentation=False,
        saliency_augmentation_recipe="imagenet",
        seed=seed,
    )

    assert not np.allclose(
        np.asarray(out_image),
        np.asarray(legacy_image),
        atol=1e-3,
    )


def test_imagenet100_salmix_source_aspect_keeps_image_and_map_aligned() -> None:
    """Image and saliency map must receive the same geometric transform."""
    seed = tf.constant(
        [11, 23],
        dtype=tf.int32,
    )
    image = _make_gradient_image(
        height=320,
        width=240,
    )
    saliency_map = tf.cast(
        image[:, :, 0],
        tf.float32,
    ) / 255.0

    out_image, _, out_map = _preprocess_salmix_train_example(
        example=_make_example(
            image,
        ),
        saliency_map=saliency_map,
        dataset_name="imagenet100",
        use_sal_basic_augmentation=False,
        image_size=224,
        saliency_augmentation_recipe="imagenet",
        seed=seed,
    )

    # Color jitter and normalization are per-channel affine on a gray image,
    # so a shared crop/flip keeps channel 0 linearly correlated with the map.
    image_values = np.asarray(
        out_image,
    )[:, :, 0].reshape(-1)
    map_values = np.asarray(
        out_map,
    ).reshape(-1)

    correlation = float(
        np.corrcoef(
            image_values,
            map_values,
        )[0, 1]
    )

    assert correlation > 0.98


def test_cars196_salmix_train_preprocess_uses_fine_grained_source_aspect() -> None:
    """Cars196 with the fine_grained recipe follows the source-aspect branch."""
    seed = tf.constant(
        [5, 9],
        dtype=tf.int32,
    )
    image = _make_gradient_image(
        height=240,
        width=360,
    )
    saliency_map = tf.constant(
        np.random.RandomState(1).randint(
            0,
            256,
            size=(224, 224),
            dtype=np.int64,
        ).astype(np.uint8),
    )

    out_image, _, out_map = _preprocess_salmix_train_example(
        example=_make_example(
            image,
        ),
        saliency_map=saliency_map,
        dataset_name="cars196",
        use_sal_basic_augmentation=False,
        image_size=224,
        saliency_augmentation_recipe="fine_grained",
        seed=seed,
    )

    assert out_image.shape == (224, 224, 3)
    assert out_map.shape == (224, 224)


def test_imagenet100_salmix_train_preprocess_traces_in_graph_mode() -> None:
    """The preprocess must keep static output shapes under tf.data tracing."""

    def generate_examples():
        image = np.asarray(
            _make_gradient_image(
                height=448,
                width=224,
            )
        )
        saliency_map = np.random.RandomState(3).randint(
            0,
            256,
            size=(224, 224),
            dtype=np.int64,
        ).astype(np.uint8)
        yield image, np.int64(3), saliency_map

    dataset = tf.data.Dataset.from_generator(
        generate_examples,
        output_signature=(
            tf.TensorSpec(
                shape=(None, None, 3),
                dtype=tf.uint8,
            ),
            tf.TensorSpec(
                shape=(),
                dtype=tf.int64,
            ),
            tf.TensorSpec(
                shape=(224, 224),
                dtype=tf.uint8,
            ),
        ),
    )

    dataset = dataset.map(
        lambda image, label, saliency_map: _preprocess_salmix_train_example(
            example={
                "image": image,
                "label": label,
            },
            saliency_map=saliency_map,
            dataset_name="imagenet100",
            use_sal_basic_augmentation=False,
            image_size=224,
            saliency_augmentation_recipe="imagenet",
            seed=tf.constant(
                [3, 7],
                dtype=tf.int32,
            ),
        ),
    )

    out_image, _, out_map = next(
        iter(
            dataset,
        )
    )

    assert out_image.shape == (224, 224, 3)
    assert out_map.shape == (224, 224)


def test_cifar_salmix_train_preprocess_keeps_legacy_square_path() -> None:
    """Square datasets keep the existing squash-then-augment behavior."""
    seed = tf.constant(
        [2, 4],
        dtype=tf.int32,
    )
    image = _make_gradient_image(
        height=32,
        width=32,
    )
    saliency_map = tf.constant(
        np.random.RandomState(2).randint(
            0,
            256,
            size=(32, 32),
            dtype=np.int64,
        ).astype(np.uint8),
    )

    out_image, _, out_map = _preprocess_salmix_train_example(
        example=_make_example(
            image,
        ),
        saliency_map=saliency_map,
        dataset_name="cifar10",
        use_sal_basic_augmentation=True,
        image_size=32,
        saliency_augmentation_recipe=None,
        seed=seed,
    )

    assert out_image.shape == (32, 32, 3)
    assert out_map.shape == (32, 32)


def test_salmix_official_eval_validation_keeps_full_train_and_sealed_final(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Saliency-paired training must obey the same official-eval protocol."""
    import allthemix.data.salmix_pipeline as salmix_module

    def id_source(first_id: int) -> tf.data.Dataset:
        ids = tf.range(
            first_id,
            first_id + 40,
            dtype=tf.int32,
        )
        images = tf.cast(
            tf.tile(
                tf.reshape(
                    ids,
                    (40, 1, 1, 1),
                ),
                (1, 32, 32, 3),
            ),
            tf.uint8,
        )
        labels = tf.math.floormod(
            tf.cast(
                ids,
                tf.int64,
            ),
            10,
        )
        return tf.data.Dataset.from_tensor_slices(
            {
                "image": images,
                "label": labels,
            }
        )

    monkeypatch.setattr(
        salmix_module,
        "_load_train_dataset_without_file_shuffle",
        lambda **_kwargs: id_source(0),
    )
    monkeypatch.setattr(
        salmix_module,
        "load_test_dataset",
        lambda **_kwargs: id_source(100),
    )
    monkeypatch.setattr(
        salmix_module,
        "load_train_saliency_maps",
        lambda **_kwargs: np.zeros(
            (40, 32, 32),
            dtype=np.float32,
        ),
    )
    monkeypatch.setattr(
        salmix_module,
        "resolve_source_train_example_count",
        lambda **_kwargs: 40,
    )

    train_ds, validation_ds = build_salmix_dataset_pipeline(
        name="cifar10",
        data_dir="unused",
        batch_size=7,
        shuffle_buffer_size=40,
        drop_remainder=False,
        use_sal_basic_augmentation=False,
        validation_split=0.5,
        eval_on_test=False,
        seed=3,
        val_source="test",
    )
    final_ds = build_salmix_test_pipeline(
        name="cifar10",
        data_dir="unused",
        batch_size=7,
        val_source="test",
        validation_split=0.5,
    )
    mean, std = get_normalization_stats(
        "cifar10",
    )
    mean_0 = float(mean.numpy()[0])
    std_0 = float(std.numpy()[0])

    def decode_ids(
        dataset: tf.data.Dataset,
    ) -> set[int]:
        ids: set[int] = set()
        for batch in tfds.as_numpy(dataset):
            images = batch[0]
            for value in images[:, 0, 0, 0]:
                ids.add(
                    int(
                        round(
                            (value * std_0 + mean_0) * 255.0,
                        )
                    )
                )
        return ids

    train_ids = decode_ids(train_ds)
    validation_ids = decode_ids(validation_ds)
    final_ids = decode_ids(final_ds)

    assert train_ids == set(range(40))
    assert validation_ids.isdisjoint(final_ids)
    assert validation_ids | final_ids == set(range(100, 140))
    assert len(validation_ids) == 20
    assert len(final_ids) == 20
