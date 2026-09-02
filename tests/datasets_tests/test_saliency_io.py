from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf


def test_load_train_dataset_for_saliency_cache_uses_unshuffled_tfds_for_cifar(
    monkeypatch,
):
    """CIFAR saliency caches must match the unshuffled training read order."""
    from allthemix.data.saliency import saliency_io

    calls = []
    sentinel_dataset = object()

    def fake_tfds_load(**kwargs):
        calls.append(
            kwargs,
        )

        return sentinel_dataset

    monkeypatch.setattr(
        saliency_io.tfds,
        "load",
        fake_tfds_load,
    )

    dataset = saliency_io._load_train_dataset_for_saliency_cache(
        dataset_name="cifar10",
        data_dir="./data",
    )

    assert dataset is sentinel_dataset
    assert calls == [
        {
            "name": "cifar10",
            "split": "train",
            "data_dir": "./data",
            "shuffle_files": False,
            "download": True,
        }
    ]


def test_load_train_dataset_for_saliency_cache_uses_custom_loader_for_local_data(
    monkeypatch,
):
    """Local saliency caches should use deterministic custom loaders."""
    from allthemix.data.saliency import saliency_io

    calls = []
    sentinel_dataset = object()

    def fake_load_train_dataset(
        name: str,
        data_dir: str,
    ):
        calls.append(
            {
                "name": name,
                "data_dir": data_dir,
            }
        )

        return sentinel_dataset

    monkeypatch.setattr(
        saliency_io,
        "load_train_dataset",
        fake_load_train_dataset,
    )

    datasets = [
        saliency_io._load_train_dataset_for_saliency_cache(
            dataset_name=dataset_name,
            data_dir="./data",
        )
        for dataset_name in (
            "tiny_imagenet",
            "cars196",
            "imagenet100",
        )
    ]

    assert datasets == [
        sentinel_dataset,
        sentinel_dataset,
        sentinel_dataset,
    ]
    assert calls == [
        {
            "name": "tiny_imagenet",
            "data_dir": "./data",
        },
        {
            "name": "cars196",
            "data_dir": "./data",
        },
        {
            "name": "imagenet100",
            "data_dir": "./data",
        },
    ]


@pytest.mark.parametrize(
    "dataset_name",
    [
        "oxford_iiit_pet",
        "caltech_birds2011",
        "imagenet100",
    ],
)
def test_build_train_saliency_maps_resizes_variable_size_maps(
    monkeypatch,
    tmp_path,
    dataset_name,
):
    """Variable-size datasets should cache fixed-size saliency maps."""
    from allthemix.data.saliency import saliency_io

    dataset = tf.data.Dataset.from_generator(
        lambda: iter(
            [
                {
                    "image": np.zeros((20, 30, 3), dtype=np.uint8),
                    "label": np.int64(0),
                },
                {
                    "image": np.zeros((40, 35, 3), dtype=np.uint8),
                    "label": np.int64(1),
                },
            ]
        ),
        output_signature={
            "image": tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
            "label": tf.TensorSpec(shape=(), dtype=tf.int64),
        },
    )

    monkeypatch.setattr(
        saliency_io,
        "_load_train_dataset_for_saliency_cache",
        lambda dataset_name, data_dir: dataset,
    )
    monkeypatch.setattr(
        saliency_io,
        "compute_saliency_map",
        lambda image, method: np.ones(image.shape[:2], dtype=np.float32),
    )
    monkeypatch.setattr(
        saliency_io,
        "resolve_source_train_example_count",
        lambda **_kwargs: 2,
    )

    saliency_path = saliency_io.build_train_saliency_maps(
        dataset_name=dataset_name,
        data_dir="./data",
        saliency_dir=str(tmp_path),
        overwrite=True,
    )

    saliency_maps = np.load(
        saliency_path,
    )

    assert saliency_maps.shape == (2, 224, 224)
    assert saliency_maps.dtype == np.uint8
    assert np.all(
        saliency_maps == 255,
    )

    loaded_maps = saliency_io.load_train_saliency_maps(
        dataset_name=dataset_name,
        saliency_dir=str(tmp_path),
    )
    assert isinstance(
        loaded_maps,
        np.memmap,
    )


def test_uint8_saliency_cache_is_decoded_to_unit_interval() -> None:
    """Quantized caches must recover their [0, 1] values before mixing."""
    from allthemix.data.salmix_pipeline import (
        _preprocess_salmix_train_example,
    )

    image = tf.zeros(
        (
            4,
            4,
            3,
        ),
        dtype=tf.uint8,
    )
    saliency_map = tf.fill(
        (
            4,
            4,
        ),
        value=tf.constant(
            128,
            dtype=tf.uint8,
        ),
    )
    _image, _label, decoded_map = _preprocess_salmix_train_example(
        example={
            "image": image,
            "label": tf.constant(
                0,
                dtype=tf.int64,
            ),
        },
        saliency_map=saliency_map,
        dataset_name="cifar10",
        use_sal_basic_augmentation=False,
        image_size=4,
        saliency_augmentation_recipe="none",
    )

    np.testing.assert_allclose(
        decoded_map.numpy(),
        np.full(
            (
                4,
                4,
            ),
            128.0 / 255.0,
            dtype=np.float32,
        ),
        rtol=0.0,
        atol=1e-7,
    )


def test_saliency_dataset_reads_indexed_maps_without_full_tensor_conversion(
    tmp_path,
) -> None:
    """Memory-mapped caches should be read in stable per-example order."""
    from allthemix.data.salmix_pipeline import _build_saliency_map_dataset

    cache_path = tmp_path / "maps.npy"
    expected_maps = np.arange(
        3 * 4 * 4,
        dtype=np.uint8,
    ).reshape(
        3,
        4,
        4,
    )
    np.save(
        cache_path,
        expected_maps,
    )
    mapped_maps = np.load(
        cache_path,
        mmap_mode="r",
    )

    dataset = _build_saliency_map_dataset(
        saliency_maps=mapped_maps,
    )
    actual_maps = np.stack(
        list(
            dataset.as_numpy_iterator(),
        )
    )

    np.testing.assert_array_equal(
        actual_maps,
        expected_maps,
    )
