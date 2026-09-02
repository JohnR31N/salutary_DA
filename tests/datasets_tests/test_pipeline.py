from __future__ import annotations

import pytest
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from allthemix.data.datasets.loader import load_train_dataset, load_test_dataset
from allthemix.data.preprocessors import tfds_image as tfds_image_preprocessor
from allthemix.data.pipeline import (
    build_dataset_pipeline,
    build_meta_validation_pipeline,
    build_raw_augmented_train_pipeline,
    build_test_pipeline,
    build_train_pipeline,
    build_validation_pipeline,
)
from allthemix.data.splits import count_class_stratified_split_examples
from allthemix.data.preprocessors.cifar import get_normalization_stats
from allthemix.data.preprocessors.selector import (
    get_preprocessor,
    get_metadata,
)


def _patch_cifar10_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use finite in-memory sources for pipeline-only shape tests."""
    import allthemix.data.pipeline as pipeline_module

    train_images = tf.zeros((256, 32, 32, 3), dtype=tf.uint8)
    train_labels = tf.math.floormod(
        tf.range(256, dtype=tf.int64),
        10,
    )
    test_images = tf.zeros((160, 32, 32, 3), dtype=tf.uint8)
    test_labels = tf.math.floormod(
        tf.range(160, dtype=tf.int64),
        10,
    )
    train_source = tf.data.Dataset.from_tensor_slices(
        {"image": train_images, "label": train_labels}
    )
    test_source = tf.data.Dataset.from_tensor_slices(
        {"image": test_images, "label": test_labels}
    )

    monkeypatch.setattr(
        pipeline_module,
        "load_train_dataset",
        lambda **_kwargs: train_source,
    )
    monkeypatch.setattr(
        pipeline_module,
        "load_test_dataset",
        lambda **_kwargs: test_source,
    )


class TestDatasetPipeline:
    def test_raw_augmented_pipeline_preserves_source_alignment(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify that IF-AugNet raw/base views retain identical examples."""
        images = tf.reshape(
            tf.range(
                4 * 32 * 32 * 3,
                dtype=tf.int32,
            ),
            (4, 32, 32, 3),
        )
        images = tf.cast(
            tf.math.floormod(
                images,
                256,
            ),
            tf.uint8,
        )
        labels = tf.constant(
            [0, 1, 2, 3],
            dtype=tf.int64,
        )
        source = tf.data.Dataset.from_tensor_slices(
            {
                "image": images,
                "label": labels,
            }
        )
        import allthemix.data.pipeline as pipeline_module

        monkeypatch.setattr(
            pipeline_module,
            "load_train_dataset",
            lambda **_kwargs: source,
        )
        dataset = build_raw_augmented_train_pipeline(
            name="cifar10",
            data_dir="unused",
            batch_size=4,
            shuffle_buffer_size=4,
            seed=0,
            use_basic_augmentation=False,
            augmentation_recipe="none",
        )
        batch = next(
            iter(
                dataset,
            )
        )

        assert batch["labels"].shape == (4,)
        np.testing.assert_allclose(
            batch["raw_images"].numpy(),
            batch["images"].numpy(),
            rtol=0.0,
            atol=0.0,
        )

    def test_get_cifar10_preprocessor(self) -> None:
        """Verify that get cifar10 preprocessor."""
        preprocessor = get_preprocessor("cifar10")

        assert preprocessor is not None

    def test_get_cifar100_preprocessor(self) -> None:
        """Verify that get cifar100 preprocessor."""
        preprocessor = get_preprocessor("cifar100")

        assert preprocessor is not None

    @pytest.mark.parametrize(
        ("dataset_name", "expected_mean", "expected_std"),
        [
            (
                "cifar10",
                [0.4914, 0.4822, 0.4465],
                [0.2470, 0.2435, 0.2616],
            ),
            (
                "cifar100",
                [0.5071, 0.4867, 0.4408],
                [0.2675, 0.2565, 0.2761],
            ),
        ],
    )
    def test_cifar_normalization_stats_are_dataset_specific(
        self,
        dataset_name: str,
        expected_mean: list[float],
        expected_std: list[float],
    ) -> None:
        """Verify that each CIFAR dataset uses its own channel statistics."""
        mean, std = get_normalization_stats(
            dataset_name,
        )

        assert mean.numpy().tolist() == pytest.approx(expected_mean)
        assert std.numpy().tolist() == pytest.approx(expected_std)

    @pytest.mark.parametrize(
        ("dataset_name", "num_classes", "image_size"),
        [
            ("svhn_cropped", 10, 32),
            ("stl10", 10, 96),
            ("oxford_iiit_pet", 37, 224),
            ("cars196", 196, 224),
            ("imagenet100", 100, 224),
            ("caltech_birds2011", 200, 224),
        ],
    )
    def test_get_extra_supported_dataset_metadata(
        self,
        dataset_name: str,
        num_classes: int,
        image_size: int,
    ) -> None:
        """Verify metadata for the extra supported TFDS datasets."""
        metadata = get_metadata(dataset_name)

        assert metadata.num_classes == num_classes
        assert metadata.image_size == image_size
        assert metadata.channels == 3

    @pytest.mark.parametrize(
        ("dataset_name", "input_shape", "image_size"),
        [
            ("svhn_cropped", (32, 32, 3), 32),
            ("stl10", (96, 96, 3), 96),
            ("oxford_iiit_pet", (180, 240, 3), 224),
            ("cars196", (240, 320, 3), 224),
            ("imagenet100", (240, 320, 3), 224),
            ("caltech_birds2011", (256, 384, 3), 224),
        ],
    )
    def test_extra_supported_dataset_preprocess_shape(
        self,
        dataset_name: str,
        input_shape: tuple[int, int, int],
        image_size: int,
    ) -> None:
        """Verify preprocessing shape for the extra supported TFDS datasets."""
        preprocessor = get_preprocessor(dataset_name)
        example = {
            "image": tf.zeros(input_shape, dtype=tf.uint8),
            "label": tf.constant(1, dtype=tf.int64),
        }

        image, label = preprocessor(
            example,
            False,
        )

        assert image.shape == (image_size, image_size, 3)
        assert label.dtype == tf.int64

    def test_imagenet100_train_crop_uses_original_aspect_ratio(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify ImageNet augmentation receives the variable-size source image."""
        observed_shape = None

        def fake_apply_augmentation_recipe(
            image: tf.Tensor,
            image_size: int,
            use_basic_augmentation: bool,
            augmentation_recipe: str | None,
        ) -> tf.Tensor:
            """Capture the image shape immediately before train augmentation."""
            nonlocal observed_shape
            del use_basic_augmentation
            assert augmentation_recipe == "imagenet"
            observed_shape = tuple(
                image.shape,
            )
            return tf.image.resize(
                image,
                [
                    image_size,
                    image_size,
                ],
            )

        monkeypatch.setattr(
            tfds_image_preprocessor,
            "apply_augmentation_recipe",
            fake_apply_augmentation_recipe,
        )
        preprocessor = get_preprocessor(
            "imagenet100",
        )
        preprocessor(
            {
                "image": tf.zeros(
                    (
                        240,
                        320,
                        3,
                    ),
                    dtype=tf.uint8,
                ),
                "label": tf.constant(
                    1,
                    dtype=tf.int64,
                ),
            },
            False,
            "imagenet",
        )

        assert observed_shape == (
            240,
            320,
            3,
        )

    def test_cars196_fine_grained_crop_uses_original_aspect_ratio(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify Cars train crops are sampled before square resizing."""
        observed_shape = None

        def fake_apply_augmentation_recipe(
            image: tf.Tensor,
            image_size: int,
            use_basic_augmentation: bool,
            augmentation_recipe: str | None,
        ) -> tf.Tensor:
            """Capture the variable-size image passed to augmentation."""
            nonlocal observed_shape
            del use_basic_augmentation
            assert augmentation_recipe == "fine_grained"
            observed_shape = tuple(
                image.shape,
            )
            return tf.image.resize(
                image,
                [
                    image_size,
                    image_size,
                ],
            )

        monkeypatch.setattr(
            tfds_image_preprocessor,
            "apply_augmentation_recipe",
            fake_apply_augmentation_recipe,
        )
        preprocessor = get_preprocessor(
            "cars196",
        )
        preprocessor(
            {
                "image": tf.zeros(
                    (
                        240,
                        320,
                        3,
                    ),
                    dtype=tf.uint8,
                ),
                "label": tf.constant(
                    1,
                    dtype=tf.int64,
                ),
            },
            False,
            "fine_grained",
        )

        assert observed_shape == (
            240,
            320,
            3,
        )

    def test_cars196_eval_uses_aspect_preserving_center_crop(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify Cars validation/test preprocessing uses a center crop."""
        observed_shape = None

        def fake_center_crop(
            image: tf.Tensor,
            image_size: int,
            resize_size: int = 256,
        ) -> tf.Tensor:
            """Capture the source geometry passed to evaluation resize."""
            nonlocal observed_shape
            del resize_size
            observed_shape = tuple(
                image.shape,
            )
            return tf.image.resize(
                image,
                [
                    image_size,
                    image_size,
                ],
            )

        monkeypatch.setattr(
            tfds_image_preprocessor,
            "resize_shorter_side_and_center_crop",
            fake_center_crop,
        )
        preprocessor = get_preprocessor(
            "cars196",
        )
        image, _ = preprocessor(
            {
                "image": tf.zeros(
                    (
                        240,
                        320,
                        3,
                    ),
                    dtype=tf.uint8,
                ),
                "label": tf.constant(
                    1,
                    dtype=tf.int64,
                ),
            },
            False,
            None,
        )

        assert observed_shape == (
            240,
            320,
            3,
        )
        assert image.shape == (
            224,
            224,
            3,
        )

    def test_imagenet100_eval_preserves_aspect_ratio_before_center_crop(
        self,
    ) -> None:
        """Verify ImageNet evaluation crops the center without square distortion."""
        horizontal_gradient = tf.linspace(
            0.0,
            1.0,
            320,
        )
        image = tf.tile(
            horizontal_gradient[
                tf.newaxis,
                :,
                tf.newaxis,
            ],
            [
                240,
                1,
                3,
            ],
        )

        cropped = tfds_image_preprocessor.resize_shorter_side_and_center_crop(
            image=image,
            image_size=224,
        )

        assert cropped.shape == (
            224,
            224,
            3,
        )
        assert float(
            tf.reduce_mean(
                cropped[:, 0, :],
            ).numpy()
        ) > 0.1
        assert float(
            tf.reduce_mean(
                cropped[:, -1, :],
            ).numpy()
        ) < 0.9

    def test_cub_augmentation_recipe_returns_bounded_image(self) -> None:
        """Verify that the CUB recipe preserves a valid fixed-size image."""
        tf.random.set_seed(
            0,
        )
        preprocessor = get_preprocessor(
            "caltech_birds2011",
        )
        example = {
            "image": tf.ones(
                (
                    320,
                    280,
                    3,
                ),
                dtype=tf.uint8,
            ) * 127,
            "label": tf.constant(
                1,
                dtype=tf.int64,
            ),
        }

        image, label = preprocessor(
            example,
            False,
            "cub",
        )

        assert image.shape == (
            224,
            224,
            3,
        )
        assert label.dtype == tf.int64
        assert bool(
            tf.reduce_all(
                tf.math.is_finite(
                    image,
                )
            )
        )

    def test_unknown_preprocessor_raises_error(self) -> None:
        """Verify that unknown preprocessor raises error."""
        with pytest.raises(ValueError):
            get_preprocessor("unknown_dataset")

    def test_tiny_imagenet_preprocessor_can_keep_zero_to_one_pixels(self) -> None:
        """Verify that Tiny ImageNet can skip ImageNet channel normalization."""
        preprocessor = get_preprocessor(
            "tiny_imagenet",
            tiny_imagenet_normalization="none",
        )
        example = {
            "image": tf.ones((64, 64, 3), dtype=tf.uint8) * 255,
            "label": tf.constant(7, dtype=tf.int64),
        }

        image, label = preprocessor(
            example,
            False,
        )

        assert image.shape == (64, 64, 3)
        assert label.dtype == tf.int64
        assert float(tf.reduce_min(image).numpy()) == pytest.approx(1.0)
        assert float(tf.reduce_max(image).numpy()) == pytest.approx(1.0)

    def test_tiny_imagenet_preprocessor_rejects_unknown_normalization(
        self,
    ) -> None:
        """Verify that Tiny ImageNet rejects unknown normalization modes."""
        preprocessor = get_preprocessor(
            "tiny_imagenet",
            tiny_imagenet_normalization="bad_mode",
        )
        example = {
            "image": tf.zeros((64, 64, 3), dtype=tf.uint8),
            "label": tf.constant(0, dtype=tf.int64),
        }

        with pytest.raises(ValueError):
            preprocessor(
                example,
                False,
            )

    def test_load_cifar10_train_dataset(self) -> None:
        """Verify that load cifar10 train dataset."""
        train_ds = load_train_dataset(
            name="cifar10",
            data_dir="./data",
        )

        assert train_ds is not None

    def test_load_cifar10_test_dataset(self) -> None:
        """Verify that load cifar10 test dataset."""
        test_ds = load_test_dataset(
            name="cifar10",
            data_dir="./data",
        )

        assert test_ds is not None

    def test_tfds_train_loader_forwards_reproducible_shuffle_seed(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify that TFDS file shuffling receives the data seed."""
        calls = []
        sentinel_dataset = object()

        def fake_tfds_load(**kwargs):
            calls.append(
                kwargs,
            )
            return sentinel_dataset

        monkeypatch.setattr(
            tfds,
            "load",
            fake_tfds_load,
        )

        train_ds = load_train_dataset(
            name="cifar10",
            data_dir="./data",
            shuffle_files=True,
            seed=37,
        )

        assert train_ds is sentinel_dataset
        assert calls[0]["shuffle_files"] is True
        assert calls[0]["read_config"].shuffle_seed == 37
        assert (
            calls[0]["read_config"].shuffle_reshuffle_each_iteration
            is True
        )

    def test_tfds_train_loader_can_forbid_worker_downloads(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify that prepared-dataset workers can use read-only TFDS."""
        calls = []
        sentinel_dataset = object()

        def fake_tfds_load(**kwargs):
            calls.append(kwargs)
            return sentinel_dataset

        monkeypatch.setattr(tfds, "load", fake_tfds_load)

        train_ds = load_train_dataset(
            name="caltech_birds2011",
            data_dir="./data",
            shuffle_files=False,
            download=False,
        )

        assert train_ds is sentinel_dataset
        assert calls[0]["download"] is False

    @pytest.mark.parametrize(
        "dataset_name",
        [
            "svhn_cropped",
            "stl10",
            "oxford_iiit_pet",
            "caltech_birds2011",
        ],
    )
    def test_tfds_image_train_and_test_splits(
        self,
        dataset_name: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify that supported TFDS image datasets use train/test splits."""
        calls = []
        sentinel_dataset = object()

        def fake_tfds_load(**kwargs):
            calls.append(kwargs)
            return sentinel_dataset

        monkeypatch.setattr(
            tfds,
            "load",
            fake_tfds_load,
        )

        train_ds = load_train_dataset(
            name=dataset_name,
            data_dir="./data",
            shuffle_files=False,
        )
        test_ds = load_test_dataset(
            name=dataset_name,
            data_dir="./data",
        )

        assert train_ds is sentinel_dataset
        assert test_ds is sentinel_dataset
        assert calls[0]["name"] == dataset_name
        assert calls[0]["split"] == "train"
        assert calls[1]["name"] == dataset_name
        assert calls[1]["split"] == "test"

    def test_cars196_train_and_test_load_from_local_class_folders(
        self,
        tmp_path,
    ) -> None:
        """Verify that Cars196 avoids the broken TFDS download URL."""
        root = tmp_path / "cars196"
        encoded_image = tf.io.encode_jpeg(
            tf.zeros(
                (
                    4,
                    4,
                    3,
                ),
                dtype=tf.uint8,
            )
        ).numpy()

        for split in (
            "train",
            "test",
        ):
            for class_index in range(196):
                class_dir = root / split / f"class_{class_index:03d}"
                class_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                if class_index == 0:
                    (
                        class_dir / f"{split}_{class_index:03d}.jpg"
                    ).write_bytes(
                        encoded_image,
                    )

        train_ds = load_train_dataset(
            name="cars196",
            data_dir=str(tmp_path),
        )
        test_ds = load_test_dataset(
            name="cars196",
            data_dir=str(tmp_path),
        )

        train_example = next(iter(train_ds))
        test_example = next(iter(test_ds))

        assert train_example["image"].shape == (4, 4, 3)
        assert test_example["image"].shape == (4, 4, 3)
        assert int(train_example["label"].numpy()) == 0
        assert int(test_example["label"].numpy()) == 0

    def test_imagenet100_train_and_val_load_from_local_class_folders(
        self,
        monkeypatch,
        tmp_path,
    ) -> None:
        """Verify that ImageNet100 uses local train/val class folders."""
        import allthemix.data.datasets.loader as loader_module
        from allthemix.data.datasets.imagenet100 import CMC_IMAGENET100_CLASS_NAMES

        monkeypatch.setattr(
            loader_module,
            "CMC_IMAGENET100_TRAIN_EXAMPLES",
            100,
        )
        monkeypatch.setattr(
            loader_module,
            "CMC_IMAGENET100_VAL_EXAMPLES",
            100,
        )
        root = tmp_path / "imagenet100"
        encoded_image = tf.io.encode_jpeg(
            tf.zeros(
                (
                    4,
                    4,
                    3,
                ),
                dtype=tf.uint8,
            )
        ).numpy()

        for split in (
            "train",
            "val",
        ):
            for class_index, class_name in enumerate(
                CMC_IMAGENET100_CLASS_NAMES,
            ):
                class_dir = root / split / class_name
                class_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                (
                    class_dir / f"{split}_{class_index:03d}.JPEG"
                ).write_bytes(
                    encoded_image,
                )

        train_ds = load_train_dataset(
            name="imagenet100",
            data_dir=str(tmp_path),
        )
        test_ds = load_test_dataset(
            name="imagenet100",
            data_dir=str(tmp_path),
        )

        train_example = next(iter(train_ds))
        test_example = next(iter(test_ds))

        assert train_example["image"].shape == (4, 4, 3)
        assert test_example["image"].shape == (4, 4, 3)
        assert int(train_example["label"].numpy()) == 0
        assert int(test_example["label"].numpy()) == 0

    def test_cifar10_pipeline_batch_shape_without_basic_aug(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify that cifar10 pipeline batch shape without basic aug."""
        _patch_cifar10_sources(monkeypatch)
        train_ds, test_ds = build_dataset_pipeline(
            name="cifar10",
            data_dir="./data",
            batch_size=128,
            shuffle_buffer_size=10_000,
            drop_remainder=True,
            use_basic_augmentation=False,
        )

        train_images, train_labels = next(iter(tfds.as_numpy(train_ds)))
        test_images, test_labels = next(iter(tfds.as_numpy(test_ds)))

        assert train_images.shape == (128, 32, 32, 3)
        assert train_labels.shape == (128,)

        assert test_images.shape[1:] == (32, 32, 3)
        assert len(test_labels.shape) == 1

    def test_cifar10_pipeline_batch_shape_with_basic_aug(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify that cifar10 pipeline batch shape with basic aug."""
        _patch_cifar10_sources(monkeypatch)
        train_ds, test_ds = build_dataset_pipeline(
            name="cifar10",
            data_dir="./data",
            batch_size=128,
            shuffle_buffer_size=10_000,
            drop_remainder=True,
            use_basic_augmentation=True,
        )

        train_images, train_labels = next(iter(tfds.as_numpy(train_ds)))
        test_images, test_labels = next(iter(tfds.as_numpy(test_ds)))

        assert train_images.shape == (128, 32, 32, 3)
        assert train_labels.shape == (128,)

        assert test_images.shape[1:] == (32, 32, 3)
        assert len(test_labels.shape) == 1

    def test_get_cifar10_metadata(self) -> None:
        """Verify that get cifar10 metadata."""
        metadata = get_metadata("cifar10")

        assert metadata.num_classes == 10
        assert metadata.image_size == 32
        assert metadata.channels == 3


class TestOfficialEvaluationValidationProtocol:
    """Membership guarantees for official-eval-sourced validation."""

    @staticmethod
    def _id_encoded_source(
        *,
        first_id: int,
        example_count: int = 40,
    ) -> tf.data.Dataset:
        ids = tf.range(
            first_id,
            first_id + example_count,
            dtype=tf.int32,
        )
        images = tf.cast(
            tf.tile(
                tf.reshape(
                    ids,
                    (example_count, 1, 1, 1),
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

    @staticmethod
    def _decode_classifier_dataset(
        dataset: tf.data.Dataset,
    ) -> tuple[set[int], dict[int, int]]:
        mean, std = get_normalization_stats(
            "cifar10",
        )
        mean_0 = float(mean.numpy()[0])
        std_0 = float(std.numpy()[0])
        ids: set[int] = set()
        label_counts = {
            label: 0
            for label in range(10)
        }

        for images, labels in tfds.as_numpy(dataset):
            for value, label in zip(
                images[:, 0, 0, 0],
                labels,
                strict=True,
            ):
                ids.add(
                    int(
                        round(
                            (value * std_0 + mean_0) * 255.0,
                        )
                    )
                )
                label_counts[int(label)] += 1

        return ids, label_counts

    def _patch_sources(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import allthemix.data.pipeline as pipeline_module

        train_source = self._id_encoded_source(
            first_id=0,
        )
        official_eval_source = self._id_encoded_source(
            first_id=100,
        )
        monkeypatch.setattr(
            pipeline_module,
            "load_train_dataset",
            lambda **_kwargs: train_source,
        )
        monkeypatch.setattr(
            pipeline_module,
            "load_test_dataset",
            lambda **_kwargs: official_eval_source,
        )

    def test_train_val_and_final_membership_is_exact_and_disjoint(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Use full train and complementary stratified official-eval folds."""
        self._patch_sources(
            monkeypatch,
        )
        train_ds, validation_ds = build_dataset_pipeline(
            name="cifar10",
            data_dir="unused",
            batch_size=7,
            shuffle_buffer_size=40,
            drop_remainder=False,
            use_basic_augmentation=False,
            validation_split=0.5,
            eval_on_test=False,
            seed=23,
            deterministic_data=True,
            val_source="test",
        )
        final_ds = build_test_pipeline(
            name="cifar10",
            data_dir="unused",
            batch_size=7,
            validation_split=0.5,
            val_source="test",
        )

        train_ids, train_labels = self._decode_classifier_dataset(train_ds)
        validation_ids, validation_labels = (
            self._decode_classifier_dataset(validation_ds)
        )
        final_ids, final_labels = self._decode_classifier_dataset(final_ds)

        assert train_ids == set(range(40))
        assert train_labels == {
            label: 4
            for label in range(10)
        }
        assert validation_ids.isdisjoint(final_ids)
        assert validation_ids | final_ids == set(range(100, 140))
        assert validation_labels == {
            label: 2
            for label in range(10)
        }
        assert final_labels == {
            label: 2
            for label in range(10)
        }

    def test_ifaugnet_raw_and_meta_streams_follow_the_same_sources(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Keep IF-AugNet train full and meta-validation on official eval."""
        self._patch_sources(
            monkeypatch,
        )
        raw_train_ds = build_raw_augmented_train_pipeline(
            name="cifar10",
            data_dir="unused",
            batch_size=8,
            shuffle_buffer_size=40,
            seed=7,
            drop_remainder=False,
            use_basic_augmentation=False,
            validation_split=0.5,
            val_source="test",
        )
        meta_validation_ds = build_meta_validation_pipeline(
            name="cifar10",
            data_dir="unused",
            batch_size=8,
            validation_split=0.5,
            shuffle_buffer_size=20,
            seed=11,
            repeat=False,
            drop_remainder=False,
            val_source="test",
        )

        raw_ids: set[int] = set()
        mean, std = get_normalization_stats(
            "cifar10",
        )
        mean_0 = float(mean.numpy()[0])
        std_0 = float(std.numpy()[0])
        for batch in tfds.as_numpy(raw_train_ds):
            for value in batch["raw_images"][:, 0, 0, 0]:
                raw_ids.add(
                    int(
                        round(
                            (value * std_0 + mean_0) * 255.0,
                        )
                    )
                )

        meta_ids, meta_labels = self._decode_classifier_dataset(
            meta_validation_ds,
        )

        assert raw_ids == set(range(40))
        assert meta_ids.issubset(set(range(100, 140)))
        assert meta_labels == {
            label: 2
            for label in range(10)
        }

    def test_unknown_validation_source_fails_before_data_loading(
        self,
    ) -> None:
        """Reject a misspelled source instead of silently using train."""
        with pytest.raises(ValueError, match="val_source"):
            build_train_pipeline(
                name="cifar10",
                data_dir="unused",
                batch_size=4,
                shuffle_buffer_size=4,
                validation_split=0.1,
                val_source="official",
            )


class TestDebugTrainSource:
    """Leakage-diagnostic train-source override membership guarantees."""

    @staticmethod
    def _id_encoded_source() -> tf.data.Dataset:
        """Build a labeled source whose pixel value identifies each example."""
        example_count = 256
        ids = tf.range(
            example_count,
            dtype=tf.int32,
        )
        images = tf.cast(
            tf.tile(
                tf.reshape(
                    ids,
                    (example_count, 1, 1, 1),
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

    @staticmethod
    def _decode_ids(
        dataset: tf.data.Dataset,
    ) -> set[int]:
        """Invert cifar10 normalization to recover example identities."""
        mean, std = get_normalization_stats(
            "cifar10",
        )
        mean_0 = float(mean.numpy()[0])
        std_0 = float(std.numpy()[0])
        ids: set[int] = set()

        for images, _labels in tfds.as_numpy(dataset):
            pixel_values = images[:, 0, 0, 0]
            for value in pixel_values:
                ids.add(
                    int(
                        round(
                            (value * std_0 + mean_0) * 255.0,
                        )
                    )
                )

        return ids

    def _patch_source(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import allthemix.data.pipeline as pipeline_module

        source = self._id_encoded_source()
        monkeypatch.setattr(
            pipeline_module,
            "load_train_dataset",
            lambda **_kwargs: source,
        )

    def test_val_only_trains_on_exactly_the_validation_partition(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify val_only train membership equals the eval-side val split."""
        self._patch_source(monkeypatch)
        train_ds = build_train_pipeline(
            name="cifar10",
            data_dir="unused",
            batch_size=1,
            shuffle_buffer_size=256,
            drop_remainder=False,
            use_basic_augmentation=False,
            validation_split=0.25,
            debug_train_source="val_only",
        )
        validation_ds = build_validation_pipeline(
            name="cifar10",
            data_dir="unused",
            batch_size=1,
            validation_split=0.25,
        )

        train_ids = self._decode_ids(train_ds)
        validation_ids = self._decode_ids(validation_ds)
        expected_count = count_class_stratified_split_examples(
            class_counts=(26, 26, 26, 26, 26, 26, 25, 25, 25, 25),
            validation_split=0.25,
            keep_validation=True,
        )

        assert train_ids == validation_ids
        assert len(train_ids) == expected_count

    def test_train_plus_val_consumes_every_source_example(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify train_plus_val keeps train and validation examples."""
        self._patch_source(monkeypatch)
        train_ds = build_train_pipeline(
            name="cifar10",
            data_dir="unused",
            batch_size=1,
            shuffle_buffer_size=256,
            drop_remainder=False,
            use_basic_augmentation=False,
            validation_split=0.25,
            debug_train_source="train_plus_val",
        )

        assert self._decode_ids(train_ds) == set(range(256))

    def test_default_mode_stays_disjoint_from_the_validation_partition(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify the honest path never sees validation examples."""
        self._patch_source(monkeypatch)
        train_ds = build_train_pipeline(
            name="cifar10",
            data_dir="unused",
            batch_size=1,
            shuffle_buffer_size=256,
            drop_remainder=False,
            use_basic_augmentation=False,
            validation_split=0.25,
            debug_train_source="none",
        )
        validation_ds = build_validation_pipeline(
            name="cifar10",
            data_dir="unused",
            batch_size=1,
            validation_split=0.25,
        )

        train_ids = self._decode_ids(train_ds)
        validation_ids = self._decode_ids(validation_ds)

        assert train_ids.isdisjoint(validation_ids)
        assert train_ids | validation_ids == set(range(256))

    def test_debug_train_source_requires_validation_split(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify the override refuses to run without a held-out split."""
        self._patch_source(monkeypatch)

        with pytest.raises(ValueError, match="validation_split"):
            build_train_pipeline(
                name="cifar10",
                data_dir="unused",
                batch_size=1,
                shuffle_buffer_size=256,
                validation_split=0.0,
                debug_train_source="val_only",
            )

    def test_debug_train_source_rejects_unknown_mode(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify unknown override modes fail fast."""
        self._patch_source(monkeypatch)

        with pytest.raises(ValueError, match="debug_train_source"):
            build_train_pipeline(
                name="cifar10",
                data_dir="unused",
                batch_size=1,
                shuffle_buffer_size=256,
                validation_split=0.25,
                debug_train_source="validation",
            )

    def test_get_cifar100_metadata(self) -> None:
        """Verify that get cifar100 metadata."""
        metadata = get_metadata("cifar100")

        assert metadata.num_classes == 100
        assert metadata.image_size == 32
        assert metadata.channels == 3
