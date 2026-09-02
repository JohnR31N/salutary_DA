from __future__ import annotations

import logging

import tensorflow as tf

from allthemix.data.datasets.loader import load_test_dataset, load_train_dataset
from allthemix.data.preprocessors.selector import get_metadata, get_preprocessor
from allthemix.data.splits import (
    resolve_training_validation_split,
    split_train_validation_dataset,
    subset_train_dataset,
    validate_validation_source,
)
from allthemix.data.utils.random import (
    apply_dataset_determinism,
    attach_random_seed_stream,
    make_stateless_seed,
)

logger = logging.getLogger(__name__)

# Compose dataset loaders with preprocessing, batching, and prefetching.


def _project_classifier_fields(
    example: dict[str, tf.Tensor],
) -> dict[str, tf.Tensor]:
    """Keep the common raw fields required by every image preprocessor."""
    return {
        "image": example["image"],
        "label": tf.cast(
            example["label"],
            tf.int64,
        ),
    }


def _interleave_datasets_by_cardinality(
    original_ds: tf.data.Dataset,
    generated_ds: tf.data.Dataset,
    original_example_count: int,
    generated_example_count: int,
) -> tf.data.Dataset:
    """Evenly interleave two finite datasets while consuming each once."""
    if original_example_count < 1 or generated_example_count < 1:
        raise ValueError(
            "Append-mode dataset counts must both be positive. "
            f"Got original={original_example_count}, "
            f"generated={generated_example_count}."
        )

    total_example_count = (
        original_example_count + generated_example_count
    )
    original_count = tf.constant(
        original_example_count,
        dtype=tf.int64,
    )
    total_count = tf.constant(
        total_example_count,
        dtype=tf.int64,
    )

    def choose_dataset(
        index: tf.Tensor,
    ) -> tf.Tensor:
        """Return a balanced selector with exactly the requested counts."""
        is_original = tf.math.floormod(
            index * original_count,
            total_count,
        ) < original_count

        return tf.where(
            is_original,
            tf.zeros(
                (),
                dtype=tf.int64,
            ),
            tf.ones(
                (),
                dtype=tf.int64,
            ),
        )

    choice_ds = tf.data.Dataset.range(
        total_example_count,
    ).map(
        choose_dataset,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )

    return tf.data.Dataset.choose_from_datasets(
        datasets=[
            original_ds,
            generated_ds,
        ],
        choice_dataset=choice_ds,
        stop_on_empty_dataset=True,
    )


def build_train_pipeline(
    name: str,
    data_dir: str,
    batch_size: int,
    shuffle_buffer_size: int,
    drop_remainder: bool = True,
    use_basic_augmentation: bool = False,
    augmentation_recipe: str | None = None,
    validation_split: float = 0.0,
    tiny_imagenet_normalization: str = "imagenet",
    train_manifest_path: str = "",
    train_manifest_kind: str = "diffusemix",
    train_manifest_mode: str = "replace",
    train_original_example_count: int | None = None,
    train_manifest_example_count: int | None = None,
    train_manifest_prevalidated: bool = False,
    train_replacement_probability: float = 1.0,
    train_subset_fraction: float = 1.0,
    seed: int = 0,
    deterministic_data: bool = True,
    debug_train_source: str = "none",
    val_source: str = "train",
) -> tf.data.Dataset:
    """Build the training dataset pipeline."""
    validate_validation_source(
        val_source=val_source,
    )
    if val_source == "test" and train_subset_fraction < 1.0:
        raise ValueError(
            "val_source=test requires the official training split in full; "
            "train_subset_fraction must be 1."
        )
    if val_source == "test" and debug_train_source != "none":
        raise ValueError(
            "debug_train_source is a legacy train-split diagnostic and is "
            "not supported with val_source=test."
        )
    validation_split = resolve_training_validation_split(
        validation_split=validation_split,
        val_source=val_source,
    )
    preprocess_example = get_preprocessor(
        name,
        tiny_imagenet_normalization=tiny_imagenet_normalization,
    )

    if debug_train_source not in {
        "none",
        "val_only",
        "train_plus_val",
    }:
        raise ValueError(
            "debug_train_source must be none, val_only, or train_plus_val. "
            f"Got {debug_train_source!r}."
        )

    if debug_train_source != "none":
        # Diagnostic leakage arms: training deliberately consumes held-out
        # validation examples while the eval-side validation pipeline stays
        # untouched, so their membership must come from the identical
        # deterministic stratified split.
        if validation_split <= 0.0:
            raise ValueError(
                "debug_train_source requires validation_split > 0."
            )
        if train_manifest_path:
            raise ValueError(
                "debug_train_source is not supported with an offline "
                "generation manifest."
            )
        if train_subset_fraction < 1.0:
            raise ValueError(
                "debug_train_source is not supported with "
                "train_subset_fraction < 1.0."
            )

    if train_subset_fraction < 1.0 and train_manifest_path:
        raise ValueError(
            "train_subset_fraction < 1.0 is not supported together with an "
            "offline generation manifest."
        )

    if train_subset_fraction < 1.0 and not deterministic_data:
        raise ValueError(
            "train_subset_fraction < 1.0 requires deterministic_data=true "
            "so subset membership is identical across pipelines."
        )

    manifest_mode = train_manifest_mode.lower()
    if manifest_mode not in {
        "replace",
        "append",
        "sample",
    }:
        raise ValueError(
            "train_manifest_mode must be replace, append, or sample. "
            f"Got {train_manifest_mode}."
        )

    train_ds = None
    source_replacer = None
    if not train_manifest_path or manifest_mode in {"append", "sample"}:
        train_ds = load_train_dataset(
            name=name,
            data_dir=data_dir,
            shuffle_files=(
                validation_split == 0.0
                and manifest_mode != "sample"
                and train_subset_fraction >= 1.0
            ),
            seed=seed if deterministic_data else None,
        )

        if manifest_mode == "sample":
            from allthemix.data.utils.source_replacement import (
                attach_source_indices,
            )

            train_ds = attach_source_indices(train_ds)

        train_ds = subset_train_dataset(
            dataset=train_ds,
            train_subset_fraction=train_subset_fraction,
        )
        if debug_train_source == "val_only":
            train_ds = split_train_validation_dataset(
                dataset=train_ds,
                validation_split=validation_split,
                keep_validation=True,
            )
        elif debug_train_source == "train_plus_val":
            pass
        else:
            train_ds = split_train_validation_dataset(
                dataset=train_ds,
                validation_split=validation_split,
                keep_validation=False,
            )

    if train_manifest_path:
        # Imported lazily so ordinary experiments do not depend on the
        # competitor package or parse an offline-generation manifest.
        manifest_kind = train_manifest_kind.lower()
        if manifest_kind == "diffusemix":
            from allthemix.competitors.diffusemix.manifest import (
                load_manifest_dataset,
            )
        elif manifest_kind == "alia":
            from allthemix.competitors.alia.manifest import (
                load_manifest_dataset,
            )
        elif manifest_kind == "saspa":
            if manifest_mode != "sample":
                raise ValueError(
                    "SaSPA uses source-aligned sample mode, not append or "
                    "replace."
                )
            from allthemix.competitors.saspa.manifest import (
                replacement_catalog,
            )
            from allthemix.data.utils.source_replacement import (
                build_source_replacer,
            )

            source_indices, generated_paths, source_labels = (
                replacement_catalog(
                    manifest_path=train_manifest_path,
                    check_images=not train_manifest_prevalidated,
                )
            )
            source_replacer = build_source_replacer(
                source_indices=source_indices,
                generated_paths=generated_paths,
                source_labels=source_labels,
                probability=train_replacement_probability,
            )
            load_manifest_dataset = None
        else:
            raise ValueError(
                "train_manifest_kind must be diffusemix, alia, or saspa. "
                f"Got {train_manifest_kind!r}."
            )

        if manifest_mode != "sample":
            generated_ds = load_manifest_dataset(
                manifest_path=train_manifest_path,
                image_size=get_metadata(
                    name,
                ).image_size,
                check_images=not train_manifest_prevalidated,
            )
            if train_ds is None:
                train_ds = generated_ds
            else:
                if (
                    train_original_example_count is None
                    or train_manifest_example_count is None
                ):
                    raise ValueError(
                        "append mode requires exact original and manifest "
                        "example counts."
                    )
                train_ds = train_ds.map(
                    _project_classifier_fields,
                    num_parallel_calls=tf.data.AUTOTUNE,
                    deterministic=True,
                )
                generated_ds = generated_ds.map(
                    _project_classifier_fields,
                    num_parallel_calls=tf.data.AUTOTUNE,
                    deterministic=True,
                )
                train_ds = _interleave_datasets_by_cardinality(
                    original_ds=train_ds,
                    generated_ds=generated_ds,
                    original_example_count=train_original_example_count,
                    generated_example_count=train_manifest_example_count,
                )

    if train_ds is None:
        raise RuntimeError("Training dataset was not initialized.")

    train_ds = train_ds.shuffle(
        buffer_size=shuffle_buffer_size,
        seed=seed if deterministic_data else None,
        reshuffle_each_iteration=True,
    )

    if deterministic_data:
        train_ds = attach_random_seed_stream(
            dataset=train_ds,
            seed=seed,
        )
        def deterministic_preprocess(example, random_value):
            random_seed = make_stateless_seed(
                base_seed=seed,
                random_value=random_value,
            )
            if source_replacer is not None:
                example = source_replacer(example, random_seed)
                random_seed = tf.random.experimental.stateless_fold_in(
                    random_seed,
                    1,
                )

            return preprocess_example(
                example,
                use_basic_augmentation,
                augmentation_recipe,
                random_seed,
            )

        train_ds = train_ds.map(
            deterministic_preprocess,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True,
        )
    else:
        def random_preprocess(example):
            if source_replacer is not None:
                example = source_replacer(example)

            return preprocess_example(
                example,
                use_basic_augmentation,
                augmentation_recipe,
            )

        train_ds = train_ds.map(
            random_preprocess,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    train_ds = train_ds.batch(
        batch_size=batch_size,
        drop_remainder=drop_remainder,
    )

    train_ds = train_ds.prefetch(
        buffer_size=tf.data.AUTOTUNE,
    )

    return apply_dataset_determinism(
        dataset=train_ds,
        deterministic=deterministic_data,
    )


def build_validation_pipeline(
    name: str,
    data_dir: str,
    batch_size: int,
    validation_split: float,
    tiny_imagenet_normalization: str = "imagenet",
    train_subset_fraction: float = 1.0,
    val_source: str = "train",
) -> tf.data.Dataset:
    """Build the held-out validation pipeline.

    val_source="train" (default): carve ``validation_split`` of the
    training split (the legacy protocol). val_source="test": take a
    deterministic class-stratified ``validation_split`` fraction of the
    OFFICIAL evaluation split (test, or the official val where one
    exists); the complement stays sealed for final testing (see
    build_test_pipeline).
    """
    validate_validation_source(
        val_source=val_source,
    )
    if val_source == "test" and train_subset_fraction < 1.0:
        raise ValueError(
            "val_source=test requires train_subset_fraction=1."
        )

    preprocess_example = get_preprocessor(
        name,
        tiny_imagenet_normalization=tiny_imagenet_normalization,
    )

    if val_source == "test":
        validation_ds = load_test_dataset(
            name=name,
            data_dir=data_dir,
        )
    else:
        validation_ds = load_train_dataset(
            name=name,
            data_dir=data_dir,
            shuffle_files=False,
        )
        validation_ds = subset_train_dataset(
            dataset=validation_ds,
            train_subset_fraction=train_subset_fraction,
        )
    validation_ds = split_train_validation_dataset(
        dataset=validation_ds,
        validation_split=validation_split,
        keep_validation=True,
    )

    validation_ds = validation_ds.map(
        lambda example: preprocess_example(
            example,
            False,
            None,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    validation_ds = validation_ds.batch(
        batch_size=batch_size,
        drop_remainder=False,
    )

    validation_ds = validation_ds.prefetch(
        buffer_size=tf.data.AUTOTUNE,
    )

    return apply_dataset_determinism(
        dataset=validation_ds,
        deterministic=True,
    )


def build_raw_augmented_train_pipeline(
    name: str,
    data_dir: str,
    batch_size: int,
    shuffle_buffer_size: int,
    seed: int,
    drop_remainder: bool = True,
    use_basic_augmentation: bool = False,
    augmentation_recipe: str | None = None,
    validation_split: float = 0.0,
    tiny_imagenet_normalization: str = "imagenet",
    deterministic_data: bool = True,
    train_subset_fraction: float = 1.0,
    val_source: str = "train",
) -> tf.data.Dataset:
    """Build aligned raw/base-augmented views from each training example."""
    validate_validation_source(
        val_source=val_source,
    )
    if val_source == "test" and train_subset_fraction < 1.0:
        raise ValueError(
            "val_source=test requires train_subset_fraction=1."
        )
    validation_split = resolve_training_validation_split(
        validation_split=validation_split,
        val_source=val_source,
    )

    if train_subset_fraction < 1.0 and not deterministic_data:
        raise ValueError(
            "train_subset_fraction < 1.0 requires deterministic_data=true "
            "so subset membership is identical across pipelines."
        )

    preprocess_example = get_preprocessor(
        name,
        tiny_imagenet_normalization=tiny_imagenet_normalization,
    )
    train_ds = load_train_dataset(
        name=name,
        data_dir=data_dir,
        shuffle_files=False,
    )
    train_ds = subset_train_dataset(
        dataset=train_ds,
        train_subset_fraction=train_subset_fraction,
    )
    train_ds = split_train_validation_dataset(
        dataset=train_ds,
        validation_split=validation_split,
        keep_validation=False,
    )
    train_ds = train_ds.shuffle(
        buffer_size=shuffle_buffer_size,
        seed=seed if deterministic_data else None,
        reshuffle_each_iteration=True,
    )

    def preprocess_views(
        example,
        augmentation_seed=None,
    ):
        """Return two views whose label and source example are identical."""
        raw_image, label = preprocess_example(
            example,
            False,
            None,
        )
        augmented_image, _ = preprocess_example(
            example,
            use_basic_augmentation,
            augmentation_recipe,
            augmentation_seed,
        )

        return {
            "images": augmented_image,
            "labels": label,
            "raw_images": raw_image,
        }

    if deterministic_data:
        train_ds = attach_random_seed_stream(
            dataset=train_ds,
            seed=seed,
        )
        train_ds = train_ds.map(
            lambda example, random_value: preprocess_views(
                example,
                make_stateless_seed(
                    base_seed=seed,
                    random_value=random_value,
                ),
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True,
        )
    else:
        train_ds = train_ds.map(
            preprocess_views,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    train_ds = train_ds.batch(
        batch_size=batch_size,
        drop_remainder=drop_remainder,
    )

    train_ds = train_ds.prefetch(
        buffer_size=tf.data.AUTOTUNE,
    )

    return apply_dataset_determinism(
        dataset=train_ds,
        deterministic=deterministic_data,
    )


def build_meta_validation_pipeline(
    name: str,
    data_dir: str,
    batch_size: int,
    validation_split: float,
    shuffle_buffer_size: int,
    seed: int,
    tiny_imagenet_normalization: str = "imagenet",
    deterministic_data: bool = True,
    repeat: bool = True,
    drop_remainder: bool = True,
    train_subset_fraction: float = 1.0,
    val_source: str = "train",
) -> tf.data.Dataset:
    """Build a shuffled validation stream for validation-aware policies."""
    if validation_split <= 0.0:
        raise ValueError(
            "A meta-validation pipeline requires validation_split > 0."
        )

    validate_validation_source(
        val_source=val_source,
    )
    if val_source == "test" and train_subset_fraction < 1.0:
        raise ValueError(
            "val_source=test requires train_subset_fraction=1."
        )
    if train_subset_fraction < 1.0 and not deterministic_data:
        raise ValueError(
            "train_subset_fraction < 1.0 requires deterministic_data=true "
            "so subset membership is identical across pipelines."
        )

    preprocess_example = get_preprocessor(
        name,
        tiny_imagenet_normalization=tiny_imagenet_normalization,
    )
    if val_source == "test":
        validation_ds = load_test_dataset(
            name=name,
            data_dir=data_dir,
        )
    else:
        validation_ds = load_train_dataset(
            name=name,
            data_dir=data_dir,
            shuffle_files=False,
        )
        validation_ds = subset_train_dataset(
            dataset=validation_ds,
            train_subset_fraction=train_subset_fraction,
        )
    validation_ds = split_train_validation_dataset(
        dataset=validation_ds,
        validation_split=validation_split,
        keep_validation=True,
    )
    validation_ds = validation_ds.map(
        lambda example: preprocess_example(
            example,
            False,
            None,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    validation_ds = validation_ds.shuffle(
        buffer_size=shuffle_buffer_size,
        seed=seed if deterministic_data else None,
        reshuffle_each_iteration=True,
    )
    if repeat:
        validation_ds = validation_ds.repeat()
    validation_ds = validation_ds.batch(
        batch_size=batch_size,
        drop_remainder=drop_remainder,
    )
    validation_ds = validation_ds.prefetch(
        buffer_size=tf.data.AUTOTUNE,
    )

    return apply_dataset_determinism(
        dataset=validation_ds,
        deterministic=deterministic_data,
    )


def build_test_pipeline(
    name: str,
    data_dir: str,
    batch_size: int,
    tiny_imagenet_normalization: str = "imagenet",
    val_source: str = "train",
    validation_split: float = 0.0,
) -> tf.data.Dataset:
    """Build the evaluation dataset pipeline.

    Under val_source="test" with validation_split > 0, the returned
    pipeline is the SEALED complement of the official-eval-sourced
    validation fraction (disjoint by the same deterministic stratified
    bucketing), so the final test never overlaps the guidance val.
    """
    validate_validation_source(
        val_source=val_source,
    )
    preprocess_example = get_preprocessor(
        name,
        tiny_imagenet_normalization=tiny_imagenet_normalization,
    )

    test_ds = load_test_dataset(
        name=name,
        data_dir=data_dir,
    )

    if val_source == "test" and validation_split > 0.0:
        test_ds = split_train_validation_dataset(
            dataset=test_ds,
            validation_split=validation_split,
            keep_validation=False,
        )

    test_ds = test_ds.map(
        lambda example: preprocess_example(
            example,
            False,
            None,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    test_ds = test_ds.batch(
        batch_size=batch_size,
        drop_remainder=False,
    )

    test_ds = test_ds.prefetch(
        buffer_size=tf.data.AUTOTUNE,
    )

    return apply_dataset_determinism(
        dataset=test_ds,
        deterministic=True,
    )


def build_dataset_pipeline(
    name: str,
    data_dir: str,
    batch_size: int,
    shuffle_buffer_size: int,
    drop_remainder: bool = True,
    use_basic_augmentation: bool = False,
    augmentation_recipe: str | None = None,
    validation_split: float = 0.0,
    eval_on_test: bool = True,
    tiny_imagenet_normalization: str = "imagenet",
    train_manifest_path: str = "",
    train_manifest_kind: str = "diffusemix",
    train_manifest_mode: str = "replace",
    train_original_example_count: int | None = None,
    train_manifest_example_count: int | None = None,
    train_manifest_prevalidated: bool = False,
    train_replacement_probability: float = 1.0,
    train_subset_fraction: float = 1.0,
    seed: int = 0,
    deterministic_data: bool = True,
    debug_train_source: str = "none",
    val_source: str = "train",
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Build paired training and evaluation dataset pipelines."""
    train_ds = build_train_pipeline(
        name=name,
        data_dir=data_dir,
        batch_size=batch_size,
        shuffle_buffer_size=shuffle_buffer_size,
        drop_remainder=drop_remainder,
        use_basic_augmentation=use_basic_augmentation,
        augmentation_recipe=augmentation_recipe,
        validation_split=validation_split,
        tiny_imagenet_normalization=tiny_imagenet_normalization,
        train_manifest_path=train_manifest_path,
        train_manifest_kind=train_manifest_kind,
        train_manifest_mode=train_manifest_mode,
        train_original_example_count=train_original_example_count,
        train_manifest_example_count=train_manifest_example_count,
        train_manifest_prevalidated=train_manifest_prevalidated,
        train_replacement_probability=train_replacement_probability,
        train_subset_fraction=train_subset_fraction,
        seed=seed,
        deterministic_data=deterministic_data,
        debug_train_source=debug_train_source,
        val_source=val_source,
    )

    if validation_split > 0.0 and not eval_on_test:
        test_ds = build_validation_pipeline(
            name=name,
            data_dir=data_dir,
            batch_size=batch_size,
            validation_split=validation_split,
            tiny_imagenet_normalization=tiny_imagenet_normalization,
            train_subset_fraction=train_subset_fraction,
            val_source=val_source,
        )

    else:
        test_ds = build_test_pipeline(
            name=name,
            data_dir=data_dir,
            batch_size=batch_size,
            tiny_imagenet_normalization=tiny_imagenet_normalization,
            val_source=val_source,
            validation_split=validation_split,
        )

    return train_ds, test_ds


def build_select_half_validation_pipeline(
    validation_ds: tf.data.Dataset,
    *,
    select_split_fraction: float,
    data_seed: int,
    batch_size: int,
) -> tf.data.Dataset:
    """Subset a validation pipeline to its deterministic ``select`` half.

    Applies the identical seed (``data_seed + 401``), permutation, and
    ``gate_count = round(N * fraction)`` arithmetic as the validation
    gate/select val split, discards the gate fraction, and returns the
    select remainder rebatched. Baselines evaluated on this pipeline
    select checkpoints from bit-identical examples (and an identical
    selection-set size) as val-guided method arms, removing the
    selection-set-size confound from matched comparisons.
    """
    import numpy as np

    image_batches = []
    label_batches = []

    for batch in validation_ds:
        if isinstance(batch, dict):
            batch_images, batch_labels = batch["image"], batch["label"]
        else:
            batch_images, batch_labels = batch
        image_batches.append(np.asarray(batch_images, dtype=np.float32))
        label_batches.append(np.asarray(batch_labels, dtype=np.int32))

    if not image_batches:
        raise ValueError(
            "val_select_split_fraction: validation pipeline produced no "
            "examples."
        )
    images = np.concatenate(image_batches, axis=0)
    labels = np.concatenate(label_batches, axis=0)
    example_count = images.shape[0]
    gate_count = round(example_count * select_split_fraction)

    if gate_count < 1 or example_count - gate_count < 1:
        raise ValueError(
            f"val_select_split_fraction {select_split_fraction} leaves "
            f"an empty half of the {example_count}-example validation "
            f"partition."
        )
    permutation = np.random.default_rng(data_seed + 401).permutation(
        example_count
    )
    select_indices = permutation[gate_count:]
    logger.info(
        "Checkpoint-selection val subset | discarded gate: %d | select: %d",
        gate_count,
        len(select_indices),
    )

    return tf.data.Dataset.from_tensor_slices(
        (images[select_indices], labels[select_indices])
    ).batch(batch_size)
