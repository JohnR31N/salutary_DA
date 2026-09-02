"""Setup stages extracted from the training entry point."""

from __future__ import annotations

from dataclasses import dataclass

from allthemix.competitors.alia.manifest import (
    validate_manifest_for_training as validate_alia_manifest,
)
from allthemix.competitors.diffusemix.manifest import (
    validate_manifest_for_training as validate_diffusemix_manifest,
)
from allthemix.competitors.saspa.manifest import (
    validate_manifest_for_training as validate_saspa_manifest,
)
from allthemix.data.datasets.loader import load_train_dataset
from allthemix.data.pipeline import (
    build_dataset_pipeline,
    build_meta_validation_pipeline,
    build_select_half_validation_pipeline,
    build_test_pipeline,
)
from allthemix.data.splits import (
    count_class_stratified_split_examples,
    count_dataset_examples_by_class,
)
from allthemix.data.utils.cardinality import resolve_train_class_counts
from allthemix.utils.checkpoint import (
    build_checkpoint_dir,
    checkpoint_exists,
    restore_checkpoint,
    restore_matching_pretrained_checkpoint,
)
from allthemix.utils.experiment_logger import (
    build_output_path,
    build_run_name,
    write_csv_header,
)
from allthemix.utils.wandb_logger import WandbLogger


def restore_initial_state(
    args,
    state,
    metaaugment_context,
):
    """Apply --resume_checkpoint / --pretrained_checkpoint to a fresh state."""
    if args.resume_checkpoint:
        if not checkpoint_exists(args.resume_checkpoint):
            raise FileNotFoundError(
                f"Checkpoint path does not exist: {args.resume_checkpoint}"
            )

        if metaaugment_context is not None:
            checkpoint_template = metaaugment_context.checkpoint_state(
                task_state=state,
            )
            restored_checkpoint = restore_checkpoint(
                state=checkpoint_template,
                checkpoint_path=args.resume_checkpoint,
            )
            state = restored_checkpoint.task_state
            metaaugment_context.restore_method_state(
                restored_checkpoint.method_state,
            )
        else:
            state = restore_checkpoint(
                state=state,
                checkpoint_path=args.resume_checkpoint,
            )

        print(f"Restored checkpoint from: {args.resume_checkpoint}")
        print(f"Restored training step: {int(state.step)}")

    if args.pretrained_checkpoint:
        if not checkpoint_exists(args.pretrained_checkpoint):
            raise FileNotFoundError(
                f"Pretrained checkpoint path does not exist: "
                f"{args.pretrained_checkpoint}"
            )

        state, loaded_keys, skipped_keys = restore_matching_pretrained_checkpoint(
            state=state,
            checkpoint_path=args.pretrained_checkpoint,
        )

        print(f"Loaded pretrained checkpoint from: {args.pretrained_checkpoint}")
        print(
            "Pretrained leaves loaded: "
            f"{len(loaded_keys)} | skipped: {len(skipped_keys)}"
        )

    return state


DIFFUSEMIX_METHOD_NAMES = (
    "diffusemix",
    "diffuse_mix",
)

ALIA_METHOD_NAMES = (
    "alia",
)

SASPA_METHOD_NAMES = (
    "saspa",
)

def _count_original_diffusemix_examples(
    dataset: str,
    data_dir: str,
    num_classes: int,
    num_train_examples: int,
    validation_split: float,
    known_class_counts: tuple[int, ...] | None,
) -> int:
    """Count original examples that remain on the training side exactly."""
    class_counts = known_class_counts
    if class_counts is None:
        print(
            "Counting source labels to determine the exact offline "
            "append cardinality."
        )
        raw_train_ds = load_train_dataset(
            name=dataset,
            data_dir=data_dir,
            shuffle_files=False,
        )
        class_counts = count_dataset_examples_by_class(
            dataset=raw_train_ds,
            num_classes=num_classes,
        )

    if len(
        class_counts,
    ) != num_classes:
        raise ValueError(
            "Training class-count metadata does not match num_classes: "
            f"counts={len(class_counts)}, num_classes={num_classes}."
        )

    actual_train_examples = sum(
        class_counts,
    )
    if actual_train_examples != num_train_examples:
        print(
            "Warning: source dataset cardinality differs from metadata: "
            f"actual={actual_train_examples}, "
            f"metadata={num_train_examples}. Using the actual count."
        )

    return count_class_stratified_split_examples(
        class_counts=class_counts,
        validation_split=validation_split,
        keep_validation=False,
    )


def prepare_run_outputs(
    args,
    extra_metric_names,
    wandb_config,
):
    """Build the run name, wandb logger, and CSV/checkpoint paths."""
    run_name = build_run_name(args)
    wandb_run = WandbLogger(
        enabled=args.wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name or run_name,
        mode=args.wandb_mode,
        tags=args.wandb_tags,
        config=wandb_config,
    )

    output_path = None

    if args.save_csv:
        output_path = build_output_path(args)
        write_csv_header(
            output_path,
            extra_metric_names=extra_metric_names,
        )

        print(f"Saving results to: {output_path}")

    checkpoint_path = None

    if args.save_checkpoint:
        checkpoint_path = build_checkpoint_dir(
            checkpoint_dir=args.checkpoint_dir,
            run_name=run_name,
        )

        print(f"Saving checkpoints to: {checkpoint_path}")

    return run_name, wandb_run, output_path, checkpoint_path


@dataclass(frozen=True)
class OfflineManifestPlan:
    """Validated offline-augmentation manifest configuration."""

    example_count: int | None
    original_example_count: int | None
    manifest_path: str
    manifest_mode: str
    manifest_kind: str


def plan_offline_manifests(
    args,
    method_name,
    metadata,
    source_validation_split,
):
    """Validate any offline manifest and count its contribution."""
    offline_example_count = None
    offline_original_example_count = None
    offline_manifest_path = ""
    offline_manifest_mode = "replace"
    offline_manifest_kind = "diffusemix"
    manifest_validation_split = (
        None if args.val_source == "test" else args.validation_split
    )
    if method_name in DIFFUSEMIX_METHOD_NAMES:
        offline_manifest_path = args.diffusemix_manifest
        offline_manifest_mode = args.diffusemix_train_mode
        offline_example_count = validate_diffusemix_manifest(
            manifest_path=args.diffusemix_manifest,
            dataset=args.dataset,
            num_classes=metadata.num_classes,
            validation_split=manifest_validation_split,
            check_images=True,
        )
        print(
            "Using offline DiffuseMix images from: "
            f"{args.diffusemix_manifest} "
            f"({offline_example_count} generated examples, "
            f"mode={args.diffusemix_train_mode})"
        )
    elif method_name in ALIA_METHOD_NAMES:
        offline_manifest_path = args.alia_manifest
        offline_manifest_mode = args.alia_train_mode
        offline_manifest_kind = "alia"
        offline_example_count = validate_alia_manifest(
            manifest_path=args.alia_manifest,
            dataset=args.dataset,
            num_classes=metadata.num_classes,
            validation_split=manifest_validation_split,
            check_images=True,
        )
        print(
            "Using filtered offline ALIA images from: "
            f"{args.alia_manifest} "
            f"({offline_example_count} generated examples, "
            f"mode={args.alia_train_mode})"
        )
    elif method_name in SASPA_METHOD_NAMES:
        offline_manifest_path = args.saspa_manifest
        offline_manifest_mode = "sample"
        offline_manifest_kind = "saspa"
        offline_example_count = validate_saspa_manifest(
            manifest_path=args.saspa_manifest,
            dataset=args.dataset,
            num_classes=metadata.num_classes,
            validation_split=manifest_validation_split,
            check_images=True,
        )
        print(
            "Using filtered offline SaSPA images from: "
            f"{args.saspa_manifest} "
            f"({offline_example_count} accepted images, "
            "mode=source-aligned sample, "
            f"probability={args.saspa_replacement_probability})"
        )

    if offline_example_count is not None and offline_manifest_mode == "append":
        offline_original_example_count = (
            _count_original_diffusemix_examples(
                dataset=args.dataset,
                data_dir=args.data_dir,
                num_classes=metadata.num_classes,
                num_train_examples=metadata.num_train_examples,
                validation_split=source_validation_split,
                known_class_counts=resolve_train_class_counts(
                    dataset_name=args.dataset,
                    data_dir=args.data_dir,
                    metadata=metadata,
                ),
            )
        )

    return OfflineManifestPlan(
        example_count=offline_example_count,
        original_example_count=offline_original_example_count,
        manifest_path=offline_manifest_path,
        manifest_mode=offline_manifest_mode,
        manifest_kind=offline_manifest_kind,
    )


def build_final_test_dataset(
    args,
    method_name: str,
    precomputed_saliency_methods: tuple[str, ...],
):
    """Build only the endpoint dataset after training has closed."""

    if method_name in precomputed_saliency_methods:
        from allthemix.data.salmix_pipeline import build_salmix_test_pipeline

        return build_salmix_test_pipeline(
            name=args.dataset,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            tiny_imagenet_normalization=args.tiny_imagenet_normalization,
            val_source=args.val_source,
            validation_split=args.validation_split,
        )

    return build_test_pipeline(
        name=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        tiny_imagenet_normalization=args.tiny_imagenet_normalization,
        val_source=args.val_source,
        validation_split=args.validation_split,
    )


def build_training_datasets(
    args,
    method_name,
    precomputed_saliency_methods,
    offline,
    include_final_test,
):
    """Build train/eval/final-test/meta-validation input pipelines."""
    meta_validation_ds = None

    if method_name in precomputed_saliency_methods:
        if args.train_subset_fraction < 1.0:
            raise ValueError(
                "train_subset_fraction < 1.0 is not supported for "
                "precomputed-saliency methods."
            )

        from allthemix.data.salmix_pipeline import build_salmix_dataset_pipeline

        train_ds, eval_ds = build_salmix_dataset_pipeline(
            name=args.dataset,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            shuffle_buffer_size=args.shuffle_buffer_size,
            drop_remainder=True,
            use_sal_basic_augmentation=args.sal_basic_aug,
            saliency_dir=args.saliency_dir,
            validation_split=args.validation_split,
            eval_on_test=args.eval_on_test_each_epoch,
            tiny_imagenet_normalization=args.tiny_imagenet_normalization,
            saliency_augmentation_recipe=args.sal_aug_recipe,
            seed=args.data_seed,
            deterministic_data=args.deterministic_data,
            val_source=args.val_source,
        )

        final_test_ds = (
            build_final_test_dataset(
                args=args,
                method_name=method_name,
                precomputed_saliency_methods=precomputed_saliency_methods,
            )
            if include_final_test
            else None
        )

    else:
        train_ds, eval_ds = build_dataset_pipeline(
            name=args.dataset,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            shuffle_buffer_size=args.shuffle_buffer_size,
            drop_remainder=True,
            use_basic_augmentation=args.basic_aug,
            augmentation_recipe=args.aug_recipe,
            validation_split=args.validation_split,
            eval_on_test=args.eval_on_test_each_epoch,
            tiny_imagenet_normalization=args.tiny_imagenet_normalization,
            train_manifest_path=offline.manifest_path,
            train_manifest_kind=offline.manifest_kind,
            train_manifest_mode=offline.manifest_mode,
            train_original_example_count=offline.original_example_count,
            train_manifest_example_count=offline.example_count,
            train_manifest_prevalidated=(
                offline.example_count is not None
            ),
            train_replacement_probability=(
                args.saspa_replacement_probability
            ),
            train_subset_fraction=args.train_subset_fraction,
            seed=args.data_seed,
            deterministic_data=args.deterministic_data,
            debug_train_source=args.debug_train_source,
            val_source=args.val_source,
        )

        final_test_ds = (
            build_final_test_dataset(
                args=args,
                method_name=method_name,
                precomputed_saliency_methods=precomputed_saliency_methods,
            )
            if include_final_test
            else None
        )

        if args.val_select_split_fraction > 0.0:
            # Matched-control checkpoint selection: evaluate and select
            # checkpoints on the same select half (same size, same
            # examples) that val-guided method arms use; the gate half
            # is discarded entirely.
            eval_ds = build_select_half_validation_pipeline(
                eval_ds,
                select_split_fraction=args.val_select_split_fraction,
                data_seed=args.data_seed,
                batch_size=args.batch_size,
            )

        if method_name == "metaaugment":
            meta_validation_ds = build_meta_validation_pipeline(
                name=args.dataset,
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                validation_split=args.validation_split,
                shuffle_buffer_size=args.shuffle_buffer_size,
                seed=args.data_seed,
                tiny_imagenet_normalization=args.tiny_imagenet_normalization,
                deterministic_data=args.deterministic_data,
                train_subset_fraction=args.train_subset_fraction,
                val_source=args.val_source,
            )

    return train_ds, eval_ds, final_test_ds, meta_validation_ds
