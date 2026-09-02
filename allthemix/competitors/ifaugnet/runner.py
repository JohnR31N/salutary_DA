from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from allthemix.competitors.ifaugnet.models import (
    AugmentationNetwork,
    FeatureDiscriminator,
    ImageDiscriminator,
    resolve_architecture,
)
from allthemix.competitors.ifaugnet.steps import (
    AugmentTrainState,
    AugNetRetrainStrategy,
    DiscriminatorTrainState,
    augnet_influence_train_step,
    augnet_pretrain_step,
    compute_feature_s_test,
    compute_feature_s_test_residual,
    create_augment_state,
    create_discriminator_state,
    denormalize_images,
    extract_s_test_feature_batch,
    get_classifier_head_params,
    infer_feature_dim,
    inherit_pretrained_decoder,
    normalization_arrays,
    parallel_augnet_influence_train_step,
    parallel_augnet_pretrain_step,
    parallel_extract_s_test_feature_batch,
)
from allthemix.data.pipeline import (
    build_dataset_pipeline,
    build_meta_validation_pipeline,
    build_raw_augmented_train_pipeline,
    build_test_pipeline,
)
from allthemix.data.preprocessors.selector import get_metadata
from allthemix.data.splits import resolve_training_validation_split
from allthemix.data.utils.cardinality import resolve_train_example_count
from allthemix.data.utils.normalization import get_normalization_stats
from allthemix.methods.selector import get_mixer
from allthemix.networks.builder import build_model
from allthemix.training.engine.parallel.parallel_loop import (
    parallel_evaluate,
    parallel_train_one_epoch,
)
from allthemix.training.engine.single.loop import evaluate, train_one_epoch
from allthemix.training.engine.single.train import create_train_state
from allthemix.training.utils.lr_scheduler import build_lr_schedule
from allthemix.utils.checkpoint import (
    build_checkpoint_dir,
    restore_model_state_file,
    save_state_file,
)
from allthemix.utils.experiment_logger import (
    append_epoch_result,
    build_output_path,
    build_run_name,
    format_epoch_message,
    format_final_test_message,
    write_csv_header,
    write_final_test_result,
)
from allthemix.utils.parallel import (
    create_device_rngs,
    replicate_state,
    shard_array,
    unreplicate_state,
)
from allthemix.utils.reproducibility import resolve_data_seed, seed_everything
from allthemix.utils.timer import Timer
from allthemix.utils.wandb_logger import WandbLogger


@struct.dataclass
class PretrainCheckpoint:
    """Checkpoint tree for adversarial G-only pretraining."""

    augment_state: AugmentTrainState
    image_discriminator_state: DiscriminatorTrainState
    feature_discriminator_state: DiscriminatorTrainState


PRETRAIN_METRIC_NAMES = (
    "d_loss",
    "d_image_loss",
    "d_feature_loss",
    "d_real_logit",
    "d_fake_logit",
    "g_loss",
    "g_image_loss",
    "g_feature_loss",
    "pretrain_identity_l2",
    "pretrain_tau_abs_mean",
    "pretrain_tau_saturation_fraction",
    "pretrain_encoder_grad_norm",
    "pretrain_decoder_grad_norm",
    "pretrain_balance_score",
    "pretrain_healthy",
)

INFLUENCE_METRIC_NAMES = (
    "learning_rate",
    "loss",
    "i_aug_loss",
    "raw_i_aug_loss",
    "replacement_influence_std",
    "label_preservation_loss",
    "augmented_influence",
    "original_influence",
    "estimated_val_loss_reduction",
    "identity_l2",
    "accuracy_on_augmented",
    "accuracy_on_original",
    "accuracy_retention",
    "tau_abs_mean",
    "tau_pre_dropout_abs_mean",
    "tau_saturation_fraction",
    "spatial_oob_fraction",
    "appearance_out_of_range_fraction",
    "augmented_out_of_range_fraction",
    "augmented_l1",
    "gradient_global_norm",
    "policy_healthy",
)

RETRAIN_METRIC_NAMES = (
    "ifaugnet_learned_aug_fraction",
    "ifaugnet_learned_aug_probability",
    "ifaugnet_aug_l1",
    "ifaugnet_spatial_oob_fraction",
    "ifaugnet_appearance_out_of_range_fraction",
    "ifaugnet_augmented_out_of_range_fraction",
)


def _build_influence_lr_schedule(args):
    """Build an update-step schedule for IF-AugNet influence learning."""
    schedule_name = (
        "step"
        if args.ifaugnet_lr_schedule == "constant"
        else args.ifaugnet_lr_schedule
    )

    return build_lr_schedule(
        schedule_name=schedule_name,
        base_learning_rate=args.ifaugnet_learning_rate,
        steps_per_epoch=1,
        epochs=args.ifaugnet_influence_steps,
        decay_epochs=(),
        decay_rate=1.0,
        min_learning_rate=args.ifaugnet_min_learning_rate,
        warmup_epochs=args.ifaugnet_warmup_steps,
    )


def _sample_pretrain_tau(
    rng: jax.Array,
    batch_size: int,
    tau_dim: int,
    dropout_rate: float = 0.0,
) -> jnp.ndarray:
    """Sample bounded latent codes matching the encoder's tanh output range."""
    if dropout_rate <= 0.0:
        return jax.random.uniform(
            rng,
            shape=(batch_size, tau_dim),
            minval=-1.0,
            maxval=1.0,
            dtype=jnp.float32,
        )

    if dropout_rate >= 1.0:
        return jnp.zeros(
            (batch_size, tau_dim),
            dtype=jnp.float32,
        )

    # Influence learning and retraining feed G inverted-dropout codes
    # 2 * mask * tanh(E(x)); matched pretraining samples the same support.
    value_rng, mask_rng = jax.random.split(
        rng,
    )
    values = jax.random.uniform(
        value_rng,
        shape=(batch_size, tau_dim),
        minval=-1.0,
        maxval=1.0,
        dtype=jnp.float32,
    )
    keep_probability = 1.0 - dropout_rate
    mask = jax.random.bernoulli(
        mask_rng,
        p=keep_probability,
        shape=values.shape,
    )

    return jnp.where(
        mask,
        values / keep_probability,
        0.0,
    )


def _policy_is_healthy(
    metrics: dict[str, float],
    min_accuracy_retention: float,
    max_tau_saturation_fraction: float,
) -> bool:
    """Reject nonfinite, semantic-destroying, or saturated policies."""
    required = (
        "loss",
        "accuracy_retention",
        "tau_saturation_fraction",
    )

    if not all(
        np.isfinite(
            metrics[name],
        )
        for name in required
    ):
        return False

    return (
        metrics["accuracy_retention"] >= min_accuracy_retention
        and metrics["tau_saturation_fraction"]
        <= max_tau_saturation_fraction
    )


def _pretrain_is_healthy(
    metrics: dict[str, float],
    max_generator_loss: float,
    max_identity_l2: float,
) -> bool:
    """Accept finite G states before adversarial or transform collapse."""
    required = (
        "g_loss",
        "d_loss",
        "pretrain_identity_l2",
        "pretrain_encoder_grad_norm",
        "pretrain_decoder_grad_norm",
    )

    if not all(
        np.isfinite(
            metrics[name],
        )
        for name in required
    ):
        return False

    return (
        metrics["g_loss"] <= max_generator_loss
        and metrics["pretrain_identity_l2"] <= max_identity_l2
        and metrics["pretrain_encoder_grad_norm"] <= 1.0e-12
    )


def _scalar_config(
    args,
) -> dict[str, Any]:
    """Build a W&B-safe config dictionary from parsed arguments."""
    return {
        key: value
        for key, value in vars(
            args,
        ).items()
        if isinstance(
            value,
            (
                str,
                int,
                float,
                bool,
                list,
            ),
        )
    }


def _repeat_dataset(
    dataset,
) -> Iterator[Any]:
    """Yield a finite re-iterable tf.data dataset forever."""
    while True:
        yielded = False

        for batch in dataset:
            yielded = True
            yield batch

        if not yielded:
            raise ValueError(
                "IF-AugNet received an empty training dataset."
            )


def _pad_leading_axis_to_multiple(
    values: jnp.ndarray,
    multiple: int,
) -> jnp.ndarray:
    """Edge-pad one nonempty batch so it can be sharded evenly."""
    values = jnp.asarray(
        values,
    )
    batch_size = int(
        values.shape[0],
    )
    if batch_size < 1:
        raise ValueError(
            "Cannot pad an empty IF-AugNet validation batch."
        )

    remainder = batch_size % multiple
    if remainder == 0:
        return values

    padding = (
        (0, multiple - remainder),
        *((0, 0) for _ in values.shape[1:]),
    )
    return jnp.pad(
        values,
        padding,
        mode="edge",
    )


def _to_float_metrics(
    metrics: dict[str, Any],
    distributed: bool = False,
) -> dict[str, float]:
    """Move scalar metric arrays to host Python floats."""
    return {
        key: float(
            jax.device_get(
                value[0]
                if distributed
                else value
            ),
        )
        for key, value in metrics.items()
    }


def _paired_batch_arrays(
    batch: dict[str, Any],
    mean: jnp.ndarray,
    std: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return normalized and raw aligned views from a paired data batch."""
    raw_normalized = jnp.asarray(
        batch["raw_images"],
    )
    base_normalized = jnp.asarray(
        batch["images"],
    )
    labels = jnp.asarray(
        batch["labels"],
    )
    raw_images = denormalize_images(
        images=raw_normalized,
        mean=mean,
        std=std,
    )
    base_images = denormalize_images(
        images=base_normalized,
        mean=mean,
        std=std,
    )

    return raw_normalized, base_normalized, raw_images, base_images, labels


def _validation_batch_arrays(
    batch,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Convert a standard validation batch to JAX arrays."""
    images, labels = batch[:2]

    return jnp.asarray(
        images,
    ), jnp.asarray(
        labels,
    )


def _prepare_step_csv(
    path: Path | None,
    metric_names: tuple[str, ...],
) -> None:
    """Create a stage-step CSV when local result logging is enabled."""
    if path is None:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
    ) as file:
        csv.writer(
            file,
        ).writerow(
            [
                "step",
                *metric_names,
            ]
        )


def _append_step_csv(
    path: Path | None,
    step: int,
    metrics: dict[str, float],
    metric_names: tuple[str, ...],
) -> None:
    """Append one logged method-stage update to a CSV."""
    if path is None:
        return

    with path.open(
        "a",
        newline="",
    ) as file:
        csv.writer(
            file,
        ).writerow(
            [
                step,
                *(
                    metrics.get(
                        name,
                        "",
                    )
                    for name in metric_names
                ),
            ]
        )


def _named_output(
    output_path: Path | None,
    suffix: str,
) -> Path | None:
    """Build a sibling output path for an internal IF-AugNet stage."""
    if output_path is None:
        return None

    return output_path.with_name(
        f"{output_path.stem}_{suffix}{output_path.suffix}"
    )


def _checkpoint_root(
    args,
    run_name: str,
) -> Path | None:
    """Resolve the stage checkpoint root for new or resumed runs."""
    if args.resume_checkpoint:
        root = Path(
            args.resume_checkpoint,
        ).resolve()

        if not root.exists():
            raise FileNotFoundError(
                f"IF-AugNet checkpoint root does not exist: {root}"
            )

        return root

    if not args.save_checkpoint:
        return None

    return build_checkpoint_dir(
        checkpoint_dir=args.checkpoint_dir,
        run_name=run_name,
    )


def _restore_named_model_state(
    template,
    checkpoint_root: Path | None,
    name: str,
):
    """Restore a stage dependency without importing its optimizer state."""
    if checkpoint_root is None:
        raise ValueError(
            f"IF-AugNet stage requires checkpoint '{name}', but no checkpoint "
            "root was configured."
        )

    path = checkpoint_root / f"{name}.msgpack"
    restored, loaded = restore_model_state_file(
        state=template,
        checkpoint_path=path,
    )
    print(
        f"Restored IF-AugNet model state from {path}; "
        f"loaded fields={','.join(loaded)}; optimizer state reinitialized"
    )
    return restored


def _save_named(
    state,
    checkpoint_root: Path | None,
    name: str,
) -> None:
    """Save one stage artifact when checkpointing is enabled."""
    if checkpoint_root is None:
        return

    save_state_file(
        state=state,
        checkpoint_dir=checkpoint_root,
        name=name,
    )


def _restore_influence_for_retrain(
    *,
    template,
    checkpoint_root: Path | None,
    restore_best_healthy: bool,
):
    """Restore the selected policy, promoting a saved healthy fallback."""
    if checkpoint_root is None:
        return _restore_named_model_state(
            template=template,
            checkpoint_root=checkpoint_root,
            name="ifaugnet_influence_final",
        )

    final_path = checkpoint_root / "ifaugnet_influence_final.msgpack"
    if final_path.is_file() or not restore_best_healthy:
        return _restore_named_model_state(
            template=template,
            checkpoint_root=checkpoint_root,
            name="ifaugnet_influence_final",
        )

    healthy_path = checkpoint_root / "ifaugnet_influence_best_healthy.msgpack"
    if not healthy_path.is_file():
        return _restore_named_model_state(
            template=template,
            checkpoint_root=checkpoint_root,
            name="ifaugnet_influence_final",
        )

    print(
        "IF-AugNet final influence checkpoint is absent; restoring the "
        "saved best healthy policy for retraining"
    )
    restored_state = _restore_named_model_state(
        template=template,
        checkpoint_root=checkpoint_root,
        name="ifaugnet_influence_best_healthy",
    )
    _save_named(
        state=restored_state,
        checkpoint_root=checkpoint_root,
        name="ifaugnet_influence_final",
    )
    return restored_state


def _run_classifier_stage(
    *,
    args,
    stage_name: str,
    state,
    train_dataset,
    validation_dataset,
    mixer_fn,
    num_classes: int,
    epochs: int,
    lr_schedule,
    steps_per_epoch: int,
    rng: jax.Array,
    output_path: Path | None,
    checkpoint_root: Path | None,
    wandb_run: WandbLogger,
    wandb_step_offset: int,
    batch_training_strategy=None,
) -> tuple[Any, Any, jax.Array, float, int]:
    """Train one shared classifier stage and select its best validation state."""
    extra_metric_names = (
        list(
            RETRAIN_METRIC_NAMES,
        )
        if batch_training_strategy is not None
        else []
    )

    if output_path is not None:
        write_csv_header(
            output_path=output_path,
            extra_metric_names=extra_metric_names,
        )

    best_error = float(
        "inf",
    )
    best_epoch = -1
    best_state = None
    logged_steps_per_epoch = steps_per_epoch
    distributed = args.distributed

    if distributed:
        state = replicate_state(
            state,
        )

    if args.max_train_steps > 0:
        logged_steps_per_epoch = min(
            logged_steps_per_epoch,
            args.max_train_steps,
        )

    for epoch in range(
        epochs,
    ):
        timer = Timer()
        epoch_time = None

        if args.log_time:
            timer.start()

        method_name = (
            "ifaugnet"
            if batch_training_strategy is not None
            else "baseline"
        )
        if distributed:
            rng, epoch_rng = jax.random.split(
                rng,
            )
            (
                state,
                _,
                train_loss,
                train_accuracy,
                extra_metrics,
            ) = parallel_train_one_epoch(
                state=state,
                rngs=create_device_rngs(
                    epoch_rng,
                ),
                train_ds=train_dataset,
                mixer_fn=mixer_fn,
                method=method_name,
                num_classes=num_classes,
                max_train_steps=args.max_train_steps,
                sync_batch_stats=args.sync_batch_stats,
                batch_training_strategy=batch_training_strategy,
            )
            (
                validation_loss,
                validation_top1_accuracy,
                validation_top5_accuracy,
                validation_top1_error,
                validation_top5_error,
            ) = parallel_evaluate(
                state=state,
                test_ds=validation_dataset,
                num_classes=num_classes,
                max_eval_steps=args.max_eval_steps,
            )
        else:
            (
                state,
                rng,
                train_loss,
                train_accuracy,
                extra_metrics,
            ) = train_one_epoch(
                state=state,
                rng=rng,
                train_ds=train_dataset,
                mixer_fn=mixer_fn,
                method=method_name,
                num_classes=num_classes,
                max_train_steps=args.max_train_steps,
                batch_training_strategy=batch_training_strategy,
            )
            (
                validation_loss,
                validation_top1_accuracy,
                validation_top5_accuracy,
                validation_top1_error,
                validation_top5_error,
            ) = evaluate(
                state=state,
                test_ds=validation_dataset,
                num_classes=num_classes,
                max_eval_steps=args.max_eval_steps,
            )

        if args.log_time:
            epoch_time = timer.stop()

        if validation_top1_error < best_error:
            best_error = validation_top1_error
            best_epoch = epoch + 1
            best_state = (
                unreplicate_state(
                    state,
                )
                if distributed
                else state
            )
            _save_named(
                state=best_state,
                checkpoint_root=checkpoint_root,
                name=f"{stage_name}_best",
            )

        if (
            stage_name == "ifaugnet_classifier"
            and args.ifaugnet_policy_classifier_save_epoch == epoch + 1
        ):
            early_state = (
                unreplicate_state(
                    state,
                )
                if distributed
                else state
            )
            _save_named(
                state=early_state,
                checkpoint_root=checkpoint_root,
                name="ifaugnet_classifier_early",
            )
            print(
                "[ifaugnet_classifier] Saved early policy-driver "
                f"checkpoint at epoch {epoch + 1}"
            )

        print(
            f"[{stage_name}] "
            + format_epoch_message(
                epoch=epoch + 1,
                total_epochs=epochs,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                eval_loss=validation_loss,
                eval_top1_accuracy=validation_top1_accuracy,
                eval_top5_accuracy=validation_top5_accuracy,
                eval_top1_error=validation_top1_error,
                eval_top5_error=validation_top5_error,
                best_top1_error=best_error,
                epoch_time=epoch_time,
                extra_metrics=extra_metrics,
                eval_name="val",
            )
        )
        current_learning_rate = float(
            lr_schedule(
                max(
                    0,
                    (epoch + 1) * logged_steps_per_epoch - 1,
                )
            )
        )
        stage_metrics = {
            f"{stage_name}/train_loss": train_loss,
            f"{stage_name}/train_accuracy": train_accuracy,
            f"{stage_name}/learning_rate": current_learning_rate,
            f"{stage_name}/val_loss": validation_loss,
            f"{stage_name}/val_top1_error": validation_top1_error,
            f"{stage_name}/best_val_top1_error": best_error,
            f"{stage_name}/best_epoch": float(
                best_epoch,
            ),
            **{
                f"{stage_name}/{key}": value
                for key, value in extra_metrics.items()
            },
        }
        wandb_run.log_metrics(
            step=wandb_step_offset + epoch + 1,
            metrics=stage_metrics,
        )

        if output_path is not None:
            append_epoch_result(
                output_path=output_path,
                epoch=epoch + 1,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                eval_loss=validation_loss,
                eval_top1_accuracy=validation_top1_accuracy,
                eval_top5_accuracy=validation_top5_accuracy,
                eval_top1_error=validation_top1_error,
                eval_top5_error=validation_top5_error,
                best_top1_error=best_error,
                best_epoch=best_epoch,
                epoch_time=epoch_time,
                extra_metrics=extra_metrics,
                extra_metric_names=extra_metric_names,
            )

    if best_state is None:
        raise RuntimeError(
            f"IF-AugNet {stage_name} stage completed no validation epoch."
        )

    final_state = (
        unreplicate_state(
            state,
        )
        if distributed
        else state
    )
    _save_named(
        state=final_state,
        checkpoint_root=checkpoint_root,
        name=f"{stage_name}_final",
    )
    print(
        f"[{stage_name}] Best val top-1 error: {best_error * 100:.2f}% "
        f"at epoch {best_epoch}"
    )

    return final_state, best_state, rng, best_error, best_epoch


def _run_pretrain_stage(
    *,
    args,
    classifier_state,
    augment_state: AugmentTrainState,
    image_discriminator_state: DiscriminatorTrainState,
    feature_discriminator_state: DiscriminatorTrainState,
    paired_dataset,
    mean: jnp.ndarray,
    std: jnp.ndarray,
    rng: jax.Array,
    output_path: Path | None,
    checkpoint_root: Path | None,
    wandb_run: WandbLogger,
    wandb_step_offset: int,
) -> tuple[
    AugmentTrainState,
    DiscriminatorTrainState,
    DiscriminatorTrainState,
    jax.Array,
]:
    """Pretrain G to map random tau codes toward baseline augmentations."""
    _prepare_step_csv(
        path=output_path,
        metric_names=PRETRAIN_METRIC_NAMES,
    )

    if args.ifaugnet_pretrain_steps == 0:
        print(
            "[ifaugnet_pretrain] skipped; influence training will start "
            "from freshly initialized E/G parameters"
        )
        checkpoint = PretrainCheckpoint(
            augment_state=augment_state,
            image_discriminator_state=image_discriminator_state,
            feature_discriminator_state=feature_discriminator_state,
        )
        _save_named(
            state=checkpoint,
            checkpoint_root=checkpoint_root,
            name="ifaugnet_pretrain_last",
        )
        _save_named(
            state=checkpoint,
            checkpoint_root=checkpoint_root,
            name="ifaugnet_pretrain_final",
        )

        return (
            augment_state,
            image_discriminator_state,
            feature_discriminator_state,
            rng,
        )

    iterator = _repeat_dataset(
        paired_dataset,
    )
    last_metrics = None
    distributed = args.distributed
    last_healthy_checkpoint = None
    last_healthy_step = -1
    pretrain_tau_dropout_rate = (
        args.ifaugnet_tau_dropout
        if args.ifaugnet_pretrain_tau_dropout_match
        else 0.0
    )

    if pretrain_tau_dropout_rate > 0.0:
        print(
            "[ifaugnet_pretrain] sampling tau with matched inverted dropout "
            f"(rate {pretrain_tau_dropout_rate:.2f}); G is pretrained on the "
            "same code support it sees during influence learning and retraining"
        )

    equilibrium_loss = (
        2.0
        * np.log(
            2.0,
        )
        * (
            args.ifaugnet_image_loss_weight
            + args.ifaugnet_feature_loss_weight
        )
    )

    if distributed:
        classifier_state = replicate_state(
            classifier_state,
        )
        augment_state = replicate_state(
            augment_state,
        )
        image_discriminator_state = replicate_state(
            image_discriminator_state,
        )
        feature_discriminator_state = replicate_state(
            feature_discriminator_state,
        )

    for step in range(
        args.ifaugnet_pretrain_steps,
    ):
        rng, step_rng = jax.random.split(
            rng,
        )
        discriminator_tau_rng, generator_tau_rng = jax.random.split(
            step_rng,
        )
        (
            _,
            _,
            raw_images,
            base_images,
            _,
        ) = _paired_batch_arrays(
            batch=next(
                iterator,
            ),
            mean=mean,
            std=std,
        )
        discriminator_tau = _sample_pretrain_tau(
            rng=discriminator_tau_rng,
            batch_size=raw_images.shape[0],
            tau_dim=args.ifaugnet_tau_dim,
            dropout_rate=pretrain_tau_dropout_rate,
        )
        generator_tau = _sample_pretrain_tau(
            rng=generator_tau_rng,
            batch_size=raw_images.shape[0],
            tau_dim=args.ifaugnet_tau_dim,
            dropout_rate=pretrain_tau_dropout_rate,
        )
        augment_state_before_step = augment_state
        if distributed:
            (
                augment_state,
                image_discriminator_state,
                feature_discriminator_state,
                metrics,
            ) = parallel_augnet_pretrain_step(
                augment_state,
                image_discriminator_state,
                feature_discriminator_state,
                classifier_state,
                shard_array(
                    raw_images,
                ),
                shard_array(
                    base_images,
                ),
                shard_array(
                    discriminator_tau,
                ),
                shard_array(
                    generator_tau,
                ),
                mean,
                std,
                args.ifaugnet_image_loss_weight,
                args.ifaugnet_feature_loss_weight,
                args.ifaugnet_pretrain_identity_l2_weight,
            )
        else:
            (
                augment_state,
                image_discriminator_state,
                feature_discriminator_state,
                metrics,
            ) = augnet_pretrain_step(
                augment_state=augment_state,
                image_discriminator_state=image_discriminator_state,
                feature_discriminator_state=feature_discriminator_state,
                classifier_state=classifier_state,
                raw_images=raw_images,
                real_images=base_images,
                discriminator_tau=discriminator_tau,
                generator_tau=generator_tau,
                mean=mean,
                std=std,
                image_loss_weight=args.ifaugnet_image_loss_weight,
                feature_loss_weight=args.ifaugnet_feature_loss_weight,
                identity_l2_weight=args.ifaugnet_pretrain_identity_l2_weight,
            )
        should_log = (
            step == 0
            or step + 1 == args.ifaugnet_pretrain_steps
            or (step + 1) % args.ifaugnet_log_every == 0
        )

        if should_log:
            last_metrics = _to_float_metrics(
                metrics,
                distributed=distributed,
            )
            balance_score = (
                abs(
                    last_metrics["g_loss"] - equilibrium_loss,
                )
                + abs(
                    last_metrics["d_loss"] - equilibrium_loss,
                )
            )
            last_metrics["pretrain_balance_score"] = balance_score
            pretrain_healthy = _pretrain_is_healthy(
                metrics=last_metrics,
                max_generator_loss=(
                    args.ifaugnet_max_pretrain_generator_loss
                ),
                max_identity_l2=args.ifaugnet_max_pretrain_identity_l2,
            )
            last_metrics["pretrain_healthy"] = float(
                pretrain_healthy,
            )
            print(
                f"[ifaugnet_pretrain] step {step + 1}/"
                f"{args.ifaugnet_pretrain_steps} | "
                f"g loss: {last_metrics['g_loss']:.4f} | "
                f"d loss: {last_metrics['d_loss']:.4f} | "
                f"identity: {last_metrics['pretrain_identity_l2']:.5f} | "
                f"balance: {balance_score:.4f} | "
                f"healthy: {pretrain_healthy}"
            )
            _append_step_csv(
                path=output_path,
                step=step + 1,
                metrics=last_metrics,
                metric_names=PRETRAIN_METRIC_NAMES,
            )
            wandb_run.log_metrics(
                step=wandb_step_offset + step + 1,
                metrics={
                    f"ifaugnet_pretrain/{key}": value
                    for key, value in last_metrics.items()
                },
            )

            if pretrain_healthy:
                last_healthy_checkpoint = PretrainCheckpoint(
                    augment_state=(
                        unreplicate_state(
                            augment_state_before_step,
                        )
                        if distributed
                        else augment_state_before_step
                    ),
                    image_discriminator_state=(
                        unreplicate_state(
                            image_discriminator_state,
                        )
                        if distributed
                        else image_discriminator_state
                    ),
                    feature_discriminator_state=(
                        unreplicate_state(
                            feature_discriminator_state,
                        )
                        if distributed
                        else feature_discriminator_state
                    ),
                )
                last_healthy_step = step + 1
                _save_named(
                    state=last_healthy_checkpoint,
                    checkpoint_root=checkpoint_root,
                    name="ifaugnet_pretrain_last_healthy",
                )

    if last_metrics is None:
        raise RuntimeError(
            "IF-AugNet pretraining did not execute any update."
        )

    last_checkpoint = PretrainCheckpoint(
        augment_state=(
            unreplicate_state(
                augment_state,
            )
            if distributed
            else augment_state
        ),
        image_discriminator_state=(
            unreplicate_state(
                image_discriminator_state,
            )
            if distributed
            else image_discriminator_state
        ),
        feature_discriminator_state=(
            unreplicate_state(
                feature_discriminator_state,
            )
            if distributed
            else feature_discriminator_state
        ),
    )
    _save_named(
        state=last_checkpoint,
        checkpoint_root=checkpoint_root,
        name="ifaugnet_pretrain_last",
    )
    if (
        args.ifaugnet_restore_last_healthy_pretrain
        and last_healthy_checkpoint is None
    ):
        raise RuntimeError(
            "IF-AugNet G pretraining produced no healthy checkpoint. "
            "Inspect the pretrain CSV or run the paper-supported "
            "skip-pretrain path."
        )

    selected_checkpoint = (
        last_healthy_checkpoint
        if args.ifaugnet_restore_last_healthy_pretrain
        else last_checkpoint
    )

    if selected_checkpoint is last_healthy_checkpoint:
        print(
            "[ifaugnet_pretrain] selected last healthy G checkpoint at "
            f"step {last_healthy_step}"
        )

    _save_named(
        state=selected_checkpoint,
        checkpoint_root=checkpoint_root,
        name="ifaugnet_pretrain_final",
    )

    return (
        selected_checkpoint.augment_state,
        selected_checkpoint.image_discriminator_state,
        selected_checkpoint.feature_discriminator_state,
        rng,
    )


def _precompute_s_test(
    *,
    args,
    classifier_state,
    paired_dataset,
    validation_dataset,
    rng: jax.Array,
) -> tuple[Any, jax.Array]:
    """Build and solve one fixed iHVP system over aggregate feature batches."""
    train_iterator = _repeat_dataset(
        paired_dataset,
    )
    validation_iterator = iter(
        validation_dataset,
    )
    train_feature_batches = []
    train_label_batches = []
    validation_feature_batches = []
    validation_label_batches = []
    distributed = args.distributed

    for _ in range(
        args.ifaugnet_s_test_batches,
    ):
        try:
            validation_batch = next(
                validation_iterator,
            )
        except StopIteration:
            break
        train_batch = next(
            train_iterator,
        )
        # Influence learning is deployed after the dataset's baseline recipe,
        # so the train side of the iHVP must use that same input distribution.
        train_images = jnp.asarray(
            train_batch["images"],
        )
        train_labels = jnp.asarray(
            train_batch["labels"],
        )
        validation_images, validation_labels = _validation_batch_arrays(
            validation_batch,
        )
        validation_example_count = int(
            validation_images.shape[0],
        )
        if distributed:
            num_devices = jax.local_device_count()
            validation_images = _pad_leading_axis_to_multiple(
                validation_images,
                num_devices,
            )
            validation_labels = _pad_leading_axis_to_multiple(
                validation_labels,
                num_devices,
            )
            train_images = shard_array(
                train_images,
            )
            train_labels = shard_array(
                train_labels,
            )
            validation_images = shard_array(
                validation_images,
            )
            validation_labels = shard_array(
                validation_labels,
            )
            feature_batch = parallel_extract_s_test_feature_batch(
                classifier_state,
                train_images,
                train_labels,
                validation_images,
                validation_labels,
            )
            feature_batch = tuple(
                np.asarray(
                    jax.device_get(
                        values[0],
                    )
                )
                for values in feature_batch
            )
        else:
            feature_batch = extract_s_test_feature_batch(
                classifier_state=classifier_state,
                train_images=train_images,
                train_labels=train_labels,
                validation_images=validation_images,
                validation_labels=validation_labels,
            )
            feature_batch = tuple(
                np.asarray(
                    jax.device_get(
                        values,
                    )
                )
                for values in feature_batch
            )
        (
            train_features,
            batch_train_labels,
            validation_features,
            batch_validation_labels,
        ) = feature_batch
        validation_features = validation_features[:validation_example_count]
        batch_validation_labels = batch_validation_labels[
            :validation_example_count
        ]
        train_feature_batches.append(
            train_features,
        )
        train_label_batches.append(
            batch_train_labels,
        )
        validation_feature_batches.append(
            validation_features,
        )
        validation_label_batches.append(
            batch_validation_labels,
        )

    if not train_feature_batches:
        raise ValueError(
            "IF-AugNet fixed s_test requires at least one validation batch."
        )

    train_features = jnp.asarray(
        np.concatenate(
            train_feature_batches,
            axis=0,
        )
    )
    train_labels = jnp.asarray(
        np.concatenate(
            train_label_batches,
            axis=0,
        )
    )
    validation_features = jnp.asarray(
        np.concatenate(
            validation_feature_batches,
            axis=0,
        )
    )
    validation_labels = jnp.asarray(
        np.concatenate(
            validation_label_batches,
            axis=0,
        )
    )
    single_classifier_state = (
        unreplicate_state(
            classifier_state,
        )
        if distributed
        else classifier_state
    )
    classifier_params = get_classifier_head_params(
        single_classifier_state.params,
    )
    s_test = compute_feature_s_test(
        classifier_params=classifier_params,
        train_features=train_features,
        train_labels=train_labels,
        validation_features=validation_features,
        validation_labels=validation_labels,
        damping=args.ifaugnet_damping,
        cg_iters=args.ifaugnet_cg_iters,
    )
    residual = compute_feature_s_test_residual(
        classifier_params=classifier_params,
        train_features=train_features,
        train_labels=train_labels,
        validation_features=validation_features,
        validation_labels=validation_labels,
        s_test=s_test,
        damping=args.ifaugnet_damping,
    )
    s_test_norm = jnp.sqrt(
        sum(
            jnp.vdot(
                leaf,
                leaf,
            )
            for leaf in jax.tree_util.tree_leaves(
                s_test,
            )
        )
    )
    residual_value = float(
        residual,
    )
    s_test_norm_value = float(
        s_test_norm,
    )

    if not np.isfinite(residual_value) or not np.isfinite(s_test_norm_value):
        raise RuntimeError(
            "IF-AugNet produced a non-finite aggregate s_test solve. "
            "Increase ifaugnet_damping or inspect the classifier features."
        )

    print(
        "[ifaugnet_influence] fixed s_test | "
        f"batches: {len(train_feature_batches)} | "
        f"train examples: {train_features.shape[0]} | "
        f"validation examples: {validation_features.shape[0]} | "
        "validation sampling: finite single pass | "
        f"global residual: {residual_value:.6f} | "
        f"norm: {s_test_norm_value:.6f}"
    )

    if distributed:
        s_test = replicate_state(
            s_test,
        )

    return s_test, rng


def _run_influence_stage(
    *,
    args,
    classifier_state,
    augment_state: AugmentTrainState,
    paired_dataset,
    validation_dataset,
    mean: jnp.ndarray,
    std: jnp.ndarray,
    rng: jax.Array,
    output_path: Path | None,
    checkpoint_root: Path | None,
    wandb_run: WandbLogger,
    wandb_step_offset: int,
    learning_rate_schedule,
) -> tuple[AugmentTrainState, jax.Array]:
    """Optimize E/G with the fixed last-layer replacement influence."""
    _prepare_step_csv(
        path=output_path,
        metric_names=INFLUENCE_METRIC_NAMES,
    )
    distributed = args.distributed

    if distributed:
        classifier_state = replicate_state(
            classifier_state,
        )
        augment_state = replicate_state(
            augment_state,
        )

    s_test, rng = _precompute_s_test(
        args=args,
        classifier_state=classifier_state,
        paired_dataset=paired_dataset,
        validation_dataset=validation_dataset,
        rng=rng,
    )
    iterator = _repeat_dataset(
        paired_dataset,
    )
    last_metrics = None
    best_healthy_state = None
    best_healthy_loss = float(
        "inf",
    )
    best_healthy_step = -1
    consecutive_unhealthy = 0
    stopped_early = False

    for step in range(
        args.ifaugnet_influence_steps,
    ):
        rng, step_rng = jax.random.split(
            rng,
        )
        (
            _,
            _,
            _,
            base_images,
            labels,
        ) = _paired_batch_arrays(
            batch=next(
                iterator,
            ),
            mean=mean,
            std=std,
        )
        augment_state_before_step = augment_state
        if distributed:
            augment_state, metrics = parallel_augnet_influence_train_step(
                augment_state,
                classifier_state,
                shard_array(
                    base_images,
                ),
                shard_array(
                    labels,
                ),
                s_test,
                create_device_rngs(
                    step_rng,
                ),
                mean,
                std,
                args.ifaugnet_identity_l2_weight,
                args.ifaugnet_influence_clip_value,
                args.ifaugnet_label_preservation_weight,
            )
        else:
            augment_state, metrics = augnet_influence_train_step(
                augment_state=augment_state,
                classifier_state=classifier_state,
                raw_images=base_images,
                labels=labels,
                s_test=s_test,
                rng=step_rng,
                mean=mean,
                std=std,
                identity_l2_weight=args.ifaugnet_identity_l2_weight,
                influence_clip_value=args.ifaugnet_influence_clip_value,
                label_preservation_weight=(
                    args.ifaugnet_label_preservation_weight
                ),
            )
        should_log = (
            step == 0
            or step + 1 == args.ifaugnet_influence_steps
            or (step + 1) % args.ifaugnet_log_every == 0
        )
        should_health_check = (
            should_log
            or step < 10
            or (step + 1) % args.ifaugnet_health_check_every == 0
        )

        if should_health_check:
            last_metrics = _to_float_metrics(
                metrics,
                distributed=distributed,
            )
            last_metrics["learning_rate"] = float(
                jax.device_get(
                    learning_rate_schedule(step),
                )
            )
            policy_healthy = _policy_is_healthy(
                metrics=last_metrics,
                min_accuracy_retention=(
                    args.ifaugnet_min_accuracy_retention
                ),
                max_tau_saturation_fraction=(
                    args.ifaugnet_max_tau_saturation_fraction
                ),
            )
            last_metrics["policy_healthy"] = float(
                policy_healthy,
            )
            if should_log or not policy_healthy:
                print(
                    f"[ifaugnet_influence] step {step + 1}/"
                    f"{args.ifaugnet_influence_steps} | "
                    f"lr: {last_metrics['learning_rate']:.7f} | "
                    f"loss: {last_metrics['loss']:.5f} | "
                    "estimated val reduction: "
                    f"{last_metrics['estimated_val_loss_reduction']:.5f} | "
                    f"identity: {last_metrics['identity_l2']:.5f} | "
                    f"accuracy retention: "
                    f"{last_metrics['accuracy_retention']:.3f} | "
                    f"tau saturation: "
                    f"{last_metrics['tau_saturation_fraction']:.3f} | "
                    f"healthy: {policy_healthy}"
                )
            _append_step_csv(
                path=output_path,
                step=step + 1,
                metrics=last_metrics,
                metric_names=INFLUENCE_METRIC_NAMES,
            )
            wandb_run.log_metrics(
                step=wandb_step_offset + step + 1,
                metrics={
                    f"ifaugnet_influence/{key}": value
                    for key, value in last_metrics.items()
                },
            )

            if policy_healthy:
                consecutive_unhealthy = 0

                if last_metrics["loss"] < best_healthy_loss:
                    best_healthy_state = (
                        unreplicate_state(
                            augment_state_before_step,
                        )
                        if distributed
                        else augment_state_before_step
                    )
                    best_healthy_loss = last_metrics["loss"]
                    best_healthy_step = step + 1
                    _save_named(
                        state=best_healthy_state,
                        checkpoint_root=checkpoint_root,
                        name="ifaugnet_influence_best_healthy",
                    )
            else:
                consecutive_unhealthy += 1

                if (
                    consecutive_unhealthy
                    >= args.ifaugnet_collapse_patience
                ):
                    print(
                        "[ifaugnet_influence] stopping early after "
                        f"{consecutive_unhealthy} consecutive unhealthy "
                        "policy checks"
                    )
                    stopped_early = True
                    break

    if last_metrics is None:
        raise RuntimeError(
            "IF-AugNet influence training did not execute any update."
        )

    last_state = (
        unreplicate_state(
            augment_state,
        )
        if distributed
        else augment_state
    )
    _save_named(
        state=last_state,
        checkpoint_root=checkpoint_root,
        name="ifaugnet_influence_last",
    )
    selected_state = last_state

    if args.ifaugnet_restore_best_healthy:
        if best_healthy_state is None:
            raise RuntimeError(
                "IF-AugNet influence training produced no healthy policy. "
                "Inspect the influence CSV and use skip-pretrain or explicit "
                "stabilization settings before retraining."
            )

        selected_state = best_healthy_state
        print(
            "[ifaugnet_influence] selected healthy policy at step "
            f"{best_healthy_step} (loss={best_healthy_loss:.6f}, "
            f"early_stop={stopped_early})"
        )
    elif not bool(
        last_metrics["policy_healthy"],
    ):
        raise RuntimeError(
            "Refusing to retrain with an unhealthy final IF-AugNet policy. "
            "Enable ifaugnet_restore_best_healthy or stabilize influence "
            "training before running the retrain stage."
        )

    _save_named(
        state=selected_state,
        checkpoint_root=checkpoint_root,
        name="ifaugnet_influence_final",
    )

    return selected_state, rng


def run_ifaugnet(
    args,
) -> None:
    """Run integrated IF-AugNet through the shared AllTheMix CLI."""
    args.deterministic_data = getattr(
        args,
        "deterministic_data",
        True,
    )
    args.data_seed = resolve_data_seed(
        experiment_seed=args.seed,
        data_seed=getattr(
            args,
            "data_seed",
            -1,
        ),
    )
    seed_everything(
        seed=args.seed,
        strict_determinism=getattr(
            args,
            "strict_determinism",
            False,
        ),
    )
    run_name = build_run_name(
        args,
    )
    output_path = (
        build_output_path(
            args,
        )
        if args.save_csv
        else None
    )
    checkpoint_root = _checkpoint_root(
        args=args,
        run_name=run_name,
    )

    if output_path is not None:
        print(f"Saving IF-AugNet retrain results to: {output_path}")

    if checkpoint_root is not None:
        print(f"Saving IF-AugNet stage checkpoints to: {checkpoint_root}")

    wandb_run = WandbLogger(
        enabled=args.wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name or run_name,
        mode=args.wandb_mode,
        tags=args.wandb_tags,
        config=_scalar_config(
            args,
        ),
    )
    metadata = get_metadata(
        args.dataset,
    )
    model = build_model(
        name=args.model,
        num_classes=metadata.num_classes,
        resnet_stem_type=args.resnet_stem_type,
        preact_stem_bn_relu=args.preact_stem_bn_relu,
        preact_pytorch_default_init=args.preact_pytorch_default_init,
    )
    source_validation_split = resolve_training_validation_split(
        validation_split=args.validation_split,
        val_source=args.val_source,
    )
    train_dataset, validation_dataset = build_dataset_pipeline(
        name=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        shuffle_buffer_size=args.shuffle_buffer_size,
        drop_remainder=True,
        use_basic_augmentation=args.basic_aug,
        augmentation_recipe=args.aug_recipe,
        validation_split=args.validation_split,
        eval_on_test=False,
        tiny_imagenet_normalization=args.tiny_imagenet_normalization,
        train_subset_fraction=args.train_subset_fraction,
        seed=args.data_seed,
        deterministic_data=args.deterministic_data,
        val_source=args.val_source,
    )
    paired_dataset = build_raw_augmented_train_pipeline(
        name=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        shuffle_buffer_size=args.shuffle_buffer_size,
        seed=args.data_seed + 31,
        drop_remainder=True,
        use_basic_augmentation=args.basic_aug,
        augmentation_recipe=args.aug_recipe,
        validation_split=args.validation_split,
        tiny_imagenet_normalization=args.tiny_imagenet_normalization,
        deterministic_data=args.deterministic_data,
        train_subset_fraction=args.train_subset_fraction,
        val_source=args.val_source,
    )
    influence_validation_dataset = build_meta_validation_pipeline(
        name=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        shuffle_buffer_size=args.shuffle_buffer_size,
        seed=args.data_seed + 41,
        tiny_imagenet_normalization=args.tiny_imagenet_normalization,
        deterministic_data=args.deterministic_data,
        repeat=False,
        drop_remainder=False,
        train_subset_fraction=args.train_subset_fraction,
        val_source=args.val_source,
    )
    final_test_dataset = (
        build_test_pipeline(
            name=args.dataset,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            tiny_imagenet_normalization=args.tiny_imagenet_normalization,
            val_source=args.val_source,
            validation_split=args.validation_split,
        )
        if args.final_test
        else None
    )
    train_examples = resolve_train_example_count(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        metadata=metadata,
        validation_split=source_validation_split,
        train_subset_fraction=args.train_subset_fraction,
    )

    if train_examples < args.batch_size:
        raise ValueError(
            "IF-AugNet requires at least one full training batch per epoch "
            "because every stage drops remainder batches: train examples "
            f"{train_examples} < batch size {args.batch_size}. Reduce "
            "batch_size or raise train_subset_fraction."
        )

    steps_per_epoch = train_examples // args.batch_size
    classifier_schedule = build_lr_schedule(
        schedule_name=args.lr_schedule,
        base_learning_rate=args.learning_rate,
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs,
        decay_epochs=args.lr_decay_epochs,
        decay_rate=args.lr_decay_rate,
        min_learning_rate=args.min_learning_rate,
        warmup_epochs=args.warmup_epochs,
    )
    rng = jax.random.PRNGKey(
        args.seed,
    )
    (
        rng,
        classifier_rng,
        augment_rng,
        image_discriminator_rng,
        feature_discriminator_rng,
        retrain_rng,
    ) = jax.random.split(
        rng,
        6,
    )
    input_shape = (
        args.batch_size,
        metadata.image_size,
        metadata.image_size,
        metadata.channels,
    )
    method_input_shape = (
        1,
        metadata.image_size,
        metadata.image_size,
        metadata.channels,
    )
    classifier_state = create_train_state(
        rng=classifier_rng,
        model=model,
        learning_rate=classifier_schedule,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        input_shape=input_shape,
        nesterov=args.nesterov,
    )
    baseline_mixer = get_mixer(
        name="baseline",
        num_classes=metadata.num_classes,
    )
    mean_values, std_values = get_normalization_stats(
        dataset=args.dataset,
        tiny_imagenet_normalization=args.tiny_imagenet_normalization,
    )
    mean, std = normalization_arrays(
        mean=mean_values,
        std=std_values,
    )
    ifaugnet_architecture = resolve_architecture(
        args.ifaugnet_architecture,
        metadata.image_size,
    )
    augment_model = AugmentationNetwork(
        image_size=metadata.image_size,
        channels=metadata.channels,
        tau_dim=args.ifaugnet_tau_dim,
        tau_dropout=args.ifaugnet_tau_dropout,
        spatial_scale=args.ifaugnet_spatial_scale,
        appearance_scale=args.ifaugnet_appearance_scale,
        smoothing_kernel=args.ifaugnet_smoothing_kernel,
        use_appearance=args.ifaugnet_use_appearance,
        encoder_widths=tuple(
            args.ifaugnet_encoder_widths,
        ),
        decoder_widths=tuple(
            args.ifaugnet_decoder_widths,
        ),
        decoder_base_width=args.ifaugnet_decoder_base_width,
        parameterization=args.ifaugnet_transform_parameterization,
        composition=args.ifaugnet_composition,
        architecture=ifaugnet_architecture,
    )
    pretrain_augment_state = create_augment_state(
        rng=augment_rng,
        model=augment_model,
        input_shape=method_input_shape,
        learning_rate=args.ifaugnet_pretrain_learning_rate,
        beta1=args.ifaugnet_pretrain_beta1,
        beta2=args.ifaugnet_pretrain_beta2,
        gradient_clip_norm=args.ifaugnet_gradient_clip_norm,
        zero_nonfinite_grads=args.ifaugnet_zero_nonfinite_grads,
    )
    image_discriminator = ImageDiscriminator(
        architecture=ifaugnet_architecture,
        image_size=metadata.image_size,
    )
    feature_discriminator = FeatureDiscriminator()
    feature_dim = infer_feature_dim(
        classifier_state=classifier_state,
        input_shape=method_input_shape,
    )
    image_discriminator_state = create_discriminator_state(
        rng=image_discriminator_rng,
        model=image_discriminator,
        input_shape=method_input_shape,
        learning_rate=args.ifaugnet_pretrain_learning_rate,
        beta1=args.ifaugnet_pretrain_beta1,
        beta2=args.ifaugnet_pretrain_beta2,
    )
    feature_discriminator_state = create_discriminator_state(
        rng=feature_discriminator_rng,
        model=feature_discriminator,
        input_shape=(1, feature_dim),
        learning_rate=args.ifaugnet_pretrain_learning_rate,
        beta1=args.ifaugnet_pretrain_beta1,
        beta2=args.ifaugnet_pretrain_beta2,
    )

    print(
        "Using integrated IF-AugNet with shared AllTheMix data, model, "
        "classifier epoch loop, evaluation, logging, and checkpoints"
    )
    if args.distributed:
        print(
            "Using distributed IF-AugNet with "
            f"{jax.local_device_count()} devices and synchronized BatchNorm"
        )
    print(
        f"IF-AugNet stage: {args.ifaugnet_stage} | "
        f"pretrain steps: {args.ifaugnet_pretrain_steps} | "
        f"influence steps: {args.ifaugnet_influence_steps}"
    )
    print(
        "IF-AugNet transform: "
        f"parameterization={args.ifaugnet_transform_parameterization} | "
        f"composition={args.ifaugnet_composition} | "
        f"architecture={ifaugnet_architecture} | "
        "policy classifier checkpoint="
        f"{args.ifaugnet_policy_classifier_checkpoint} | "
        f"retrain policy source={args.ifaugnet_retrain_policy_source} | "
        "retrain probability="
        f"{args.ifaugnet_learned_aug_probability:.3f}"
    )

    requested_stage = args.ifaugnet_stage
    run_all = requested_stage == "all"
    use_pretrain_policy = (
        requested_stage == "retrain"
        and args.ifaugnet_retrain_policy_source == "pretrain"
    )
    classifier_best_state = None
    classifier_best_error = float(
        "inf",
    )
    classifier_best_epoch = -1

    if run_all or requested_stage == "classifier":
        (
            classifier_state,
            classifier_best_state,
            rng,
            classifier_best_error,
            classifier_best_epoch,
        ) = _run_classifier_stage(
            args=args,
            stage_name="ifaugnet_classifier",
            state=classifier_state,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            mixer_fn=baseline_mixer,
            num_classes=metadata.num_classes,
            epochs=args.epochs,
            lr_schedule=classifier_schedule,
            steps_per_epoch=steps_per_epoch,
            rng=rng,
            output_path=_named_output(
                output_path=output_path,
                suffix="classifier",
            ),
            checkpoint_root=checkpoint_root,
            wandb_run=wandb_run,
            wandb_step_offset=0,
        )

        if requested_stage == "classifier":
            wandb_run.finish()
            return

        if args.ifaugnet_policy_classifier_checkpoint == "best":
            classifier_state = classifier_best_state
        elif args.ifaugnet_policy_classifier_checkpoint == "early":
            classifier_state = _restore_named_model_state(
                template=classifier_state,
                checkpoint_root=checkpoint_root,
                name="ifaugnet_classifier_early",
            )
    else:
        classifier_state = _restore_named_model_state(
            template=classifier_state,
            checkpoint_root=checkpoint_root,
            name=(
                "ifaugnet_classifier_"
                f"{args.ifaugnet_policy_classifier_checkpoint}"
            ),
        )

    if run_all or requested_stage == "pretrain":
        (
            pretrain_augment_state,
            image_discriminator_state,
            feature_discriminator_state,
            rng,
        ) = _run_pretrain_stage(
            args=args,
            classifier_state=classifier_state,
            augment_state=pretrain_augment_state,
            image_discriminator_state=image_discriminator_state,
            feature_discriminator_state=feature_discriminator_state,
            paired_dataset=paired_dataset,
            mean=mean,
            std=std,
            rng=rng,
            output_path=_named_output(
                output_path=output_path,
                suffix="pretrain",
            ),
            checkpoint_root=checkpoint_root,
            wandb_run=wandb_run,
            wandb_step_offset=args.epochs,
        )

        if requested_stage == "pretrain":
            wandb_run.finish()
            return
    elif (
        requested_stage == "influence" or use_pretrain_policy
    ) and args.ifaugnet_pretrain_steps > 0:
        pretrain_checkpoint = _restore_named_model_state(
            template=PretrainCheckpoint(
                augment_state=pretrain_augment_state,
                image_discriminator_state=image_discriminator_state,
                feature_discriminator_state=feature_discriminator_state,
            ),
            checkpoint_root=checkpoint_root,
            name="ifaugnet_pretrain_final",
        )
        pretrain_augment_state = pretrain_checkpoint.augment_state

    influence_lr_schedule = _build_influence_lr_schedule(
        args,
    )
    influence_optimizer_lr = (
        args.ifaugnet_learning_rate
        if args.ifaugnet_lr_schedule == "constant"
        else influence_lr_schedule
    )
    print(
        "IF-AugNet influence optimizer: "
        f"schedule={args.ifaugnet_lr_schedule} | "
        f"peak_lr={args.ifaugnet_learning_rate:.7f} | "
        f"min_lr={args.ifaugnet_min_learning_rate:.7f} | "
        f"warmup_steps={args.ifaugnet_warmup_steps}"
    )
    influence_augment_state = create_augment_state(
        rng=augment_rng,
        model=augment_model,
        input_shape=method_input_shape,
        # Keep the strict constant-LR optimizer state compatible with older
        # stage checkpoints; scheduled profiles need Optax's step counter.
        learning_rate=influence_optimizer_lr,
        beta1=args.ifaugnet_beta1,
        beta2=args.ifaugnet_beta2,
        gradient_clip_norm=args.ifaugnet_gradient_clip_norm,
        zero_nonfinite_grads=args.ifaugnet_zero_nonfinite_grads,
    )

    if requested_stage != "retrain" or use_pretrain_policy:
        influence_augment_state = inherit_pretrained_decoder(
            fresh_state=influence_augment_state,
            pretrained_state=pretrain_augment_state,
        )

    if run_all or requested_stage == "influence":
        influence_augment_state, rng = _run_influence_stage(
            args=args,
            classifier_state=classifier_state,
            augment_state=influence_augment_state,
            paired_dataset=paired_dataset,
            validation_dataset=influence_validation_dataset,
            mean=mean,
            std=std,
            rng=rng,
            output_path=_named_output(
                output_path=output_path,
                suffix="influence",
            ),
            checkpoint_root=checkpoint_root,
            wandb_run=wandb_run,
            wandb_step_offset=(
                args.epochs
                + args.ifaugnet_pretrain_steps
            ),
            learning_rate_schedule=influence_lr_schedule,
        )

        if requested_stage == "influence":
            wandb_run.finish()
            return
    elif not use_pretrain_policy:
        influence_augment_state = _restore_influence_for_retrain(
            template=influence_augment_state,
            checkpoint_root=checkpoint_root,
            restore_best_healthy=args.ifaugnet_restore_best_healthy,
        )

    retrain_epochs = (
        args.epochs
        if args.ifaugnet_retrain_epochs < 0
        else args.ifaugnet_retrain_epochs
    )
    retrain_learning_rate = (
        args.learning_rate
        if args.ifaugnet_retrain_learning_rate < 0.0
        else args.ifaugnet_retrain_learning_rate
    )
    retrain_schedule = build_lr_schedule(
        schedule_name=args.lr_schedule,
        base_learning_rate=retrain_learning_rate,
        steps_per_epoch=steps_per_epoch,
        epochs=retrain_epochs,
        decay_epochs=args.lr_decay_epochs,
        decay_rate=args.lr_decay_rate,
        min_learning_rate=args.min_learning_rate,
        warmup_epochs=args.warmup_epochs,
    )
    retrained_state = create_train_state(
        rng=retrain_rng,
        model=model,
        learning_rate=retrain_schedule,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        input_shape=input_shape,
        nesterov=args.nesterov,
    )
    retrain_strategy = AugNetRetrainStrategy(
        augment_state=influence_augment_state,
        mean=mean_values,
        std=std_values,
        learned_aug_probability=args.ifaugnet_learned_aug_probability,
        distributed=args.distributed,
        sync_batch_stats=args.sync_batch_stats,
    )
    (
        retrained_state,
        retrained_best_state,
        rng,
        retrained_best_error,
        retrained_best_epoch,
    ) = _run_classifier_stage(
        args=args,
        stage_name="ifaugnet_retrain",
        state=retrained_state,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        mixer_fn=baseline_mixer,
        num_classes=metadata.num_classes,
        epochs=retrain_epochs,
        lr_schedule=retrain_schedule,
        steps_per_epoch=steps_per_epoch,
        rng=rng,
        output_path=output_path,
        checkpoint_root=checkpoint_root,
        wandb_run=wandb_run,
        wandb_step_offset=(
            args.epochs
            + args.ifaugnet_pretrain_steps
            + args.ifaugnet_influence_steps
        ),
        batch_training_strategy=retrain_strategy,
    )

    if args.final_test and final_test_dataset is not None:
        final_state = (
            retrained_best_state
            if args.final_test_checkpoint == "best"
            else retrained_state
        )
        selected_epoch = (
            retrained_best_epoch
            if args.final_test_checkpoint == "best"
            else retrain_epochs
        )
        print(
            "Final test checkpoint: "
            f"{args.final_test_checkpoint} retrain epoch {selected_epoch}"
        )
        if args.distributed:
            (
                test_loss,
                test_top1_accuracy,
                test_top5_accuracy,
                test_top1_error,
                test_top5_error,
            ) = parallel_evaluate(
                state=replicate_state(
                    final_state,
                ),
                test_ds=final_test_dataset,
                num_classes=metadata.num_classes,
                max_eval_steps=args.max_eval_steps,
            )
        else:
            (
                test_loss,
                test_top1_accuracy,
                test_top5_accuracy,
                test_top1_error,
                test_top5_error,
            ) = evaluate(
                state=final_state,
                test_ds=final_test_dataset,
                num_classes=metadata.num_classes,
                max_eval_steps=args.max_eval_steps,
            )
        print(
            format_final_test_message(
                test_loss=test_loss,
                test_top1_accuracy=test_top1_accuracy,
                test_top5_accuracy=test_top5_accuracy,
                test_top1_error=test_top1_error,
                test_top5_error=test_top5_error,
            )
        )
        wandb_run.log_final_test(
            {
                "loss": test_loss,
                "top1_accuracy": test_top1_accuracy,
                "top5_accuracy": test_top5_accuracy,
                "top1_error": test_top1_error,
                "top5_error": test_top5_error,
                "selected_retrain_epoch": float(
                    selected_epoch,
                ),
                "best_validation_top1_error": retrained_best_error,
            }
        )

        if output_path is not None:
            write_final_test_result(
                output_path=output_path,
                test_loss=test_loss,
                test_top1_accuracy=test_top1_accuracy,
                test_top5_accuracy=test_top5_accuracy,
                test_top1_error=test_top1_error,
                test_top5_error=test_top5_error,
            )

    wandb_run.finish()
