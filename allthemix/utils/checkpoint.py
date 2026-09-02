from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from flax import serialization
from flax.core import freeze, unfreeze
from flax.training.train_state import TrainState


def _get_orbax_checkpoint():
    """Import Orbax checkpointing lazily with a helpful error."""
    try:
        import orbax.checkpoint as ocp
    except Exception as exc:
        raise RuntimeError(
            "Checkpointing requires a working orbax-checkpoint installation. "
            "Install it in the active environment or run with "
            "save_checkpoint: false and without --resume_checkpoint."
        ) from exc

    return ocp


def build_checkpoint_dir(
    checkpoint_dir: str,
    run_name: str,
) -> Path:
    """Build and create the checkpoint directory for a run."""
    path = Path(checkpoint_dir).resolve() / run_name

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def save_checkpoint(
    state: Any,
    checkpoint_dir: Path,
    epoch: int,
) -> None:
    """Save a checkpoint for a numbered epoch."""
    ocp = _get_orbax_checkpoint()
    checkpointer = ocp.PyTreeCheckpointer()

    save_path = (checkpoint_dir / f"epoch_{epoch}").resolve()

    checkpointer.save(
        save_path,
        state,
        force=True,
    )


def save_best_checkpoint(
    state: Any,
    checkpoint_dir: Path,
) -> None:
    """Save the best-so-far checkpoint."""
    ocp = _get_orbax_checkpoint()
    checkpointer = ocp.PyTreeCheckpointer()

    save_path = (checkpoint_dir / "best").resolve()

    checkpointer.save(
        save_path,
        state,
        force=True,
    )


def save_state_file(
    state: Any,
    checkpoint_dir: Path,
    name: str,
) -> Path:
    """Atomically save a Flax state as a stage-oriented msgpack file."""
    if not name or Path(name).name != name:
        raise ValueError(
            f"Checkpoint name must be one path component, got: {name!r}"
        )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    save_path = (checkpoint_dir / f"{name}.msgpack").resolve()
    temporary_path = save_path.with_name(
        f".{save_path.name}.tmp"
    )
    temporary_path.write_bytes(
        serialization.to_bytes(
            state,
        )
    )
    temporary_path.replace(
        save_path,
    )

    return save_path


def restore_state_file(
    state: Any,
    checkpoint_path: str | Path,
) -> Any:
    """Restore a Flax state from a msgpack file using a state template."""
    path = Path(
        checkpoint_path,
    ).resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"State checkpoint file does not exist: {path}"
        )

    return serialization.from_bytes(
        state,
        path.read_bytes(),
    )


_MODEL_STATE_FIELDS = frozenset(
    {
        "params",
        "batch_stats",
    }
)


def _validate_serialized_tree(
    target: Any,
    source: Any,
    path: str,
) -> None:
    """Require an exact model-tree structure and matching array shapes."""
    if isinstance(target, Mapping):
        if not isinstance(source, Mapping):
            raise ValueError(
                f"Checkpoint model field {path} is not a mapping."
            )
        target_keys = set(target)
        source_keys = set(source)
        if target_keys != source_keys:
            missing = sorted(target_keys - source_keys)
            extra = sorted(source_keys - target_keys)
            raise ValueError(
                f"Checkpoint model field {path} has incompatible keys; "
                f"missing={missing}, extra={extra}."
            )
        for key in target:
            _validate_serialized_tree(
                target=target[key],
                source=source[key],
                path=f"{path}/{key}",
            )
        return

    if isinstance(target, (list, tuple)):
        if not isinstance(source, (list, tuple)) or len(target) != len(source):
            source_size = len(source) if isinstance(source, (list, tuple)) else -1
            raise ValueError(
                f"Checkpoint model field {path} has incompatible sequence "
                f"length: expected {len(target)}, got {source_size}."
            )
        for index, (target_value, source_value) in enumerate(
            zip(target, source)
        ):
            _validate_serialized_tree(
                target=target_value,
                source=source_value,
                path=f"{path}/{index}",
            )
        return

    target_shape = getattr(target, "shape", None)
    source_shape = getattr(source, "shape", None)
    if target_shape is not None or source_shape is not None:
        if target_shape != source_shape:
            raise ValueError(
                f"Checkpoint model field {path} has incompatible shape: "
                f"expected {target_shape}, got {source_shape}."
            )


def _merge_serialized_model_fields(
    target: Any,
    source: Any,
    prefix: str = "",
) -> tuple[Any, list[str]]:
    """Replace only params and batch statistics in a serialized state tree."""
    if not isinstance(target, Mapping) or not isinstance(source, Mapping):
        return target, []

    merged = dict(target)
    loaded: list[str] = []
    for key, target_value in target.items():
        path = f"{prefix}/{key}" if prefix else str(key)
        if key in _MODEL_STATE_FIELDS:
            if key not in source:
                raise ValueError(
                    f"Checkpoint is missing required model field: {path}."
                )
            _validate_serialized_tree(
                target=target_value,
                source=source[key],
                path=path,
            )
            merged[key] = source[key]
            loaded.append(path)
            continue

        if key in source:
            merged_value, child_loaded = _merge_serialized_model_fields(
                target=target_value,
                source=source[key],
                prefix=path,
            )
            merged[key] = merged_value
            loaded.extend(child_loaded)

    return merged, loaded


def restore_model_state_file(
    state: Any,
    checkpoint_path: str | Path,
) -> tuple[Any, list[str]]:
    """Restore model fields while retaining the current optimizer structure."""
    path = Path(checkpoint_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"State checkpoint file does not exist: {path}"
        )

    target_state = serialization.to_state_dict(state)
    source_state = serialization.msgpack_restore(path.read_bytes())
    merged_state, loaded = _merge_serialized_model_fields(
        target=target_state,
        source=source_state,
    )
    if not loaded:
        raise ValueError(
            f"Checkpoint contains no compatible model fields: {path}"
        )

    return serialization.from_state_dict(state, merged_state), loaded


def restore_checkpoint(
    state: Any,
    checkpoint_path: str,
) -> Any:
    """Restore train state from a checkpoint path."""
    ocp = _get_orbax_checkpoint()
    checkpointer = ocp.PyTreeCheckpointer()
    restore_args = ocp.checkpoint_utils.construct_restore_args(
        state,
    )

    restored_state = checkpointer.restore(
        Path(checkpoint_path).resolve(),
        args=ocp.args.PyTreeRestore(
            item=state,
            restore_args=restore_args,
        ),
    )

    return restored_state


def _as_mutable_mapping(
    tree: Any,
) -> Any:
    """Convert a Flax tree into a mutable mapping when possible."""
    try:
        return unfreeze(
            tree,
        )
    except TypeError:
        return tree


def _merge_matching_leaves(
    target,
    source,
    prefix: str = "",
) -> tuple[Any, list[str], list[str]]:
    """Merge leaves with matching paths and shapes from source into target."""
    target = _as_mutable_mapping(
        target,
    )
    source = _as_mutable_mapping(
        source,
    )

    if isinstance(target, dict) and isinstance(source, dict):
        loaded: list[str] = []
        skipped: list[str] = []
        merged = {}

        for key, target_value in target.items():
            path = f"{prefix}/{key}" if prefix else str(key)

            if key not in source:
                skipped.append(
                    path,
                )
                merged[key] = target_value
                continue

            merged_value, child_loaded, child_skipped = _merge_matching_leaves(
                target=target_value,
                source=source[key],
                prefix=path,
            )
            merged[key] = merged_value
            loaded.extend(
                child_loaded,
            )
            skipped.extend(
                child_skipped,
            )

        return merged, loaded, skipped

    if hasattr(
        target,
        "shape",
    ) and hasattr(
        source,
        "shape",
    ) and target.shape == source.shape:
        return source, [prefix], []

    return target, [], [prefix]


def _extract_tree_field(
    restored,
    field_name: str,
):
    """Extract a field from restored TrainState-like or dict-like checkpoints."""
    if hasattr(
        restored,
        field_name,
    ):
        return getattr(
            restored,
            field_name,
        )

    if isinstance(
        restored,
        dict,
    ):
        return restored.get(
            field_name,
        )

    return None


def restore_matching_pretrained_checkpoint(
    state: TrainState,
    checkpoint_path: str,
) -> tuple[TrainState, list[str], list[str]]:
    """Load matching params and batch stats from a checkpoint into a state."""
    ocp = _get_orbax_checkpoint()
    checkpointer = ocp.PyTreeCheckpointer()

    restored = checkpointer.restore(
        Path(checkpoint_path).resolve(),
    )

    source_params = _extract_tree_field(
        restored,
        "params",
    )
    source_batch_stats = _extract_tree_field(
        restored,
        "batch_stats",
    )

    loaded: list[str] = []
    skipped: list[str] = []

    new_params = state.params
    if source_params is not None:
        merged_params, param_loaded, param_skipped = _merge_matching_leaves(
            target=state.params,
            source=source_params,
            prefix="params",
        )
        new_params = freeze(
            merged_params,
        )
        loaded.extend(
            param_loaded,
        )
        skipped.extend(
            param_skipped,
        )

    new_batch_stats = getattr(
        state,
        "batch_stats",
        None,
    )
    if source_batch_stats is not None and new_batch_stats is not None:
        merged_batch_stats, stats_loaded, stats_skipped = _merge_matching_leaves(
            target=new_batch_stats,
            source=source_batch_stats,
            prefix="batch_stats",
        )
        new_batch_stats = freeze(
            merged_batch_stats,
        )
        loaded.extend(
            stats_loaded,
        )
        skipped.extend(
            stats_skipped,
        )

    if new_batch_stats is None:
        new_state = state.replace(
            params=new_params,
        )
    else:
        new_state = state.replace(
            params=new_params,
            batch_stats=new_batch_stats,
        )

    return new_state, loaded, skipped


def checkpoint_exists(
    checkpoint_path: str,
) -> bool:
    """Return whether a checkpoint path exists."""
    path = Path(checkpoint_path).resolve()

    return path.exists()
