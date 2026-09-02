from __future__ import annotations

from typing import Any


class WandbLogger:
    """Small optional wrapper around Weights & Biases logging."""

    def __init__(
        self,
        enabled: bool,
        project: str,
        entity: str,
        run_name: str,
        mode: str,
        tags: str,
        config: dict[str, Any],
        group: str = "",
        job_type: str = "",
    ) -> None:
        """Initialize a W&B run only when explicitly enabled."""
        self.enabled = enabled
        self._wandb = None
        self._run = None
        self._mode = mode
        self._finished = False

        if not enabled:
            return

        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "wandb logging was requested, but wandb is not installed. "
                "Install it with: pip install wandb"
            ) from exc

        tag_list = [
            tag.strip()
            for tag in tags.split(",")
            if tag.strip()
        ]
        init_kwargs: dict[str, Any] = {
            "project": project,
            "name": run_name or None,
            "tags": tag_list or None,
            "config": config,
        }

        if entity:
            init_kwargs["entity"] = entity

        if mode:
            init_kwargs["mode"] = mode

        if group:
            init_kwargs["group"] = group

        if job_type:
            init_kwargs["job_type"] = job_type

        self._wandb = wandb
        self._run = wandb.init(
            **init_kwargs,
        )

    def log_epoch(
        self,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        """Log one epoch worth of scalar metrics."""
        if not self.enabled or self._wandb is None:
            return

        self._wandb.log(
            {
                "epoch": epoch,
                **metrics,
            },
            step=epoch,
        )

    def log_final_test(
        self,
        metrics: dict[str, float],
    ) -> None:
        """Log final test scalars."""
        if not self.enabled or self._wandb is None:
            return

        self._wandb.log(
            {
                f"final_test/{key}": value
                for key, value in metrics.items()
            }
        )

    def log_metrics(
        self,
        step: int,
        metrics: dict[str, float],
    ) -> None:
        """Log arbitrary staged-training metrics at one monotonic step."""
        if not self.enabled or self._wandb is None:
            return

        self._wandb.log(
            metrics,
            step=step,
        )

    def finish(
        self,
    ) -> None:
        """Close the active W&B run."""
        if not self.enabled or self._wandb is None:
            self._finished = True
            return

        self._wandb.finish()
        self._finished = True

    def closure_metadata(self) -> dict[str, Any]:
        """Return stable run identity and whether ``finish`` completed."""

        return {
            "enabled": self.enabled,
            "mode": self._mode,
            "run_id": (
                str(self._run.id)
                if self._run is not None and getattr(self._run, "id", None)
                else None
            ),
            "url": (
                str(self._run.url)
                if self._run is not None and getattr(self._run, "url", None)
                else None
            ),
            "finish_completed": self._finished,
        }
