import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from accelerate import Accelerator
from accelerate.scheduler import AcceleratedScheduler
from diffusers.training_utils import EMAModel
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.models.model import Model
from src.trainer.checkpoint_manager import CheckpointManager
from src.utils.torch_utils import aggregate_metrics, batch_size, cycle

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Run task-specific training with shared validation and checkpoint logic."""

    def __init__(
        self,
        model: Model,
        iterations: int,
        dataloaders: Dict[str, DataLoader],
        optimizer: Optimizer,
        lr_scheduler: AcceleratedScheduler,
        accelerator: Accelerator,
        result_dir: Union[str, Path],
        run_config: Dict,
        validate_every: int = 100,
        patience: Optional[int] = None,
        grad_clip_norm: Optional[float] = None,
        use_ema: bool = False,
        ema: Optional[Dict] = None,
        checkpoint_every: int = 500,
    ):

        if iterations < 1 or validate_every < 1 or checkpoint_every < 1:
            raise ValueError(
                "iterations, validate_every, and checkpoint_every must be positive."
            )
        if len(dataloaders["train"]) == 0:
            raise ValueError("Training dataloader must contain batches.")
        if accelerator.gradient_accumulation_steps != 1:
            raise ValueError(
                "This training loop requires gradient_accumulation_steps=1."
            )

        # The model, optimizer, scheduler, and DataLoaders are prepared in main.py.
        self.model = model
        self.iterations = iterations
        self.train_dataloader = dataloaders["train"]
        self.val_dataloader = dataloaders["val"]
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        self.config = run_config
        self.validate_every = validate_every
        self.checkpoint_every = checkpoint_every
        self.grad_clip_norm = grad_clip_norm

        # EMA tracks the underlying model parameters, independently of the wrapper.
        parameters = accelerator.unwrap_model(model).parameters()
        self.ema: Optional[EMAModel] = (
            EMAModel(parameters, **(ema or {})) if use_ema else None
        )
        if self.ema is not None:
            self.ema.to(accelerator.device)

        # Pass the stateful training loader so the manager can restore each rank's position.
        checkpoint_loader = (
            self.train_dataloader
            if getattr(self.train_dataloader, "use_stateful_dataloader", False)
            else None
        )
        self.manager = CheckpointManager(
            Path(result_dir) / "checkpoints",
            accelerator,
            patience=patience,
            ema=self.ema,
            dataloader=checkpoint_loader,
        )
        self.start_iteration = 0

    def compute_loss(self, batch: Any) -> Dict[str, torch.Tensor]:
        """Return {'loss': scalar differentiable tensor, ...scalar batch-mean metrics}."""
        raise NotImplementedError(
            "Implement TrainingPipeline.compute_loss for your project."
        )

    def metric_weight(self, batch: Any) -> int:
        # Weight batch-mean validation metrics by the number of samples.
        return batch_size(batch)

    def train(self) -> Dict[str, float]:
        self.model.train()

        # Continue across epochs until the target number of optimizer updates is reached.
        # A restored stateful loader starts from its saved position.
        iterator = cycle(self.train_dataloader)
        last_metrics: Dict[str, float] = {}
        step = self.start_iteration

        # Every process trains; only the main process displays progress.
        progress = tqdm(
            total=self.iterations,
            initial=step,
            desc="Training",
            disable=not self.accelerator.is_main_process,
        )

        while step < self.iterations and not self.manager.early_stop:
            # Accelerate places each process's batch on the selected device.
            batch = next(iterator)
            self.optimizer.zero_grad(set_to_none=True)

            # Run the forward pass with autocasting and let Accelerate handle backward.
            with self.accelerator.autocast():
                losses = self.compute_loss(batch)
            self.accelerator.backward(losses["loss"])

            # Optionally clip gradients
            if self.grad_clip_norm is not None:
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_norm
                )

            # Record the learning rate used for this optimizer update.
            lr = self.optimizer.param_groups[0]["lr"]
            self.optimizer.step()

            # An overflow consumes the batch but does not complete an optimizer update.
            if self.accelerator.optimizer_step_was_skipped:
                continue

            # Advance the scheduler, EMA, and update count only after a successful step.
            self.lr_scheduler.step()
            if self.ema is not None:
                self.ema.step(self.accelerator.unwrap_model(self.model).parameters())
            step += 1

            # Reduce detached training metrics across processes for logging.
            metrics = {
                f"train/{key}": self.accelerator.reduce(
                    value.detach(), reduction="mean"
                ).item()
                for key, value in losses.items()
            }
            metrics["lr"] = lr

            # Validation returns global sample-weighted means on every rank.
            # The manager synchronizes best-model selection and early stopping state.
            if step % self.validate_every == 0 or step == self.iterations:
                last_metrics = self.validate()
                metrics.update(
                    {f"val/{key}": value for key, value in last_metrics.items()}
                )
                self.manager.save_if_best(
                    last_metrics["loss"], self.model, self.config, step
                )

            # Save full training state independently of validation.
            # Include the final update and any update that triggers early stopping.
            if (
                step % self.checkpoint_every == 0
                or step == self.iterations
                or self.manager.early_stop
            ):
                self.manager.save_training(step)

            # Log and display the completed update.
            self.accelerator.log(metrics, step=step)
            progress.update(1)
            progress.set_postfix(loss=f"{metrics['train/loss']:.4f}")

            if self.manager.early_stop:
                logger.info("Early stopping at iteration %s", step)
                break

        progress.close()
        return last_metrics

    @torch.inference_mode()
    def validate(self) -> Dict[str, float]:
        # Preserve the prepared training model and its current mode.
        training_model = self.model
        was_training = training_model.training

        # Unwrap the model so uneven validation shards can finish independently.
        self.model = self.accelerator.unwrap_model(training_model)
        parameters = list(self.model.parameters())
        ema_stored = False

        totals: Dict[str, float] = {}
        total_weight = 0
        try:
            # Temporarily use EMA weights, keeping the live training weights for restoration.
            if self.ema is not None:
                self.ema.store(parameters)
                ema_stored = True
                self.ema.copy_to(parameters)
            self.model.eval()

            # Accumulate weighted metric sums and sample counts within each process.
            for batch in self.val_dataloader:
                with self.accelerator.autocast():
                    losses = self.compute_loss(batch)

                weight = self.metric_weight(batch)
                total_weight += weight
                for key, value in losses.items():
                    totals[key] = totals.get(key, 0.0) + value.item() * weight

            # Combine process totals into global means, accounting for unequal shard sizes.
            return aggregate_metrics(totals, total_weight)
        finally:
            # Restore the live weights, prepared model, and original mode after validation.
            try:
                if ema_stored:
                    self.ema.restore(parameters)
            finally:
                self.model = training_model
                self.model.train(was_training)

    def resume(self, checkpoint_path: Union[str, Path]) -> None:
        if self.manager.dataloader is None:
            raise ValueError(
                "Enable use_stateful_dataloader to resume the training DataLoader."
            )

        # Restore Accelerate state, EMA, counters, and each rank's DataLoader position.
        self.manager.load_training(checkpoint_path)

        # iterations is the total target, including updates completed before the checkpoint.
        self.start_iteration = self.manager.iteration
        if self.start_iteration >= self.iterations:
            raise ValueError(
                "trainer.iterations must exceed the checkpoint's completed iterations."
            )
