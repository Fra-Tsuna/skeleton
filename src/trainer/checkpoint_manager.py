import logging
import os
import re
import shutil
from pathlib import Path
from uuid import uuid4

import torch
from accelerate.utils import broadcast_object_list, gather_object
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


class CheckpointManager:
    def __init__(self, checkpoints_path, accelerator, patience=None, ema=None, dataloader=None):
        self.checkpoints_path = Path(checkpoints_path).resolve()
        self.accelerator = accelerator
        self.ema = ema
        self.dataloader = dataloader
        if accelerator.project_configuration.automatic_checkpoint_naming:
            raise ValueError("Disable automatic_checkpoint_naming; CheckpointManager manages checkpoint paths.")
        if dataloader is not None and not getattr(dataloader, "use_stateful_dataloader", False):
            raise ValueError("Enable use_stateful_dataloader for mid-epoch checkpointing.")

        def initialize():
            self.checkpoints_path.mkdir(parents=True, exist_ok=True)
            owner = self.checkpoints_path / ".checkpoint-owner"
            if not owner.exists():
                owner.write_text(uuid4().hex)
            return owner.read_text()

        self.owner = self._main_call(initialize)

        self.iteration = 0
        self.best_val_loss = float("inf")
        self.patience = patience
        self.counter = 0
        self.early_stop = False

        # Accelerate saves and restores these objects alongside the training state.
        # Keep their registration order and EMA setting the same when resuming.
        accelerator.register_for_checkpointing(self)
        if ema is not None:
            accelerator.register_for_checkpointing(ema)

    def state_dict(self):
        return {
            "iteration": self.iteration,
            "best_val_loss": self.best_val_loss,
            "counter": self.counter,
            "early_stop": self.early_stop,
            "num_processes": self.accelerator.num_processes,
        }

    def load_state_dict(self, state):
        if state["num_processes"] != self.accelerator.num_processes:
            raise ValueError("Resume training with the same number of processes.")
        self.iteration = state["iteration"]
        self.best_val_loss = state["best_val_loss"]
        self.counter = state["counter"]
        self.early_stop = state["early_stop"]

    def save_training(self, iteration):
        self.iteration = iteration
        staging = self._new_path("state")

        # All processes participate so Accelerate can save their individual RNG states.
        # Publish the new directory only after every process has finished writing.
        error = None
        try:
            # Prevent Accelerate from writing the same DataLoader state file on every rank.
            if self.dataloader is not None:
                self.dataloader.use_stateful_dataloader = False
            try:
                self.accelerator.save_state(str(staging))
            finally:
                if self.dataloader is not None:
                    self.dataloader.use_stateful_dataloader = True
            if self.dataloader is not None:
                # Retain each rank's own DataLoader state instead.
                epoch = self.dataloader.iteration + int(self.dataloader.end_of_dataloader)
                torch.save({"epoch": epoch, "state": self.dataloader.state_dict()},
                           staging / f"dataloader_{self.accelerator.process_index}.pth")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        path = self.checkpoints_path / "last"
        self._finish_write(error, staging, path)
        logger.info("Saved training state: %s", path)
        return path

    def load_training(self, checkpoint_path):
        # Model, optimizer, scheduler, scaler, RNG, EMA, and counters are restored here.
        devices = [parameter.device for parameter in self.ema.shadow_params] if self.ema is not None else []
        error = None
        try:
            if self.dataloader is not None:
                self.dataloader.use_stateful_dataloader = False
            try:
                self.accelerator.load_state(str(checkpoint_path), load_kwargs={"weights_only": False})
            finally:
                if self.dataloader is not None:
                    self.dataloader.use_stateful_dataloader = True
            if self.ema is not None:
                self.ema.shadow_params = [parameter.to(device) for parameter, device in zip(self.ema.shadow_params, devices)]
            if self.dataloader is not None:
                saved = torch.load(Path(checkpoint_path) / f"dataloader_{self.accelerator.process_index}.pth",
                                   map_location="cpu", weights_only=False)
                self.dataloader.set_epoch(saved["epoch"])
                self.dataloader.load_state_dict(saved["state"])
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        errors = [message for message in gather_object([error]) if message is not None]
        if errors:
            raise RuntimeError("Checkpoint load failed: " + "; ".join(errors))

        # The best model may be newer than the last recovery checkpoint.
        # Copy it from the resumed run so this run's best.pth matches best_val_loss.
        source = Path(checkpoint_path).parent / "best.pth"
        best_path = self.checkpoints_path / "best.pth"

        def import_best():
            if not source.is_file():
                return None
            if source.resolve() != best_path.resolve():
                shutil.copyfile(source, best_path)
            return self.read(best_path)["best_val_loss"]

        best_loss = self._main_call(import_best)
        if best_loss is not None and best_loss < self.best_val_loss:
            self.best_val_loss = best_loss
            self.counter = 0
            self.early_stop = False

    def save_model(self, model, config, iteration):
        # Export ordinary and optional EMA weights for standalone evaluation.
        # This file contains no optimizer, scheduler, scaler, or RNG state.
        staging = self._new_path("best").with_suffix(".pth")
        error = None
        try:
            weights = self._model_weights(model)
            ema_weights = None
            if self.ema is not None:
                unwrapped = self.accelerator.unwrap_model(model)
                parameters = list(unwrapped.parameters())
                if len(parameters) != len(self.ema.shadow_params):
                    raise ValueError("EMA parameter count does not match model.")
                shadows = {id(parameter): shadow for parameter, shadow in zip(parameters, self.ema.shadow_params)}
                ema_weights = dict(weights)
                for name, parameter in unwrapped.named_parameters(remove_duplicate=False):
                    ema_weights[name] = shadows[id(parameter)].detach().cpu().clone()

            config = OmegaConf.to_container(OmegaConf.create(config), resolve=True, throw_on_missing=True, enum_to_str=True)
            self.accelerator.save({
                "iteration": iteration,
                "model_state_dict": weights,
                "ema_model_state_dict": ema_weights,
                "config": config,
                "best_val_loss": self.best_val_loss,
            }, staging)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        path = self.checkpoints_path / "best.pth"
        self._finish_write(error, staging, path)
        logger.info("Saved best model: %s", path)
        return path

    def save_if_best(self, val_loss, model, config, iteration):
        # Validation must already be aggregated by sample count, not by rank count.
        losses = gather_object([float(val_loss)])
        if any(loss != losses[0] for loss in losses):
            raise ValueError("Pass the same globally aggregated validation loss on every rank.")
        val_loss = losses[0]

        def decide():
            improved = val_loss < self.best_val_loss
            self.best_val_loss = val_loss if improved else self.best_val_loss
            self.counter = 0 if improved else self.counter + 1
            self.early_stop = not improved and self.patience is not None and self.counter >= self.patience
            return improved, self.best_val_loss, self.counter, self.early_stop

        improved, self.best_val_loss, self.counter, self.early_stop = self._main_call(decide)
        if improved:
            self.save_model(model, config, iteration)
        return improved

    def _model_weights(self, model):
        state = self.accelerator.get_state_dict(model)
        return {key: value.detach().cpu().clone() for key, value in state.items()}

    def _new_path(self, prefix):
        # Every process must use the same checkpoint path.
        def create():
            path = self.checkpoints_path / f".{prefix}-{uuid4().hex}"
            if prefix == "state":
                path.mkdir()
                (path / ".checkpoint-owner").write_text(self.owner)
            return str(path)

        return Path(self._main_call(create))

    def _main_call(self, action):
        result, error = None, None
        if self.accelerator.is_main_process:
            try:
                result = action()
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        result, error = broadcast_object_list([(result, error)])[0]
        if error is not None:
            raise RuntimeError(f"Checkpoint operation failed: {error}")
        return result

    def _finish_write(self, error, staging, destination):
        errors = [message for message in gather_object([error]) if message is not None]
        if errors:
            self._main_call(lambda: self._discard(staging))
            raise RuntimeError("Checkpoint write failed: " + "; ".join(errors))
        self._main_call(lambda: self._publish(staging, destination))

    def _owned_directory(self, path):
        if path.is_symlink() or path.parent != self.checkpoints_path:
            return False
        if re.fullmatch(r"\.?state-[0-9a-f]{32}", path.name) is None:
            return False
        marker = path / ".checkpoint-owner"
        return marker.is_file() and marker.read_text() == self.owner

    def _discard(self, path):
        # Never follow symlinks or remove directories outside this manager's root.
        try:
            if self._owned_directory(path):
                shutil.rmtree(path)
            elif path.parent == self.checkpoints_path and re.fullmatch(r"\.best-[0-9a-f]{32}\.pth", path.name):
                path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove abandoned checkpoint: %s", path)

    def _publish(self, staging, destination):
        previous = None
        link = staging.with_name(f"{staging.name}.link")
        published = staging
        try:
            if staging.is_dir():
                # Flush every file before atomically replacing the directory link.
                # A failed save leaves the previous 'last' checkpoint accessible.
                for root, dirs, files in os.walk(staging, topdown=False):
                    for name in files:
                        self._sync(Path(root) / name)
                    self._sync(Path(root))
                if destination.is_symlink():
                    previous = destination.resolve()
                published = staging.with_name(staging.name.lstrip("."))
                os.replace(staging, published)
                link.symlink_to(published.name, target_is_directory=True)
                os.replace(link, destination)
            else:
                self._sync(staging)
                os.replace(staging, destination)
            self._sync(destination.parent)
        finally:
            try:
                link.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not remove temporary checkpoint link: %s", link)
            if destination.resolve() != published:
                self._discard(published)
                self._discard(staging)

        # Remove the previous generation only after the new one is committed.
        if previous is not None and previous != published:
            self._discard(previous)

    @staticmethod
    def _sync(path):
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def read(checkpoint_path, device="cpu"):
        return torch.load(checkpoint_path, map_location=device, weights_only=True)

    @staticmethod
    def restore_model(checkpoint, model, apply_ema=False):
        weights = checkpoint["ema_model_state_dict"] if apply_ema else checkpoint["model_state_dict"]
        if weights is None:
            raise ValueError("No EMA weights in checkpoint; set apply_ema=false.")
        model.load_state_dict(weights)
        model.eval()