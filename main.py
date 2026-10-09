import logging
from typing import Dict

import hydra
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from src.models.model import Model
from src.trainer.trainer import TrainingPipeline
from src.utils.mylogging import init_wandb, pretty_print_config
from src.utils.torch_utils import (
    MiB,
    make_dataloader,
    model_size_b,
    prepare_eval_dataloader,
)

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig) -> Dict[str, float]:

    # Initialize Accelerate and seed the random number generators.
    accelerator: Accelerator = instantiate(cfg.accelerator, log_with="wandb", project_dir=cfg.result_dir)
    set_seed(cfg.seed)
    if accelerator.is_main_process:  # Only the main process prints the cfg
        pretty_print_config(cfg)

    # Build training and validation DataLoaders.
    # Use separate seeds for their sampling and worker initialization.
    train_loader: DataLoader = make_dataloader(cfg.train_dataloader, cfg.seed)
    val_loader: DataLoader = make_dataloader(cfg.val_dataloader, cfg.seed + 1)

    # Instantiate the model with its configured parameters and
    # any additional dataset-dependent params.
    model_kwargs = train_loader.dataset.model_kwargs()
    model: Model = instantiate(cfg.model, **model_kwargs)

    # Initialize the optimizer with the model's trainable parameters.
    optimizer: Optimizer = instantiate(cfg.optimizer, params=model.trainable_parameters())

    # Build the configured learning-rate scheduler.
    lr_scheduler: LRScheduler = get_scheduler(**cfg.lr_scheduler, optimizer=optimizer)

    # Report the training dataset size and model storage footprint.
    logger.info("Dataset size: %s", len(train_loader.dataset))
    logger.info("Model size: %.3f MiB", model_size_b(model) / MiB)

    # Let Accelerate prepare the training objects for the selected device and precision.
    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, lr_scheduler
    )
    # Prepare validation separately to avoid padding it with duplicate samples.
    val_loader = prepare_eval_dataloader(val_loader, accelerator)

    # Initialize W&B tracking with the config and dataset-dependent model params.
    # Saving them in checkpoints lets evaluation rebuild the same model.
    resolved_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    resolved_config["model"].update(model_kwargs)
    init_wandb(accelerator, cfg.wandb, resolved_config, cfg.result_dir)

    try:
        # Instantiate the trainer with the prepared runtime objects.
        trainer: TrainingPipeline = instantiate(
            cfg.trainer,
            model=model,
            dataloaders={"train": train_loader, "val": val_loader},
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            accelerator=accelerator,
            run_config=resolved_config,
        )

        # Restore training state when a checkpoint is provided, then start training.
        if cfg.resume_from is not None:
            trainer.resume(cfg.resume_from)
        return trainer.train()
    finally:
        # Finish W&B tracking even if training raises an exception.
        accelerator.end_training()


if __name__ == "__main__":
    main()
