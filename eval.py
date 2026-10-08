import json
import logging
from pathlib import Path

import hydra
from accelerate.utils import set_seed
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.trainer.checkpoint_manager import CheckpointManager
from src.utils.mylogging import init_wandb, pretty_print_config
from src.utils.torch_utils import make_dataloader, prepare_eval_dataloader

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="eval")
def main(cfg: DictConfig):

    # Initialize Accelerate and seed the random number generators.
    accelerator = instantiate(cfg.accelerator, log_with="wandb", project_dir=cfg.result_dir)
    set_seed(cfg.seed)
    if accelerator.is_main_process:  # Only the main process prints the cfg
        pretty_print_config(cfg)

    # Instantiate the model from its saved configuration and restore its weights.
    # Apply EMA weights when requested.
    checkpoint = CheckpointManager.read(cfg.checkpoint_path)
    model = instantiate(checkpoint["config"]["model"])
    CheckpointManager.restore_model(checkpoint, model, apply_ema=cfg.apply_ema)

    # Let Accelerate prepare the model for evaluation.
    model = accelerator.prepare_model(model, evaluation_mode=True)

    # Build the validation DataLoader.
    # Prepare it separately to avoid padding it with duplicate samples.
    dataloader = make_dataloader(cfg.val_dataloader, cfg.seed)
    dataloader = prepare_eval_dataloader(dataloader, accelerator)

    # Initialize W&B tracking with the evaluation config and saved model config.
    resolved_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    resolved_config["model"] = checkpoint["config"]["model"]
    init_wandb(accelerator, cfg.wandb, resolved_config, cfg.result_dir)

    try:
        # Instantiate the inference pipeline with the prepared runtime objects.
        pipeline = instantiate(
            cfg.inference,
            model=model,
            dataloader=dataloader,
            accelerator=accelerator,
        )

        # Run evaluation and log the resulting metrics to W&B.
        metrics = pipeline.run()
        accelerator.log(
            {f"eval/{key}": value for key, value in metrics.items()},
            step=0,
        )

        # Only the main process saves and prints the evaluation metrics.
        if accelerator.is_main_process:
            result_dir = Path(cfg.result_dir)
            (result_dir / "metrics.json").write_text(
                json.dumps(metrics, indent=2, allow_nan=False) + "\n"
            )
            logger.info("Evaluation metrics: %s", metrics)

        return metrics
    finally:
        # Finish W&B tracking even if evaluation raises an exception.
        accelerator.end_training()


if __name__ == "__main__":
    main()