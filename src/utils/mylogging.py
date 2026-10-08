from pathlib import Path

import wandb  # Required dependency; no optional logging fallback.
from omegaconf import OmegaConf
from rich.console import Console
from rich.syntax import Syntax

console = Console()


def pretty_print_config(cfg):
    console.print(Syntax(OmegaConf.to_yaml(cfg, resolve=True), "yaml", theme="monokai"))


def init_wandb(accelerator, wandb_config, config, result_dir):
    """Always initialize W&B through Accelerate; disabled logging is unsupported."""
    args = OmegaConf.to_container(wandb_config, resolve=True)
    project = args.pop("project")
    if args["mode"] not in ("online", "offline"):
        raise ValueError("W&B logging is mandatory. wandb.mode must be online or offline.")
    if accelerator.is_main_process:
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        (Path(result_dir) / "config.yaml").write_text(OmegaConf.to_yaml(OmegaConf.create(config)))
    accelerator.init_trackers(project, config=config, init_kwargs={"wandb": {**args, "dir": str(result_dir)}})
    if accelerator.is_main_process and accelerator.get_tracker("wandb", unwrap=True).disabled:
        raise RuntimeError("W&B initialized in disabled mode; enable online or offline logging.")
