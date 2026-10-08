import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from accelerate import Accelerator
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from hydra import compose, initialize_config_dir

from main import main as training_main
from eval import main as evaluation_main
from src.trainer.checkpoint_manager import CheckpointManager
from src.utils.mylogging import init_wandb
from src.utils.torch_utils import cycle, prepare_eval_dataloader
from tests.fixtures import TestDataset, TestModel, TestTrainer, TestInference

ROOT = Path(__file__).resolve().parents[1]


def configuration(tmp_path, eval_config=False):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "config")):
        cfg = compose(config_name="eval" if eval_config else "train")
    cfg.original_work_dir = str(tmp_path)
    cfg.result_dir = str(tmp_path / "run")
    cfg.model._target_ = "tests.fixtures.TestModel"
    cfg.train_dataloader.dataset._target_ = "tests.fixtures.TestDataset"
    cfg.val_dataloader.dataset._target_ = "tests.fixtures.TestDataset"
    cfg.train_dataloader.batch_size = 2
    cfg.val_dataloader.batch_size = 2
    cfg.trainer._target_ = "tests.fixtures.TestTrainer"
    cfg.trainer.iterations = 4
    cfg.trainer.validate_every = 2
    cfg.lr_scheduler.num_warmup_steps = 0
    cfg.accelerator.cpu = True
    cfg.wandb.mode = "offline"
    if eval_config:
        cfg.inference._target_ = "tests.fixtures.TestInference"
    return cfg


@pytest.mark.parametrize("use_ema", [False, True])
def test_train_checkpoint_resume_and_evaluate(tmp_path, use_ema, monkeypatch):
    logged = []
    original_log = Accelerator.log

    def record_log(self, values, step=None, **kwargs):
        logged.append({"step": step, **values})
        return original_log(self, values, step=step, **kwargs)

    monkeypatch.setattr(Accelerator, "log", record_log)
    cfg = configuration(tmp_path)
    cfg.trainer.use_ema = use_ema
    result = training_main.__wrapped__(cfg)
    assert result["loss"] >= 0
    last_path = tmp_path / "run/checkpoints/last.pth"
    best_path = tmp_path / "run/checkpoints/best.pth"
    checkpoint = CheckpointManager.read(last_path)
    assert best_path.is_file()
    assert checkpoint["iteration"] == 4
    assert checkpoint["config"]["model"]["input_dim"] == 1
    assert (checkpoint["ema_state_dict"] is not None) == use_ema
    assert [record["step"] for record in logged] == [1, 2, 3, 4]
    assert all("val/loss" in logged[index] for index in [1, 3])
    assert list((tmp_path / "run/wandb").rglob("*.wandb"))

    cfg.resume_from = str(last_path)
    cfg.trainer.iterations = 6
    cfg.result_dir = str(tmp_path / "resumed")
    training_main.__wrapped__(cfg)
    resumed = CheckpointManager.read(tmp_path / "resumed/checkpoints/last.pth")
    assert resumed["iteration"] == 6
    assert [record["step"] for record in logged[-2:]] == [5, 6]
    assert resumed["optimizer_state_dict"]["state"][0]["step"].item() == 6

    eval_cfg = configuration(tmp_path, eval_config=True)
    eval_cfg.result_dir = str(tmp_path / "evaluation")
    eval_cfg.checkpoint_path = str(best_path)
    eval_cfg.apply_ema = use_ema
    metrics = evaluation_main.__wrapped__(eval_cfg)
    assert metrics["mse"] == pytest.approx(CheckpointManager.read(best_path)["best_val_loss"])
    assert json.loads((tmp_path / "evaluation/metrics.json").read_text()) == metrics
    assert list((tmp_path / "evaluation/wandb").rglob("*.wandb"))
    assert "eval/mse" in logged[-1]


def test_sample_weighted_validation_and_mode_restoration(tmp_path):
    accelerator = Accelerator(cpu=True)
    dataset = TestDataset("", "val")
    model = TestModel(1)
    with torch.no_grad():
        model.linear.weight.zero_()
        model.linear.bias.zero_()
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = get_scheduler("constant", optimizer=optimizer)
    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)
    val_loader = prepare_eval_dataloader(loader, accelerator)
    trainer = TestTrainer(model=model, iterations=1, dataloaders={"train": train_loader, "val": val_loader},
                          optimizer=optimizer, lr_scheduler=scheduler, accelerator=accelerator,
                          result_dir=tmp_path, run_config={})
    expected = dataset.y.square().mean().item()
    assert trainer.validate()["loss"] == pytest.approx(expected)
    assert model.training
    assert all(parameter.grad is None for parameter in model.parameters())
    model.eval()
    trainer.validate()
    assert not model.training
    assert TestInference(model, val_loader, accelerator).run()["mse"] == pytest.approx(expected)


def test_early_stopping(tmp_path):
    cfg = configuration(tmp_path)
    cfg.optimizer.lr = 0.0
    cfg.trainer.iterations = 10
    cfg.trainer.validate_every = 1
    cfg.trainer.patience = 1
    training_main.__wrapped__(cfg)
    checkpoint = CheckpointManager.read(tmp_path / "run/checkpoints/last.pth")
    assert checkpoint["iteration"] == 2
    assert checkpoint["counter"] == 1
    assert CheckpointManager.read(tmp_path / "run/checkpoints/best.pth")["iteration"] == 1


def test_library_ema_update_and_missing_ema_error():
    model = TestModel(1)
    ema = EMAModel(model.parameters(), decay=0.5)
    ema.step(model.parameters())
    old_weight = model.linear.weight.detach().clone()
    with torch.no_grad():
        model.linear.weight.add_(2)
    ema.step(model.parameters())
    assert torch.allclose(ema.shadow_params[0], old_weight + 2 * (1 - ema.cur_decay_value))
    with pytest.raises(ValueError, match="No EMA weights"):
        CheckpointManager.restore_model({"model_state_dict": model.state_dict(), "ema_state_dict": None}, model, apply_ema=True)


def test_disabled_wandb_is_rejected(tmp_path):
    cfg = configuration(tmp_path)
    cfg.wandb.mode = "disabled"
    with pytest.raises(ValueError, match="mandatory"):
        init_wandb(Accelerator(cpu=True, log_with="wandb"), cfg.wandb, {}, str(tmp_path))


@pytest.mark.parametrize("overrides", [[], ["experiment=default"], ["lr_scheduler=constant"]])
def test_config_composition(overrides):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "config")):
        cfg = compose(config_name="train", overrides=overrides)
    assert cfg.train_dataloader.dataset.split == "train"
    assert cfg.val_dataloader.dataset.split == "val"
    assert cfg.wandb.mode == "online"
    assert cfg.optimizer._target_ == "torch.optim.AdamW"
    assert not cfg.trainer.use_ema
    assert cfg.accelerator._target_ == "accelerate.Accelerator"
    assert not (ROOT / "config/hydra").exists()
    assert not (ROOT / "src/run.py").exists()
    assert not (ROOT / "config/optimizer/adam.yaml").exists()


def test_warmup_cosine_and_constant_scheduler():
    model = TestModel(1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    scheduler = get_scheduler("cosine", optimizer, num_warmup_steps=2, num_training_steps=6)
    rates = [scheduler.get_last_lr()[0]]
    for _ in range(6):
        optimizer.step()
        scheduler.step()
        rates.append(scheduler.get_last_lr()[0])
    assert rates[:3] == pytest.approx([0, 0.05, 0.1])
    assert rates[-1] == pytest.approx(0)
    constant_optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    constant = get_scheduler("constant", constant_optimizer)
    for _ in range(3):
        constant_optimizer.step()
        constant.step()
        assert constant.get_last_lr()[0] == pytest.approx(0.1)


def test_project_placeholders_and_empty_cycle(tmp_path):
    from src.dataset.dataset import Dataset
    from src.models.model import Model
    with pytest.raises(NotImplementedError, match="dataset loading"):
        Dataset(str(tmp_path), "train")
    with pytest.raises(NotImplementedError, match="architecture"):
        Model()
    with pytest.raises(ValueError, match="empty dataloader"):
        next(cycle([]))


def test_cli_training_and_evaluation(tmp_path):
    train_dir = tmp_path / "cli_train"
    overrides = [
        "model._target_=tests.fixtures.TestModel",
        "train_dataloader.dataset._target_=tests.fixtures.TestDataset",
        "val_dataloader.dataset._target_=tests.fixtures.TestDataset",
        "trainer._target_=tests.fixtures.TestTrainer",
        "trainer.iterations=2", "trainer.validate_every=1", "accelerator.cpu=true",
        "lr_scheduler.num_warmup_steps=0", "wandb.mode=offline",
        f"hydra.run.dir={train_dir}",
    ]
    training = subprocess.run([sys.executable, "main.py", *overrides], cwd=ROOT, capture_output=True, text=True)
    assert training.returncode == 0, training.stdout + training.stderr
    eval_dir = tmp_path / "cli_eval"
    evaluation = subprocess.run([
        sys.executable, "eval.py", "val_dataloader.dataset._target_=tests.fixtures.TestDataset",
        "inference._target_=tests.fixtures.TestInference", "accelerator.cpu=true", "wandb.mode=offline",
        f"checkpoint_path={train_dir}/checkpoints/best.pth", f"hydra.run.dir={eval_dir}",
    ], cwd=ROOT, capture_output=True, text=True)
    assert evaluation.returncode == 0, evaluation.stdout + evaluation.stderr
    assert json.loads((eval_dir / "metrics.json").read_text())["mse"] >= 0
    assert list((train_dir / "wandb").rglob("*.wandb"))
    assert list((eval_dir / "wandb").rglob("*.wandb"))
