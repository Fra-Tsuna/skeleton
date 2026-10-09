# Project template

A project-agnostic PyTorch skeleton with Hydra configuration, mandatory W&B
logging, and Hugging Face Accelerate integrated into training and evaluation.
Implement the dataset, model, training loss, and evaluation metrics for each new
project. No cluster dispatch configuration is included.

```text
main.py                       Training orchestration and entrypoint
eval.py                       Evaluation orchestration and entrypoint
config/
  train.yaml / eval.yaml      Top-level configurations
  accelerator/                Device and precision settings
  dataloader/dataset/          Training and validation datasets
  model/                      Model constructor configuration
  optimizer/adamw.yaml         AdamW only
  lr_scheduler/               Warmup plus cosine; constant LR
  trainer/                    Training and optional EMA settings
  inference/                  Evaluation pipeline configuration
  experiment/                 Named experiment overrides
  wandb/                      Mandatory W&B configuration
src/
  dataset/dataset.py          Dataset extension point
  models/model.py             Model extension point
  trainer/trainer.py          TrainingPipeline and validation
  trainer/checkpoint_manager.py
  inference/inference.py      InferencePipeline
  utils/                      Logging, DataLoader helpers, figure saving
scripts/                      Project preprocessing scripts
assets/                       Project documentation media
data/                         Local data, ignored by Git
ckpt/                         Selected checkpoints, ignored by Git
tests/                        Infrastructure checks
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
wandb login
```

W&B, Accelerate, Diffusers, and torchdata are required dependencies. Diffusers
supplies the LR schedulers and EMA implementation. There is no custom EMA or LR
scheduler. torchdata supplies the stateful DataLoader used for exact resume.

## Add project code

Implement these extension points:

1. `Dataset.__init__`, `__len__`, and `__getitem__` in `src/dataset/dataset.py`.
2. `Model.__init__` and `forward` in `src/models/model.py`.
3. `TrainingPipeline.compute_loss(batch)`: return scalar tensor batch means,
   including a differentiable `loss`.
4. `InferencePipeline.compute_metrics(predictions, batch)`: return scalar batch
   means. Override `predict` if your model needs custom inference or sampling.

Constructor arguments belong in the corresponding Hydra configuration.
`Dataset.model_kwargs()` can supply inferred model dimensions. Metrics are
weighted by sample count; override `metric_weight` for another denominator.
The default placeholders deliberately raise `NotImplementedError`.

Training orchestration lives directly in `main.py`; evaluation orchestration
lives directly in `eval.py`. There is no `src/run.py` or forwarded `main` call.

## Train and evaluate

Inspect configurations without running the project placeholders:

```bash
python main.py --cfg job
python eval.py --cfg job
```

After implementing the project code:

```bash
python main.py
python main.py experiment=default optimizer.lr=3e-4
python main.py accelerator.cpu=true
python main.py accelerator.mixed_precision=bf16
python main.py trainer.use_ema=true
python main.py lr_scheduler=constant
python eval.py checkpoint_path=/absolute/path/to/checkpoints/best.pth
python eval.py checkpoint_path=/absolute/path/to/checkpoints/best.pth apply_ema=true
```

Accelerate prepares the model, optimizer, training DataLoader, and scheduler.
It handles device placement, autocasting, backward passes, gradient scaling,
gradient clipping, tracking, and checkpoint writes. Evaluation DataLoaders use
Accelerate's preparation helper without padded duplicate samples. Validation
uses the unwrapped model so uneven evaluation shards do not invoke DDP
collectives. Scalar metrics are combined across processes with Accelerate.
No job submission or cluster launcher is configured. Accelerate keeps its
device and precision state for the whole process, so a Hydra multirun can sweep
any setting except those under `accelerator`.

The default scheduler uses linear warmup for `lr_scheduler.num_warmup_steps`
updates followed by cosine decay over `trainer.iterations` total updates.
`lr_scheduler=constant` keeps the LR fixed without warmup. Schedulers and EMA
advance only after successful optimizer updates, including under mixed precision.
Mixed precision support depends on your hardware.

EMA is disabled by default. Enable it with `trainer.use_ema=true`; its settings
are under `trainer.ema`. Validation uses EMA weights when enabled. Standalone
evaluation selects them with `apply_ema=true`.

## Logging and checkpoints

W&B always initializes for training and evaluation. Its default mode is online.
Offline mode still records a real W&B run and can be selected for local testing:

```bash
python main.py wandb.mode=offline
```

`wandb.mode=disabled` is rejected. There is no optional logger or JSONL fallback.
Run directories contain the resolved `config.yaml`, W&B records, and training
checkpoints. Evaluation also saves `metrics.json`.

`best.pth` stores the model and optional EMA weights with the lowest validation
loss, plus the run configuration. `last/` is an Accelerate state directory
written every `trainer.checkpoint_every` updates; it holds the model, optimizer,
scheduler, optional EMA, optional gradient scaler, RNG states, and completed
iteration count.

```bash
python main.py resume_from=/absolute/path/to/checkpoints/last
```

Resume with matching architecture, optimizer, scheduler, EMA settings, and
number of processes. `trainer.iterations` is the desired total number of
updates. The cosine schedule spans `trainer.iterations`, so raising it on resume
changes the remaining LR curve. The resumed run copies `best.pth` from the
original run. The shuffle order is derived from the seed and epoch, and each
rank resumes from its saved DataLoader position, so a resumed run sees the same
batches as an uninterrupted one. With `num_workers > 0`, a resume exactly at an
epoch boundary reshuffles that epoch. Resume requires
`accelerator.dataloader_config.use_stateful_dataloader=true`. Evaluation reconstructs
its model from the checkpoint configuration, including `Dataset.model_kwargs()`,
and uses the current evaluation dataset and pipeline settings.

## Checks

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The tests use a small CPU fixture and real offline W&B runs. They cover training,
evaluation, resume, EMA, scheduling, mandatory tracking, and metric aggregation.
