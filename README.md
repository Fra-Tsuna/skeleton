# PyTorch project skeleton

A reusable training and evaluation skeleton built with PyTorch, Hydra,
Accelerate, Diffusers, and W&B. It provides configuration, iteration-based
training, validation, EMA, logging, and checkpoints. You supply the dataset,
model, loss, and evaluation metrics. The default configuration is intentionally
not a runnable experiment: those project-specific methods raise
`NotImplementedError` until you implement them.

## Set up a project

Create a Conda environment, then install the dependencies from the repository's
`requirements.txt`. Choose a PyTorch build suited to your hardware if the one
installed from that file is not appropriate.

```bash
conda create -n my_project python=3.12 pip -y
conda activate my_project
python -m pip install -r requirements.txt
```

W&B is required. Log in for online runs with `wandb login`, or set
`wandb.mode=offline` when running locally. Accelerate and torchdata are also
required; torchdata provides the stateful DataLoader used for mid-epoch resume.
Diffusers provides the learning-rate schedulers and EMA implementation.

Implement these four extension points:

| File | What to implement |
| --- | --- |
| `src/dataset/dataset.py` | Load a split and implement `__len__` and `__getitem__`. Optionally return model constructor arguments from `model_kwargs()`. |
| `src/models/model.py` | Define the model and its `forward(batch)` method. `trainable_parameters()` supplies the optimizer parameters. |
| `src/trainer/trainer.py` | Implement `TrainingPipeline.compute_loss(batch)`. Return a dictionary containing a scalar, differentiable `loss` tensor and any other scalar tensor metrics. |
| `src/inference/inference.py` | Implement `InferencePipeline.compute_metrics(predictions, batch)`. Override `predict` if inference needs custom model calls or sampling. |

Put constructor arguments in `config/model/` and `config/dataloader/dataset/`.
Values returned by `Dataset.model_kwargs()` augment the model configuration and
are saved in the best-model checkpoint, so evaluation can rebuild the model.
The methods above are hooks: you can subclass the pipelines and select your
subclasses through their Hydra `_target_` settings.

Training and evaluation metrics should be **means over each batch** with stable
keys. Validation and evaluation weight those means by sample count, including
the smaller final batch. Override `metric_weight(batch)` in the relevant
pipeline if your metric is averaged over tokens, pixels, or another unit.
`compute_loss()` must return a `loss` key because training backpropagates it
and validation uses it to select the best checkpoint.

## Configure and run

Inspect the composed configuration before implementing the project hooks:

```bash
python main.py --cfg job
python eval.py --cfg job
```

After implementing them, start training or override settings from the command
line:

```bash
python main.py
python main.py name=my_project trainer.iterations=20000 optimizer.lr=3e-4
python main.py accelerator.cpu=true wandb.mode=offline
python main.py accelerator.mixed_precision=bf16 trainer.use_ema=true
python main.py lr_scheduler=constant
```

`trainer.iterations` counts successful optimizer updates. The default cosine
schedule warms up for `lr_scheduler.num_warmup_steps` updates and decays over
`trainer.iterations` total updates. `lr_scheduler=constant` keeps the learning
rate fixed. A skipped mixed-precision optimizer update does not advance the
scheduler, EMA, or iteration counter. Mixed-precision support depends on your
hardware. This trainer requires `gradient_accumulation_steps=1`.

The training DataLoader batch size is **per process**. For multiple processes,
configure Accelerate for your machine and give every process the same explicit
Hydra output directory:

```bash
accelerate config
accelerate launch --num_processes 2 main.py \
  hydra.run.dir=/shared/path/to/my_run
```

Validation and evaluation shard the dataset without adding duplicate samples.
The unwrapped model allows processes to finish uneven shards independently;
their metric totals are combined across processes. There is no cluster job
launcher in this repository. Accelerate retains device and precision state for
the life of a process, so Hydra multiruns should not sweep `accelerator`
settings within the same process.

## Logging, checkpoints, and resume

W&B starts for both training and evaluation. Its default mode is online; use
`wandb.mode=offline` for local runs. `wandb.mode=disabled` is rejected. Hydra
keeps the working directory fixed and writes each run under `outputs/` by
default. A training run contains:

```text
outputs/<name>/<date>/<time>/
  .hydra/                 Hydra configuration and overrides
  config.yaml             Resolved run configuration
  wandb/                  W&B run files
  checkpoints/
    best.pth              Best model for standalone evaluation
    last/                 Latest resumable training state
```

Validation runs every `trainer.validate_every` updates and at the final update.
`best.pth` is updated when the globally aggregated validation loss improves. It
contains model weights, optional EMA weights, the model configuration, and the
best loss. Enable EMA with `trainer.use_ema=true`; validation then uses EMA
weights. `trainer.patience` counts validation checks without improvement;
`null` disables early stopping.

`last/` is a resumable Accelerate state directory, saved every
`trainer.checkpoint_every` updates and at the final update or early stop. It
contains the model, optimizer, scheduler, RNG, optional EMA and gradient-scaler
state, the completed update count, and each process's DataLoader position. A
stateful DataLoader lets a resumed run continue partway through an epoch rather
than starting its sampling from the beginning. Resume requires
`accelerator.dataloader_config.use_stateful_dataloader=true`.

```bash
python main.py resume_from=/absolute/path/to/checkpoints/last \
  trainer.iterations=12000
```

Use the same model, optimizer, scheduler, EMA settings, and number of processes
as the original run. `trainer.iterations` is the **total** desired update count,
not the number of extra updates. Increasing it on resume also changes the
remaining cosine learning-rate curve. The resumed run copies the previous
`best.pth`. With `num_workers > 0`, resuming exactly at an epoch boundary can
reshuffle that epoch.

## Evaluate a model

Use `best.pth` for standalone evaluation; `last/` is for resuming training.

```bash
python eval.py checkpoint_path=/absolute/path/to/checkpoints/best.pth
python eval.py checkpoint_path=/absolute/path/to/checkpoints/best.pth apply_ema=true
```

The model is rebuilt from the saved model configuration. The evaluation
DataLoader and inference pipeline come from the **current** evaluation config,
so provide matching dataset or experiment overrides when needed.
`apply_ema=true` requires a checkpoint produced with EMA enabled. Evaluation
logs metrics to W&B and writes `metrics.json` under
`outputs/<name>/eval/<date>/<time>/` by default.

`InferencePipeline.on_prediction()` is an optional output hook. In a
multi-process run, it is called only for batches on the main process; global
metrics still include every process's shard.

## Repository layout

```text
main.py, eval.py            Training and evaluation entrypoints
config/                     Hydra groups for data, model, training, and tracking
src/dataset/, src/models/   Project-specific dataset and model hooks
src/trainer/                Training loop and checkpoint manager
src/inference/              Evaluation pipeline
src/utils/                  DataLoader, metric, logging, and figure helpers
tests/                      Small CPU fixtures and infrastructure checks
scripts/, assets/           Project scripts and documentation media
data/, ckpt/               Ignored local data and selected checkpoints
```

## Check the infrastructure

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The tests use a small synthetic CPU dataset and offline W&B runs. They exercise
training, evaluation, resume, EMA, scheduling, and metric aggregation without
requiring project data.
