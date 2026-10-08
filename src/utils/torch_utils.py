import random

import numpy as np
import torch
from accelerate.data_loader import prepare_data_loader
from accelerate.utils import gather_object
from hydra.utils import instantiate

MiB = 1024 ** 2


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def make_dataloader(config, seed):
    generator = torch.Generator().manual_seed(seed)
    return instantiate(config, generator=generator, worker_init_fn=seed_worker)


def prepare_eval_dataloader(dataloader, accelerator):
    """Shard evaluation without duplicate samples; the unwrapped model tolerates uneven shards."""
    return prepare_data_loader(
        dataloader, device=accelerator.device, num_processes=accelerator.num_processes,
        process_index=accelerator.process_index, put_on_device=True, even_batches=False,
    )


def aggregate_metrics(totals, weight):
    shards = gather_object([{"totals": totals, "weight": weight}])
    total_weight = sum(shard["weight"] for shard in shards)
    if total_weight == 0:
        raise ValueError("Evaluation dataloader must contain samples.")
    keys = set().union(*(shard["totals"] for shard in shards))
    return {key: sum(shard["totals"].get(key, 0.0) for shard in shards) / total_weight for key in keys}


def model_size_b(model):
    return sum(tensor.nelement() * tensor.element_size() for tensor in (*model.parameters(), *model.buffers()))


def batch_size(batch):
    """Infer B from the first batched tensor. Override metric_weight if needed."""
    if torch.is_tensor(batch) and batch.ndim > 0:
        return batch.shape[0]
    values = batch.values() if isinstance(batch, dict) else batch if isinstance(batch, (list, tuple)) else []
    for value in values:
        try:
            return batch_size(value)
        except ValueError:
            continue
    raise ValueError("Cannot infer batch size. Override metric_weight for your batch format.")


def cycle(iterable):
    while True:
        found_batch = False
        for batch in iterable:
            found_batch = True
            yield batch
        if not found_batch:
            raise ValueError("Cannot cycle an empty dataloader.")
