"""Small CPU fixtures for infrastructure tests; never used by default configs."""

import torch
from torch import nn
from torch.utils.data import Dataset

from src.inference.inference import InferencePipeline
from src.trainer.trainer import TrainingPipeline


class TestDataset(Dataset):
    __test__ = False

    def __init__(self, data_root, split):
        self.x = torch.arange(1, 6, dtype=torch.float32).unsqueeze(1) / 5
        self.y = 2 * self.x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return {"x": self.x[index], "y": self.y[index], "metadata": "sample"}

    def model_kwargs(self):
        return {"input_dim": 1}


class TestModel(nn.Module):
    __test__ = False

    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, batch):
        return self.linear(batch["x"])

    def trainable_parameters(self):
        return self.parameters()


class TestTrainer(TrainingPipeline):
    __test__ = False

    def compute_loss(self, batch):
        loss = ((self.model(batch) - batch["y"]) ** 2).mean()
        return {"loss": loss}


class TestInference(InferencePipeline):
    __test__ = False

    def compute_metrics(self, predictions, batch):
        return {"mse": ((predictions - batch["y"]) ** 2).mean()}
