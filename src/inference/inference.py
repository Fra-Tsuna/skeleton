from typing import Any, Dict, Union

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from src.models.model import Model
from src.utils.torch_utils import aggregate_metrics, batch_size


class InferencePipeline:
    """Run task-specific inference and aggregate evaluation metrics."""

    def __init__(self, model: Model, dataloader: DataLoader, accelerator: Accelerator):
        self.model: Model = accelerator.unwrap_model(model)
        self.dataloader = dataloader
        self.accelerator = accelerator

    def predict(self, batch: Any) -> Any:
        """Run the model on a batch; override for task-specific prediction logic."""
        return self.model(batch)

    def compute_metrics(self, predictions: Any, batch: Any) -> Dict[str, Union[torch.Tensor, float]]:
        """Return scalar batch means as tensors or Python numbers."""
        raise NotImplementedError("Implement InferencePipeline.compute_metrics for your project.")

    def metric_weight(self, batch: Any) -> int:
        # Weight batch-mean metrics by the number of samples in the batch.
        return batch_size(batch)

    def on_prediction(self, predictions: Any, batch: Any, batch_index: int) -> None:
        """Optional hook for saving predictions or qualitative figures."""
        pass

    @torch.inference_mode()
    def run(self) -> Dict[str, float]:
        # Disable training behavior and gradient tracking during evaluation.
        self.model.eval()

        # Accumulate weighted metric sums and sample counts on each process.
        totals: Dict[str, float] = {}
        total_weight = 0

        for index, batch in enumerate(self.dataloader):
            # Use Accelerate's configured precision for prediction and metric computation.
            with self.accelerator.autocast():
                predictions = self.predict(batch)
                metrics = self.compute_metrics(predictions, batch)

            # Weight each batch mean so smaller batches contribute proportionally.
            weight = self.metric_weight(batch)
            total_weight += weight

            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value) * weight

            # Run the optional output hook only for batches handled by the main process.
            if self.accelerator.is_main_process:
                self.on_prediction(predictions, batch, index)

        # Combine metric sums and sample counts across processes into global means.
        return aggregate_metrics(totals, total_weight)