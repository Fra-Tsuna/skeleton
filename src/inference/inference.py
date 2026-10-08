import torch

from src.utils.torch_utils import aggregate_metrics, batch_size


class InferencePipeline:
    """Run task-specific inference and aggregate evaluation metrics."""

    def __init__(self, model, dataloader, accelerator):
        self.model = accelerator.unwrap_model(model)
        self.dataloader = dataloader
        self.accelerator = accelerator

    def predict(self, batch):
        """Run the model on a batch; override for task-specific prediction logic."""
        return self.model(batch)

    def compute_metrics(self, predictions, batch) -> dict:
        """Return scalar batch means as tensors or Python numbers."""
        raise NotImplementedError("Implement InferencePipeline.compute_metrics for your project.")

    def metric_weight(self, batch):
        # Weight batch-mean metrics by the number of samples in the batch.
        return batch_size(batch)

    def on_prediction(self, predictions, batch, batch_index):
        """Optional hook for saving predictions or qualitative figures."""
        pass

    @torch.inference_mode()
    def run(self):
        # Disable training behavior and gradient tracking during evaluation.
        self.model.eval()

        # Accumulate weighted metric sums and sample counts on each process.
        totals = {}
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