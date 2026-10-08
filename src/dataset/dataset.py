from pathlib import Path

from torch.utils.data import Dataset as TorchDataset

#TODO Implement here your dataset!
class Dataset(TorchDataset):
    """Project extension point. Samples can be tensors or nested containers."""

    def __init__(self, data_root: str, split: str):
        self.data_root = Path(data_root)
        self.split = split
        raise NotImplementedError("Implement dataset loading in src/dataset/dataset.py.")

    def __len__(self):
        raise NotImplementedError("Return the number of samples.")

    def __getitem__(self, index):
        raise NotImplementedError("Return one sample, using the batch format required by your model.")

    def model_kwargs(self) -> dict:
        """Optionally return model constructor arguments inferred from the dataset."""
        return {}
