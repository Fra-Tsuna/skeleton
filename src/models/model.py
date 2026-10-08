from torch import nn

#TODO Implement here your model!
class Model(nn.Module):
    """Project extension point. Keep the architecture independent of the trainer."""

    def __init__(self):
        super().__init__()
        raise NotImplementedError("Implement your architecture in src/models/model.py.")

    def forward(self, batch):
        raise NotImplementedError("Implement the forward pass for your batch format.")

    def trainable_parameters(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)
