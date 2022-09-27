from torch import nn
import torch.nn.functional as F
from pytorch_metric_learning.utils import common_functions
import pretrainedmodels


class MLP(nn.Module):
    def __init__(self, num_outputs: int):
        super().__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_outputs)
        )

    def forward(self, x):
        return self.model(x)
