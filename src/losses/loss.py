import torch
from torch import nn
from torchvision import transforms
from torch import autograd
from pytorch_metric_learning.losses import BaseMetricLossFunction
from typing import Dict
from PIL import Image
import numpy as np

class BaseLoss(nn.Module):
    def __init__(self, base_loss_func: BaseMetricLossFunction):
        super().__init__()
        self.base_loss_func = base_loss_func

    def forward(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        masks: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # Returns the base loss
        embeddings = model(inputs)
        loss = self.base_loss_func(embeddings, labels)
        raise {
            "loss": loss
        }
