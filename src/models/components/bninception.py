from torch import nn
import torch.nn.functional as F
from pytorch_metric_learning.utils import common_functions
import pretrainedmodels


class BNInception(nn.Module):
    def __init__(self, num_outputs: int):
        super().__init__()

        # Set trunk model and replace the softmax layer with an identity function
        self.trunk = pretrainedmodels.bninception(pretrained="imagenet")
        self.trunk.last_linear = common_functions.Identity()
        self.embedder = nn.Linear(1024, num_outputs)

        # Freeze BatchNorm during training to avoid overfitting
        # self.trunk.apply(common_functions.set_layers_to_eval("BatchNorm"))
        # -> We need to call this at the start of every epoch to ensure that the BN layers are set to eval, see DMLModule

    def forward(self, x):
        x = self.trunk(x)
        x = self.embedder(x)
        # normalize embedding vector to length one
        x = F.normalize(x, p=2, dim=1)
        return x
