
import torch
import torch.nn as nn
import pytorch_lightning
import pytorch_lightning as pl
from torch.nn import functional as F
import lightning

import torch
import torch.nn as nn
import inspect

import numpy
import numpy as np

try:
    from torchmetrics.functional import accuracy
except ImportError:
    from pytorch_lightning.metrics.functional import accuracy

#ptorch out-of-box libs needed
from torchvision.utils import _log_api_usage_once

from functools import partial
from typing import Any, Callable, List, Optional, Sequence

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from torchvision import models
import torchvision

# from torchvision.ops.misc import Conv2dNormActivation, Permute
# from torchvision.ops.stochastic_depth import StochasticDepth
# from torchvision.transforms._presets import ImageClassification
# from torchvision.utils import _log_api_usage_once
# from torchvision.models._api import register_model, Weights, WeightsEnum
# from torchvision.models._meta import _IMAGENET_CATEGORIES
# from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface

# from torchvision.models.convnext import *
# from torchvision.models.convnext import CNBlockConfig

# Modlee imports
import modlee


class ModleeModel(modlee.modlee_model.ModleeModel):

    def __init__(self, classifier=None):
        super().__init__()
        if not classifier:
            self.classifier = GarmentClassifier()
        else:
            self.classifier = classifier

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        r"""Same as :meth:`torch.nn.Module.forward`.

        Args:
            *args: Whatever you decide to pass into the forward method.
            **kwargs: Keyword arguments are also possible.

        Return:
            Your model's output
        """
        return super().forward(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        y_out = self.classifier(x)
        loss = F.cross_entropy(y_out,y)
        return {'loss':loss}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=0.001,
            momentum=0.9
        )
        return optimizer



class GarmentClassifier(torch.nn.modules.module.Module):

    def __init__(self):
        super(GarmentClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


