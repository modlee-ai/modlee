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

# ptorch out-of-box libs needed
from torchvision.utils import _log_api_usage_once

from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Literal

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from torchvision import models
import torchvision
import torchmetrics

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


class ModleeModel(modlee.model.ModleeModel):
    def __init__(self, model=None, loss_fn=F.cross_entropy, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = model
        self.loss_fn = loss_fn

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_out = self(x)
        loss = self.loss_fn(y_out, y)
        return {"loss": loss}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_out = self(x)
        loss = self.loss_fn(y_out, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        return optimizer
