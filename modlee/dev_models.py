import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F

import torch
import torch.nn as nn
import inspect

import numpy as np
from torchvision.models import *

from torchvision import models
from torch.nn import *
import torch.nn as nn
import torch

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor


try:
    from torchmetrics.functional import accuracy
except ImportError:
    from pytorch_lightning.metrics.functional import accuracy


class Pass(nn.Module):
    def __init__(self):
        super(Pass, self).__init__()

    def forward(self, x):
        return x


class NetCustom(nn.Module):
    def __init__(self, input_dim=1,num_classes=10):
        super(NetCustom, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Pass(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

class NetCustom2(nn.Module):
    def __init__(self, num_classes):
        super(NetCustom2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Sequential(Pass()),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


class MNISTModel(pl.LightningModule):
    def __init__(self):
        super(MNISTModel,self).__init__()
        self.flatten = torch.nn.Flatten()
        self.l1 = torch.nn.Linear(28 * 28, 10)
        self.relu_1 = torch.nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.l1(x)
        x = self.relu_1(x)
        return x

    def training_step(self, batch, batch_nb):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        pred = logits.argmax(dim=1)
        acc = accuracy(pred, y)

        # Use the current of PyTorch logger
        self.log("train_loss", loss, on_epoch=True)
        self.log("acc", acc, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


class ConcatModule(nn.Module):
    def __init__(self):
        super(ConcatModule, self).__init__()

    def forward(self, x1, x2):
        return torch.cat((x1, x2), dim=1)

class AddModule(nn.Module):
    def __init__(self):
        super(AddModule, self).__init__()

    def forward(self, x1, x2):
        return x1 + x2

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(10, 5)
        self.relu = nn.ReLU()
        self.concat = ConcatModule()
        self.add = AddModule()
        self.fc3 = nn.Linear(5, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        x1 = self.fc1(x1)
        x2 = self.fc2(x2)
        x = x1 + x2
        # x = self.add(x, x1)  # Element-wise addition
        # x = self.concat(x1, x2)
        x = torch.cat((x, x2), dim=1)
        x = self.relu(x1)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

class MyModelCustom(nn.Module):
    def __init__(self):
        super(MyModelCustom, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(10, 5)
        self.relu = nn.ReLU()
        self.concat = ConcatModule()
        self.add = AddModule()
        self.fc3 = nn.Linear(5, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        x1 = self.fc1(x1)
        x2 = self.fc2(x2)
        # x = x1 + x2
        x = self.add(x1, x1)  # Element-wise addition
        x = self.concat(x, x2)
        # x = torch.cat((x, x2), dim=1)
        x = self.relu(x1)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

class OutOfBoxModel1(nn.Module):
    def __init__(self):
        super(OutOfBoxModel1, self).__init__()
        self.pretrained = models.alexnet(pretrained=True)

    def forward(self, x):
        return self.pretrained(x)


class OutOfBoxModel2(nn.Module):
    def __init__(self,dropout=0.1,num_classes=10):
        super(OutOfBoxModel2, self).__init__()
        self.pretrained = models.alexnet(pretrained=True)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.SELU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 1224),
            nn.SELU(inplace=True),
            nn.Linear(1224, num_classes),
        )

    def forward(self, x):
        x = self.pretrained.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



#NEED TO ADD TABULAR MODELS



#NEED TO ADD TRANSFORMER MODELS FROM HUGGING FACE


def HFAutoTransformer(model_checkpoint,num_labels):
  from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

  return AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=num_labels)









