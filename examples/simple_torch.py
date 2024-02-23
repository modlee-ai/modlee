# %%
import lightning.pytorch as pl
import torch.nn.functional as F
import torch.nn as nn
import torch
import os
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

import modlee

modlee.init(api_key="local")
from modlee.utils import get_fashion_mnist
from modlee.model import ModleeModel

# %% Build models


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
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
        x = F.softmax(x)
        return x


class LightningClassifier(ModleeModel):
    def __init__(self, classifier=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not classifier:
            self.classifier = Classifier()
        else:
            self.classifier = classifier

    def forward(self, x):
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_out = self(x)
        loss = F.cross_entropy(y_out, y)
        return {"loss": loss}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_out = self(x)
        loss = F.cross_entropy(y_out, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        return optimizer


# %% Load data
training_loader, test_loader = get_fashion_mnist()
num_classes = len(training_loader.dataset.classes)
model = LightningClassifier()

# %% Run training loop
with modlee.start_run() as run:
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(
        model=model, train_dataloaders=training_loader, val_dataloaders=test_loader
    )

# %%
