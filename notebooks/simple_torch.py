
#%%
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
# for enable GPU
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='1'
if torch.cuda.is_available():
    torch.set_default_device('cuda')
elif torch.backends.mps.is_available():
    torch.set_default_device('mps')

import modlee_pypi
from modlee_pypi.modlee_model import ModleeModel
from modlee_pypi.dev_data import get_fashion_mnist


#%% Build models
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
        return x

class LightningClassifier(ModleeModel):
    def __init__(self, classifier=None):
        super().__init__()
        if not classifier:
            self.classifier = Classifier()
        else:
            self.classifier = classifier

    def forward(self,x):
        return self.classifier(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_out = self(x)
        loss = F.cross_entropy(y_out,y)
        return {'loss':loss}
    
    def validation_step(self, val_batch, batch_idx):
        x,y=val_batch
        y_out = self(x)
        loss = F.cross_entropy(y_out,y)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=0.001,
            momentum=0.9
        )
        return optimizer
    
model = LightningClassifier()

#%% Load data
training_loader,test_loader = get_fashion_mnist()

#%% Run training loop
with modlee_pypi.start_run() as run:
    trainer = pl.Trainer(max_epochs=2,)
    trainer.fit(
        model=model,
        train_dataloaders=training_loader,
        val_dataloaders=test_loader)

# %%
