
#%%
from importlib import reload
import pathlib
import os
os.chdir(f"{pathlib.Path(__file__).parent.resolve()}/..")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import lightning.pytorch as pl

import mlflow

import modlee_pypi
from modlee_pypi.modlee_model import ModleeModel
from modlee_pypi.dev_data import get_fashion_mnist
reload(modlee_pypi)

os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='1'
mps = torch.device('mps')

#%%
print(os.getcwd())

#%%
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
        
    '''
    def training_step(self, batch, batch_idx)
    
    @training_step
    def classification_training_step(self, batch, batch_idx):
        x,y = batch
        y_out = self.forward(x)
    '''
    
    def forward(self,x):
        return self.classifier(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
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

#%%
training_loader,test_loader = get_fashion_mnist()

#%%
training_loader.__dict__.keys()
training_loader.dataset.__dict__.keys()
import glob
print(glob.glob(f"./{training_loader.dataset.root}/*/*"))
images,labels = next(iter(training_loader))
#%%
# type(images)
labels
# ds = mlflow.data.Dataset(training_oader)
# ds.load(training_loader)
#%%
ds.name
# print(ds.profile)
# ds.profile = {'test':5}

#%%
# Run training loop
with mlflow.start_run() as run:
    trainer = pl.Trainer(max_epochs=2,)
    trainer.fit(
        model=model,
        train_dataloaders=training_loader,
        val_dataloaders=test_loader)

#%%
exp = mlflow.search_experiments()[0]
runs = mlflow.search_runs(output_format='list')
runs_pd = mlflow.search_runs()
run = runs[0]

#%%
# reload(modlee_pypi.rep

from modlee_pypi.rep import Rep
model = None
rep = Rep.from_run(run)
print(rep.info.run_id)
model = rep.model
rep_loaded_model = rep.model
print(model)

# lit_model = modlee_model.ModleeModel()
#%%
print(run.info.artifact_uri)
mlflow_loaded_model = mlflow.pytorch.load_model(
    f"{run.info.artifact_uri}/model/"
)
#%%
print('mlflow-loaded model\n'.upper(),mlflow_loaded_model)
print('rep-loaded model:\n'.upper(),rep_loaded_model)
print('String representations same? ',str(mlflow_loaded_model)==str(rep_loaded_model))

#%%
print(rep.data)
print(rep.info.run_id)
run_id = rep.info.run_id
# print(rep.data_stats['dataset_dims'])
# type(rep.data_stats)

#%%
# # %%
# loss_fn = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(
#     lit_model.parameters(),
#     lr=0.001,
#     momentum=0.9
# )
# # %%
# n_epochs = 10
# for epoch in range(n_epochs):
#     for train_batch in training_loader:
#         x_train,y_train = train_batch
#         x_train.to(mps), y_train.to(mps)
        
#         optimizer.zero_grad()
        
#         y_out = model(x_train)
#         loss = loss_fn(y_out,y_train)
#         loss.backward()
        
#         optimizer.step()
        
        
# # %%

# %%
