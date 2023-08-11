#%%
from importlib import reload
import pathlib
import os
os.chdir(f"{pathlib.Path(__file__).parent.resolve()}/..")

import mlflow
from mlflow.client import MlflowClient
import modlee_pypi

from modlee_pypi.rep import Rep
from modlee_pypi.dev_data import get_fashion_mnist
import lightning.pytorch as pl

import torch
from torchinfo import summary
import numpy as np


#%%
client = MlflowClient()
experiments = client.search_experiments()
runs = client.search_runs(experiments[0].experiment_id)
run = runs[0]

# %%
rep = Rep.from_run(run)
model = rep.model

#%%
training_loader,test_loader = get_fashion_mnist()

#%%
with mlflow.start_run() as run:
    trainer = pl.Trainer(max_epochs=2,)
    trainer.fit(
        model=model,
        train_dataloaders=training_loader,
        val_dataloaders=test_loader)

# %%
rep_model = model
mlflow_model = mlflow.pytorch.load_model(f"{run.info.artifact_uri}/model").to(rep_model.device)
# %%
for rep_param,mlflow_param in zip(rep_model.parameters(),mlflow_model.parameters()):
    # print(rep_param,mlflow_param)
    rep_param = mlflow_param
    print(rep_param-mlflow_param)
# %%
# rep_model.parameters()[0]
rep_model.state_dict()

# %%
inputs,classes = next(iter(training_loader))
inputs = inputs.to(rep_model.device)
# torch.max(inputs)
with torch.no_grad():
    y_rep = rep_model(inputs[:10])
    y_mlflow = mlflow_model(inputs[:10])
    
print(f"abs(max(y_modlee-y_mlflow): {torch.abs(torch.max(y_rep - y_mlflow))}")
# %%
# mlflow_model.
# %%
