#%%
from importlib import reload
import pathlib
import os
os.chdir(f"{pathlib.Path(__file__).parent.resolve()}/..")

import mlflow
from mlflow.client import MlflowClient
import modlee
from modlee.config import ROOT_DIR
from modlee.dev_data import get_fashion_mnist
import lightning.pytorch as pl
#%%
mlflow.set_registry_uri(f"{ROOT_DIR / 'mlruns'}")
print(os.path.abspath(mlflow.get_registry_uri()))
#%%
client = MlflowClient()

def get_run(run_idx=0):
    experiments = client.search_experiments()
    print(experiments)
    runs = client.search_runs(experiments[0].experiment_id)
    print(runs)
    return runs[run_idx]

run = get_run(0)
rep = Rep.from_run(run)
rep_model = rep.model
mlflow_model = mlflow.pytorch.load_model(f"{run.info.artifact_uri}/models")

#%%
training_loader,test_loader = get_fashion_mnist()

#%%
with mlflow.start_run() as run:
    trainer = pl.Trainer(max_epochs=2,)
    trainer.fit(
        model=model,
        train_dataloaders=training_loader,
        val_dataloaders=test_loader)

