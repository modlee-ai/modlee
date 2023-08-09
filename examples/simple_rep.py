#%%
from importlib import reload
import os
# TODO - get rid of this hardcoding
os.chdir('/Users/psychomugs/Developer/modlee/modlee_pypi/')
import mlflow
from mlflow.client import MlflowClient
import modlee_pypi
from modlee_pypi.rep import Rep
from modlee_pypi.dev_data import get_fashion_mnist
import lightning.pytorch as pl

#%%
client = MlflowClient()
experiments = client.search_experiments()
runs = client.search_runs(experiments[0].experiment_id)
run = runs[0]

# %%
rep = Rep.from_run(run)
model = rep.model
print(run.info.run_id)
print(model)

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
