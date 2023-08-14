from functools import partial
import pickle
from typing import Any
import numpy as np

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

import lightning.pytorch as pl
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT

import modlee_pypi
from modlee_pypi.data_stats import DataStats
from modlee_pypi.config import TMP_DIR
import mlflow


class ModleeModel(LightningModule):
    def __init__(self, data_snapshot_size=1e7, *args, **kwargs) -> None:
        LightningModule.__init__(self, *args, **kwargs)
        mlflow.pytorch.autolog(
            log_datasets=True,
        )
        self.data_snapshot_size = data_snapshot_size

    def configure_callbacks(self):
        return [ModleeCallback(self.data_snapshot_size)]


class ModleeCallback(Callback):
    def __init__(self, data_snapshot_size=1e7, *args, **kwargs):
        Callback.__init__(self, *args, **kwargs)
        self.data_snapshot_size = data_snapshot_size
        self.on_train_batch_end = partial(self._on_batch_end, phase='train')
        self.on_validation_batch_end = partial(self._on_batch_end, phase='val')

    def _on_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, phase='train') -> None:
        if isinstance(outputs, dict):
            for output_key, output_value in outputs.items():
                pl_module.log(output_key, output_value)
        elif isinstance(outputs, list):
            for output_idx, output_value in outputs:
                pl_module.log(
                    f"{phase}_step_output_{output_idx}", output_value)
        elif outputs is not None:
            pl_module.log(f"{phase}_loss", outputs)
        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        # log the code text as a python file
        code_text = modlee_pypi.get_code_text_for_model(
            pl_module, include_header=True)
        mlflow.log_text(code_text, 'model.py')
        return super().setup(trainer, pl_module, stage)

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        
        labels = []
        data = trainer.train_dataloader.dataset.data.numpy()
        self.save_snapshot(data,'data')
        mlflow_data = mlflow.data.from_numpy(data)
        mlflow.log_input(mlflow_data, 'training_snapshot')
        if hasattr(trainer.train_dataloader.dataset, 'targets'):
            targets = trainer.train_dataloader.dataset.targets
            self.save_snapshot(targets,'targets',max_len=len(data))
        
        # log the data statistics
        data_stats = DataStats(x=data,y=labels)
        mlflow.log_dict(data_stats.data_stats, 'data_stats')
        
        mlflow.log_param('batch_size', trainer.train_dataloader.batch_size)
        return super().on_train_start(trainer, pl_module)
    
    def save_snapshot(self, data, snapshot_name='data', max_len=None):
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        
        if max_len is None:
            data_size = data.nbytes
            # take a slice that should be no larger than 10MB
            max_len = int(np.min([
                (len(data)*self.data_snapshot_size)//data_size,
                len(data),
                ]))

        data = data[:max_len]
        data_filename = f"{TMP_DIR}/{snapshot_name}_snapshot.npy"
        np.save(data_filename, data)
        mlflow.log_artifact(data_filename) 

class ModleeDatasetWrapper():
    def __init__(self,dataset):
        self.dataset = dataset
        
    def __getitem__(self):
        pass
    
# class ModleeDataset(Dataset):
#     def __init__(self,**kwargs):
#         self.__dict__.update(**kwargs)
#         pass
    
#     @classmethod
#     def from_dataset(cls, dataset, **kwargs):
        
    