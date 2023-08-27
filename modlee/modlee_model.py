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

import modlee
from modlee import logging
from modlee.api_client import ModleeAPIClient

from modlee.config import TMP_DIR, MLRUNS_DIR
from modlee import utils as modlee_utils
import mlflow


class ModleeModel(LightningModule):
    def __init__(self, data_snapshot_size=10e6, *args, **kwargs) -> None:
        """
        data_snapshot_size is the size limit of the chunk of data saved in each experiment
        We save this chunk in case we modify the calculation of data statistics (e.g. complexity)
        in future releases. Recalculating these measures enables backwards compatibility of 
        prior experiments
        """
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
        self._log_code_text(pl_module=pl_module)
        return super().setup(trainer, pl_module, stage)
    
    def _log_code_text(self, pl_module: LightningModule):
        _get_code_text_for_model = getattr(modlee, 'get_code_text_for_model', None)
        if _get_code_text_for_model is not None:
            code_text = modlee.get_code_text_for_model(
                pl_module, include_header=True)
            mlflow.log_text(code_text, 'model.py')
        else: 
            logging.warning("Could not access model-text converter, not logging but continuing experiment")
            
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        
        data,targets = self._get_data_targets(trainer)
        # log the data statistics
        self._log_data_stats(data,targets)
        
        mlflow.log_param('batch_size', trainer.train_dataloader.batch_size)
        return super().on_train_start(trainer, pl_module)
    
    def _log_data_stats(self,data,targets=[]) -> None:
        DataStats = getattr(modlee.data_stats, 'DataStats', None)
        if DataStats is not None:
            if isinstance(data, torch.Tensor):
                data,targets = data.numpy(), targets.numpy()
            data_stats = DataStats(x=data,y=targets)
            mlflow.log_dict(data_stats.data_stats, 'data_stats')
        else:
            logging.warning("Could not access data statistics calculation from server, not logging but continuing experiment")
        
    def _get_data_targets(self,trainer:Trainer):    
        _dataset = trainer.train_dataloader.dataset
        if isinstance(_dataset,list):
            data = np.array(_dataset)
        elif isinstance(_dataset, torch.utils.data.dataset.IterableDataset):
            data = list(_dataset)
            # data = np.array(list(_dataset))
        else:
            if isinstance(_dataset, torch.utils.data.Subset):
                _dataset = _dataset.dataset
            data = _dataset.data
            if isinstance(data, torch.Tensor):
                data = data.numpy()
            # data = _dataset.data.numpy()
            
        self._save_snapshot(data,'data')

        targets = []
        if hasattr(_dataset, 'targets'):
            targets = _dataset.targets
            self._save_snapshot(targets,'targets',max_len=len(data))
        return data,targets
        
    def _save_snapshot(self, data, snapshot_name='data', max_len=None):
        data = self._get_snapshot(data=data,max_len=max_len)
        modlee_utils.safe_mkdir(TMP_DIR)
        print(f"Making directory {TMP_DIR}")
        data_filename = f"{TMP_DIR}/{snapshot_name}_snapshot.npy"
        np.save(data_filename, data)
        mlflow.log_artifact(data_filename) 
        
    def _get_snapshot(self,data,max_len=None):
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        if max_len is None:
            data_size = data.nbytes
            # take a slice that should be no larger than 10MB
            max_len = int(np.min([
                (len(data)*self.data_snapshot_size)//data_size,
                len(data),
                ]))
        return data[:max_len]
        
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
        
    