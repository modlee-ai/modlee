from functools import partial
import pickle
from typing import Any
import numpy as np

import os
import pandas as pd
import torch
from torch.utils.data import Dataset

import lightning.pytorch as pl
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT

import modlee
from modlee import logging, \
    utils as modlee_utils
from modlee.config import TMP_DIR, MLRUNS_DIR

import mlflow
import json

base_lightning_module = LightningModule()
base_lm_keys = list(LightningModule.__dict__.keys())

class ModleeModel(LightningModule):
    def __init__(self, data_snapshot_size=10e6, vars_cache={}, *args, **kwargs) -> None:
        """
        data_snapshot_size is the size limit of the chunk of data saved in each experiment
        We save this chunk in case we modify the calculation of data statistics (e.g. complexity)
        in future releases. Recalculating these measures enables backwards compatibility of 
        prior experiments
        """
        LightningModule.__init__(self, *args, **kwargs)
        mlflow.pytorch.autolog(
            log_datasets=False,
        )
        self.data_snapshot_size = data_snapshot_size
        self.vars_cache = vars_cache
        self.vars_cache.update(kwargs)
        
        self_keys = list(self.__dict__.keys())
        # self_dict = self.__dict__.
        # for self_key in self_keys:
        
    def _update_vars_cached(self):
        for self_key,self_val in self.__dict__.items():
            if self_key=='vars_cache':
                continue
            if self_key[0]=='_':
                continue
            if self_key not in base_lm_keys:
                if modlee_utils.is_cacheable(self_val):
                    self.vars_cache.update({
                        self_key:self.__dict__[self_key]
                    })

    @property
    def run_dir(self):
        """Get the current run directory

        Returns:
            _type_: the directory to the mlruns/{experiment_id}/{run_id}
        """
        return os.path.dirname(
            modlee_utils.uri_to_path(mlflow.get_artifact_uri()))

    def configure_callbacks(self):
        return [
            DataStatsCallback(self.data_snapshot_size),
            LogCodeTextCallback(self.vars_cache),
            LogOutputCallback(),
            LogParamsCallback(),
            PushAPICallback(),
        ]


class PushAPICallback(Callback):
    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        modlee.save_run(pl_module.run_dir)
        return super().on_fit_end(trainer, pl_module)


class LogParamsCallback(Callback):
    def __init__(self, *args, **kwargs):
        Callback.__init__(self, *args, **kwargs)

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        mlflow.log_param('batch_size', trainer.train_dataloader.batch_size)
        return super().on_train_start(trainer, pl_module)


class LogCodeTextCallback(Callback):
    def __init__(self, vars_to_save={}, *args, **kwargs):
        Callback.__init__(self, *args, **kwargs)
        self.vars_cache = vars_to_save

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        # log the code text as a python file
        self._log_code_text(pl_module=pl_module)
        return super().setup(trainer, pl_module, stage)

    def _log_code_text(self, pl_module: LightningModule):
        _get_code_text_for_model = getattr(
            modlee, 'get_code_text_for_model', None)
        if _get_code_text_for_model is not None:
            code_text = modlee.get_code_text_for_model(
                pl_module, include_header=True)
            mlflow.log_text(code_text, 'model.py')
            pl_module._update_vars_cached()
            mlflow.log_dict(self.vars_cache, 'cached_vars')
        else:
            logging.warning(
                "Could not access model-text converter, \
                    not logging but continuing experiment")


class LogOutputCallback(Callback):
    def __init__(self, *args, **kwargs):
        Callback.__init__(self, *args, **kwargs)
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


class DataStatsCallback(Callback):
    def __init__(self, data_snapshot_size=1e7, DataStats=None, *args, **kwargs):
        Callback.__init__(self, *args, **kwargs)
        self.data_snapshot_size = data_snapshot_size
        if not DataStats:
            DataStats = getattr(modlee.data_stats, 'DataStats', None)
        self.DataStats = DataStats

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:

        data, targets = self._get_data_targets(trainer)
        # log the data statistics
        self._log_data_stats(data, targets)

        return super().on_train_start(trainer, pl_module)

    def _log_data_stats(self, data, targets=[]) -> None:
        if self.DataStats:
            if isinstance(data, torch.Tensor):
                data, targets = data.numpy(), targets.numpy()
            data_stats = self.DataStats(x=data, y=targets)
            mlflow.log_dict(data_stats.data_stats, 'data_stats')
        else:
            logging.warning(
                "Could not access data statistics calculation from server, \
                    not logging but continuing experiment")

    def _get_data_targets(self, trainer: Trainer):
        _dataset = trainer.train_dataloader.dataset
        if isinstance(_dataset, list):
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

        self._save_snapshot(data, 'data')

        targets = []
        if hasattr(_dataset, 'targets'):
            targets = _dataset.targets
            self._save_snapshot(targets, 'targets', max_len=len(data))
        return data, targets

    def _save_snapshot(self, data, snapshot_name='data', max_len=None):
        data = self._get_snapshot(data=data, max_len=max_len)
        modlee_utils.safe_mkdir(TMP_DIR)
        data_filename = f"{TMP_DIR}/{snapshot_name}_snapshot.npy"
        np.save(data_filename, data)
        mlflow.log_artifact(data_filename)

    def _get_snapshot(self, data, max_len=None):
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
