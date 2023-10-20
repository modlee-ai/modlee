from functools import partial
import pickle
from typing import Any, Optional
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
    utils as modlee_utils, \
    exp_loss_logger
from modlee.converter import Converter
modlee_converter = Converter()

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
            LogONNXCallback(),
        ]



class ModleeCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        
    def get_input(self, trainer, pl_module):
        """
        Retrieve an input from a trainer 

        Args:
            trainer (_type_): _description_
            pl_module (_type_): _description_

        Returns:
            _type_: _description_
        """
        _dataloader = trainer.train_dataloader
        _batch = next(iter(_dataloader))
        # NOTE - how can we generalize to different input schemes?
        # e.g. siamese network with multiple inputs
        # Right now, this makes the assumption that that only the network 
        # uses only the first element
        # NOTE - maybe using inspect.signature(pl_module.forward)
        # could help generalize to different forward() calls
        if type(_batch) in [list,tuple]:
            _input = _batch[0]
            # print(_batch[0].shape)
            # _batch = torch.Tensor(_batch[0])
        try:
            _input = _input.to(pl_module.device)
        except:
            pass
        return _input
        
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


class LogCodeTextCallback(ModleeCallback):
    def __init__(self, vars_to_save={}, *args, **kwargs):
        Callback.__init__(self, *args, **kwargs)
        self.vars_cache = vars_to_save

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        # log the code text as a python file
        # self._log_code_text(trainer=trainer, pl_module=pl_module)
        return super().setup(trainer, pl_module, stage)
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._log_code_text(trainer=trainer, pl_module=pl_module)
        
        return super().on_train_start(trainer, pl_module)

    def _log_code_text(self, trainer: Trainer, pl_module: LightningModule):
        _get_code_text_for_model = getattr(
            modlee, 'get_code_text_for_model', None)
        code_text = ''
        if _get_code_text_for_model is not None:
            # ==== METHOD 1 ====
            # Save model as code using parsing
            code_text = modlee.get_code_text_for_model(
                pl_module, include_header=True)
            mlflow.log_text(code_text, 'model.py')
            # Save variables required to rebuild the model
            pl_module._update_vars_cached()
            mlflow.log_dict(self.vars_cache, 'cached_vars')
            
            # ==== METHOD 2 ====
            # Save model as code by converting to a graph through ONNX
            input_dummy = self.get_input(trainer, pl_module)
            onnx_model = modlee_converter.torch2onnx(pl_module, input_dummy=input_dummy)
            onnx_text = modlee_converter.onnx2onnx_text(onnx_model)
            mlflow.log_text(onnx_text, 'model_graph.txt')
            torch_graph_code = modlee_converter.onnx_text2code(onnx_text)
            mlflow.log_text(torch_graph_code, 'model_graph.py')
            
            # Save model size
            model_size = modlee_utils.get_model_size(pl_module, as_MB=False)
            mlflow.log_text(str(model_size), 'model_size')
            
        else:
            logging.warning(
                "Could not access model-text converter, \
                    not logging but continuing experiment")

        if exp_loss_logger.module_available:
            _extract_loss_functions = getattr(
                exp_loss_logger, 'extract_loss_functions', None)
            
            if _extract_loss_functions is not None:
                loss_calls = exp_loss_logger.extract_loss_functions(
                    code_text)
                #logging.warning(loss_calls)
                #mlflow.log_text(code_text, 'model.py')
                if len(loss_calls) > 0:

                    loss_calls_str = str.join('\n', loss_calls)
                    mlflow.log_text(loss_calls_str, 'loss_calls.txt')
                else:
                    logging.warning("Could not record loss functions explicitly, \
                                    check for usage of custom loss definitions")
            else:
                logging.warning(
                "exp_loss_logger has no attribute extract_loss_functions")

        else:
            logging.warning(
                "Could not access exp_loss_logger, \
                    not logging but continuing experiment")
class LogONNXCallback(ModleeCallback):
    def __init__(self) -> None:
        super().__init__()
        
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        # self._log_onnx(trainer, pl_module)
        return super().setup(trainer, pl_module, stage)
    
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._log_onnx(trainer, pl_module)
        return super().on_fit_start(trainer, pl_module)
    
    def _log_onnx(self, trainer, pl_module, ):
        
        # train_input,_ = next(iter(trainer.train_dataloader))
        # print(train_input)
        # NOTE assumes that model input is the first output of a batch
        modlee_utils.safe_mkdir(TMP_DIR)
        data_filename = f"{TMP_DIR}/model.onnx"
        
        _input = self.get_input(trainer, pl_module)
        
        model_output = pl_module.forward(_input)
        torch.onnx.export(
            pl_module,
            # train_input,
            _input,
            data_filename,
            export_params=False
        )
        mlflow.log_artifact(data_filename)

        pass
        

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

class DataStatsCallback(ModleeCallback):
    def __init__(self, data_snapshot_size=1e7, DataStats=None, *args, **kwargs):
        Callback.__init__(self, *args, **kwargs)
        super().__init__()
        self.data_snapshot_size = data_snapshot_size
        if not DataStats:
            DataStats = getattr(modlee.data_stats, 'DataStats', None)
        self.DataStats = DataStats

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:

        # data, targets = self._get_data_targets(trainer)
        data_snapshots = self._get_snapshots_batched(trainer.train_dataloader)
        self._save_snapshots_batched(data_snapshots)
        # log the data statistics
        # self._log_data_stats(data, targets)
        self._log_data_stats_dataloader(trainer.train_dataloader)
        
        self._log_output_size(trainer, pl_module)

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

    def _log_data_stats_dataloader(self, dataloader) -> None:
        if self.DataStats:
            # TODO - use data batch and model to get output size
            data_stats = self.DataStats(dataloader)
            mlflow.log_dict(
                data_stats._serializable_stats_rep,
                'stats_rep')
        else:
            logging.warning(
                "Cannot log data statistics, could not access from server"
            )

    def _log_output_size(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # _dataloader = trainer.train_dataloader
        # _batch = next(iter(_dataloader))
        # # NOTE - how can we generalize to different input schemes?
        # # e.g. siamese network with multiple inputs
        # # Right now, this makes the assumption that that only the network 
        # # uses only the first element
        # # NOTE - maybe using inspect.signature(pl_module.forward)
        # # could help generalize to different forward() calls
        # if type(_batch) in [list,tuple]:
        #     _batch = _batch[0]
        #     # print(_batch[0].shape)
        #     # _batch = torch.Tensor(_batch[0])
        # try:
        #     _batch = _batch.to(pl_module.device)
        # except:
        #     pass
        
        _input = self.get_input(trainer, pl_module)
        
        try:
            _output = pl_module.forward(_input)
            output_shape = list(_output.shape[1:])
            mlflow.log_param("output_shape",output_shape)
        except:
            logging.warning(
                "Cannot log output shape, could not pass batch through network"
            )  

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
    
    def _get_data_targets_batch(self, trainer: Trainer):
        _dataloader = trainer.train_dataloader
        data_snapshot = np.array()
        

    def _save_snapshot(self, data, snapshot_name='data', max_len=None):
        data = self._get_snapshot(data=data, max_len=max_len)
        modlee_utils.safe_mkdir(TMP_DIR)
        data_filename = f"{TMP_DIR}/{snapshot_name}_snapshot.npy"
        np.save(data_filename, data)
        mlflow.log_artifact(data_filename)

    def _get_snapshot(self, data, max_len=None):
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        elif not isinstance(data, np.ndarray):
            data = np.array(data)
        if max_len is None:
            data_size = data.nbytes
            # take a slice that should be no larger than 10MB
            max_len = int(np.min([
                (len(data)*self.data_snapshot_size)//data_size,
                len(data),
            ]))
        return data[:max_len]
    
    def _save_snapshots_batched(self, data_snapshots):
        modlee_utils.safe_mkdir(TMP_DIR)
        for snapshot_idx, data_snapshot in enumerate(data_snapshots):
            
            data_filename = f"{TMP_DIR}/snapshot_{snapshot_idx}.npy"
            np.save(data_filename, data_snapshot)
            mlflow.log_artifact(data_filename)
    
    def _get_snapshots_batched(self, dataloader, max_len=None):
        # Use batch to determine how many "sub"batches to create
        _batch = next(iter(dataloader))

        data_snapshot_size = self.data_snapshot_size
        if type(_batch) in [list,tuple]:
            n_snapshots = len(_batch)
        else:
            n_snapshots = 1
        data_snapshots = [np.array([])]*n_snapshots

        # Keep appending to batches until the combined size reaches the limit
        batch_ctr = 0
        while np.sum([ds.nbytes for ds in data_snapshots])<data_snapshot_size:
            _batch = next(iter(dataloader))
            
            # If there are multiple elements in the batch, 
            # append to respective subbranches
            if type(_batch) in [list,tuple]:            
                for batch_idx, _subbatch in enumerate(_batch):
                    if str(_subbatch.device)!='cpu':
                        _subbatch = _subbatch.cpu()
                    if data_snapshots[batch_idx].size==0:
                        data_snapshots[batch_idx] = _subbatch.numpy()
                    else:
                        data_snapshots[batch_idx] = np.vstack([
                            data_snapshots[batch_idx],
                            (_subbatch.numpy())])
            else:
                if data_snapshots[0].size==0:
                        data_snapshots[0] = _batch.numpy()
                else:
                        data_snapshots[0] = np.vstack([
                            data_snapshots[0],
                            (_batch.numpy())])
            batch_ctr += 1
        return data_snapshots

