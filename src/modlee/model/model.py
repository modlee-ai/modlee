"""
Modlee model class and callbacks. 
"""
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
from modlee import data_metafeatures, save_run, get_code_text_for_model
from modlee import logging, utils as modlee_utils, exp_loss_logger
from modlee.converter import Converter
from modlee.model.callbacks import *

modlee_converter = Converter()

from modlee.config import TMP_DIR, MLRUNS_DIR

import mlflow
import json

base_lightning_module = LightningModule()
base_lm_keys = list(LightningModule.__dict__.keys())


class ModleeModel(LightningModule):
    def __init__(self, data_snapshot_size=10e6, kwargs_cache={}, *args, **kwargs) -> None:
        """
        ModleeModel constructor.
        
        :param data_snapshot_size: The maximum size of the data snapshots saved during meta-feature calculation.
        :param vars_cache: A dictionary of variables to cache for later rebuilding the model.
        """
        LightningModule.__init__(self, *args, **kwargs)
        mlflow.pytorch.autolog(log_datasets=False)
        self.data_snapshot_size = data_snapshot_size
        self.kwargs_cache = kwargs_cache
        self.kwargs_cache.update(kwargs)

        self_keys = list(self.__dict__.keys())
        # self_dict = self.__dict__.
        # for self_key in self_keys:

    def _update_kwargs_cached(self):
        """ 
        Update the cached variable dictionary with any custom arguments.
        """
        for self_key, self_val in self.__dict__.items():
            if self_key == "kwargs_cache":
                continue
            if self_key[0] == "_":
                continue
            if self_key not in base_lm_keys:
                if modlee_utils.is_cacheable(self_val):
                    self.kwargs_cache.update({self_key: self.__dict__[self_key]})

    @property
    def run_path(self):
        """
        The path to the current run.

        :return: The path to the current run. 
        """
        return os.path.dirname(modlee_utils.uri_to_path(mlflow.get_artifact_uri()))

    def configure_callbacks(self):
        """ 
        Configure callbacks for auto-documentation.

        :return: A list of callbacks for auto-documentation.
        """
        return [
            DataMetafeaturesCallback(self.data_snapshot_size),
            LogCodeTextCallback(self.kwargs_cache),
            LogOutputCallback(),
            LogParamsCallback(),
            PushServerCallback(),
            # LogONNXCallback(),
        ]

