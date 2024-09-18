"""
Modlee model class and callbacks. 
"""
from functools import partial, partialmethod
import inspect
import pickle
from typing import Any, Optional
import numpy as np

import os
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset

import lightning.pytorch as pl
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT

import modlee
from modlee import (
    data_metafeatures,
    save_run,
    get_code_text_for_model,
    save_run_as_json,
)
from modlee import logging, utils as modlee_utils
from modlee.converter import Converter
from modlee.model.callbacks import *

modlee_converter = Converter()

from modlee.config import TMP_DIR, MLRUNS_DIR

import mlflow
import json

base_lightning_module = LightningModule()
base_lm_keys = list(LightningModule.__dict__.keys())


class ModleeModel(LightningModule):
    def __init__(
        self,
        data_snapshot_size=10e6,
        kwargs_cache={},
        modality=None,
        task=None,
        *args,
        **kwargs,
    ) -> None:
        """
        ModleeModel constructor.

        :param data_snapshot_size: The maximum size of the data snapshots saved during meta-feature calculation.
        :param vars_cache: A dictionary of variables to cache for later rebuilding the model.
        """
        LightningModule.__init__(self, *args, **kwargs)
        mlflow.pytorch.autolog(log_datasets=False)
        self.data_snapshot_size = data_snapshot_size
        modality, task = modlee_utils.get_modality_task(self)
        self.modality = modality
        self.task = task
        self.kwargs_cache = kwargs_cache
        self.kwargs_cache.update(kwargs)

        self_keys = list(self.__dict__.keys())
        self.data_metafeatures_callback = DataMetafeaturesCallback(
            DataMetafeatures=self._get_data_metafeature_class()
        )
        self.model_metafeatures_callback = ModelMetafeaturesCallback(
            ModelMetafeatures=self._get_model_metafeature_class()
        )
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

    def _check_step_defined(self, method_name):
        """
        Check if a step method (i.e. training_step, validation_step, test_step) is defined.

        :param method_name: The name of the method to check.
        :return: Whether the method is defined.
        """
        method_params = inspect.signature(getattr(self, method_name)).parameters
        return "batch" in method_params

    def _get_metafeature_class(self, metafeature_type="data"):
        class_prefix = ""
        if self.modality is not None:
            class_prefix = f"{self.modality.capitalize()}"
        if metafeature_type == "model" and self.task is not None:
            class_prefix += f"{self.task.capitalize()}"

        class_name = f"{class_prefix}{metafeature_type.capitalize()}Metafeatures"

        mf_class = getattr(
            getattr(modlee, f"{metafeature_type.lower()}_metafeatures"),
            class_name,
            None,
        )
        if mf_class is None:
            logging.warning(
                f"No {metafeature_type} metafeatures implemented for {self.modality} {self.task}"
            )
        else:
            return mf_class

    _get_data_metafeature_class = partialmethod(
        _get_metafeature_class, metafeature_type="data"
    )
    _get_model_metafeature_class = partialmethod(
        _get_metafeature_class, metafeature_type="model"
    )

    def configure_callbacks(self):
        """
        Configure callbacks for auto-documentation.

        :return: A list of callbacks for auto-documentation.
        """
        if self.modality:
            dmf_metafeature_cls=\
                DataMetafeatures=getattr(modlee.data_metafeatures, f"{self.modality.capitalize()}DataMetafeatures")
        else:
            dmf_metafeature_cls=getattr(modlee.data_metafeatures, "DataMetafeatures")
        callbacks = [
            self.data_metafeatures_callback,
            self.model_metafeatures_callback,
            LogCodeTextCallback(self.kwargs_cache),
            LogOutputCallback(),
            LogParamsCallback(),
            PushServerCallback(),
            LogTransformsCallback(),
            LogModelCheckpointCallback(monitor="loss"),
        ]

        # # If the validation step is defined, add
        if self._check_step_defined("validation_step"):
            callbacks.append(LogModelCheckpointCallback(monitor="val_loss"))

        return callbacks


class SimpleModel(ModleeModel):
    """
    A simple Modlee model.
    """

    def __init__(self, input_shape=(1, 10), output_shape=(1, 20)):
        """
        Construct the model.

        :param input_shape: The model's input shape.
        :param output_shape: The model's output shape.
        """
        super().__init__()
        self.input_shape = input_shape
        self.model = nn.Linear(input_shape[-1], output_shape[-1])
        self.loss = nn.functional.cross_entropy
        self.dataset = SimpleDataset(input_shape, output_shape)

    def forward(self, x):
        """
        Forward pass.

        :param x: The input to the model.
        :return: The output of the model.
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Perform a training step.

        :param batch: The batch of dat.a
        :param batch_idx: The index of the batch.
        :return: A dictionary of the loss.
        """
        x, y = batch
        return {"loss": self.loss(self(x), y)}

    def configure_optimizers(self):
        """
        Configure optimizers for the model.

        :return: The optimizer object.
        """
        return torch.optim.Adam(self.parameters(), lr=0.001)


class SimpleDataset(Dataset):
    """
    A simple dataset class.
    """

    def __init__(self, input_shape=(1, 10), output_shape=(1, 20)):
        """
        Construct the simple dataset.

        :param input_shape: The shape of the input.
        :param output_shape: The shape of the output.
        """
        super().__init__()
        self.inputs = torch.rand(10, input_shape[-1])
        self.outputs = torch.rand(10, output_shape[-1])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.outputs[index]
