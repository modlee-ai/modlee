from functools import partial

from typing import Any
import lightning.pytorch as pl
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT

import modlee_pypi
from modlee_pypi.data_stats import DataStats
import mlflow


class ModleeModel(LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        LightningModule.__init__(self, *args, **kwargs)
        mlflow.pytorch.autolog()

    def configure_callbacks(self):
        return [ModleeCallback()]


class ModleeCallback(Callback):
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
        else:
            pl_module.log(f"{phase}_loss", outputs)
        return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        # log the code text as a python file
        code_text = modlee_pypi.get_code_text_for_model(
            pl_module, include_header=True)
        mlflow.log_text(code_text, 'model.py')
        return super().setup(trainer, pl_module, stage)

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # log the data statistics
        data_stats = DataStats(x=trainer.train_dataloader.dataset.data)
        mlflow.log_dict(data_stats.data_stats, 'data_stats')
        mlflow.log_param('batch_size', trainer.train_dataloader.batch_size)
        return super().on_train_start(trainer, pl_module)
