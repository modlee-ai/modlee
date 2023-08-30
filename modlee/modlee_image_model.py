from typing import Any, Optional
import lightning.pytorch as pl
from lightning.pytorch import Trainer, LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT

import mlflow
import modlee
from modlee.modlee_model import ModleeModel, DataStatsCallback
from lightning.pytorch.callbacks import Callback

import torchmetrics
from torchmetrics import Accuracy


class ModleeImageModel(ModleeModel):
    """
    Subclass of ModleeModel with image-specific convenience wrappers
    - Logs classification accuracy
    - Calculates data-specific data statistics
    """

    def __init__(self, num_classes=None, *args, **kwargs):
        if not num_classes:
            raise AttributeError("Must provide argument for num_classes")
        else:
            self.num_classes = num_classes
        vars_cache = {
            'num_classes':num_classes
        }
        ModleeModel.__init__(self, vars_cache=vars_cache, *args, **kwargs)

    def configure_callbacks(self):
        base_callbacks = ModleeModel.configure_callbacks(self)
        # save accuracy
        image_callback = ImageCallback(self.num_classes)
        # save image-specific datastats
        image_datastats_callback = DataStatsCallback(
            DataStats=getattr(modlee.data_stats, 'ImageDataStats', None))
        return [*base_callbacks, image_callback, image_datastats_callback]


class ImageCallback(Callback):
    """
    Saves accuracy
    """
    def __init__(self, num_classes=None, *args, **kwargs):
        Callback.__init__(self, *args, **kwargs)
        self.calculate_accuracy = Accuracy(
            task='binary' if num_classes == 1 else 'multiclass',
            num_classes=num_classes,
        )

    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT | None, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if batch_idx == 0:
            self._calculate_accuracy(pl_module, batch)
        return super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def _calculate_accuracy(self, pl_module, batch):
        data, targets = batch
        preds = pl_module(data)
        self.calculate_accuracy.to(device=pl_module.device)
        acc = self.calculate_accuracy(
            preds, targets,
        )
        mlflow.log_metric('val_acc', acc)
