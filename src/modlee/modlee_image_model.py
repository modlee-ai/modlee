import inspect
from typing import Any, Optional
import lightning.pytorch as pl
from lightning.pytorch import Trainer, LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT

import mlflow
import modlee
from modlee.model import ModleeModel, DataMetafeaturesCallback
from lightning.pytorch.callbacks import Callback

import torchmetrics
from torchmetrics import Accuracy

TASK_METRIC = {
    'classification':'Accuracy',
    'regression':'MeanSquaredError'
}

class ModleeImageModel(ModleeModel):
    """
    Subclass of ModleeModel with image-specific convenience wrappers
    - Logs classification accuracy
    - Calculates data-specific data statistics
    """

    def __init__(self, task='classification', num_classes=None, *args, **kwargs):
        if not num_classes:
            raise AttributeError("Must provide argument for num_classes")
        else:
            self.num_classes = num_classes
        self.task = task
        vars_cache = {
            'num_classes':num_classes,
            'task':task,
        }
        # self.image_callback = ImageCallback(
        #     metric = TASK_METRIC[self.task],
        #     **kwargs
        # )
        ModleeModel.__init__(self, kwargs_cache=vars_cache, *args, **kwargs)

    def configure_callbacks(self):
        base_callbacks = ModleeModel.configure_callbacks(self)
        # save accuracy
        # image_callback = self.image_callback        
        image_callback = ImageCallback(self.num_classes)
        # save image-specific data_metafeatures
        image_data_mf_callback = DataMetafeaturesCallback(
            DataMetafeatures=getattr(modlee.data_metafeatures, 'ImageDataMetafeatures', None))
        return [*base_callbacks, image_callback, image_data_mf_callback]


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
        
class _ImageCallback(Callback):
    def __init__(self, metric='Accuracy', *args, **kwargs):
        self.metric = metric
        self.metric_func = getattr(torchmetrics, metric, None)
        if self.metric_func is not None:
            metric_func_kwargs = inspect.signature(self.metric_func).parameters.items()
            metric_func_kwargs = {
                k:kwargs.get(k,v.default) for k,v in inspect.signature(
                    self.metric_func).parameters.items()
            }
            self.metric_func = self.metric_func(**metric_func_kwargs)
        
        self.__dir__.update(kwargs) 
        
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.metric_func:
            self.metric_func.to(device=pl_module.device)
        return super().on_fit_start(trainer, pl_module)

    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT | None, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if batch_idx == 0:
            # _metric_func = getattr(self, f"_calculate_{self.metric}".lower(), None)
            # if _metric_func is not None:
            if self.metric_func is not None:
                data, targets = batch
                preds = pl_module(data)
                mlflow.log(
                    f"val_{self.metric}",
                    self.metric_func(preds,targets)
                )
        return super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def _calculate_accuracy(self, pl_module, batch):
        data, targets = batch
        preds = pl_module(data)
        # self.calculate_accuracy.to(device=pl_module.device)
        acc = self.metric_func(
            preds, targets,
        )
        mlflow.log_metric('val_acc', acc)
        
    def _calculate_meansquarederror(self, pl_module, batch):
        data, targets = batch
        preds = pl_module(data)
        # self.
        
    # @self._calculate    
    # def _calculate_mae(self,)

        
    # def _calculate(func):
    #     def wrapper(pl_module, batch):
    #         data, targets = batch
    #         preds = pl_module(data)
    #         return func(preds, targets)
    #         # return 
    #     return wrapper