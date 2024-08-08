""" 
Modlee model for images. 
"""
import inspect
from typing import Any, Optional
import lightning.pytorch as pl
from lightning.pytorch import Trainer, LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT

import mlflow
from modlee import data_metafeatures as dmf, model_metafeatures as mmf
from modlee.model import ModleeModel, DataMetafeaturesCallback, ModelMetafeaturesCallback
from lightning.pytorch.callbacks import Callback

import torchmetrics
from torchmetrics import Accuracy

TASK_METRIC = {"classification": "Accuracy", "regression": "MeanSquaredError"}


class ImageModleeModel(ModleeModel):
    """
    Subclass of ModleeModel with image-specific convenience wrappers

    - Logs classification accuracy
    - Calculates data-specific data statistics
    """

    def __init__(self,
        task="classification",
        num_classes=None,
        *args, **kwargs):
        """ 
        ModleeImageModel constructor.
        
        :param task: The task ('classification','segmentation')
        :param num_classes: The number of classes, defaults to None.
        """
        if not num_classes:
            raise AttributeError("Must provide argument for num_classes")
        else:
            self.num_classes = num_classes
        vars_cache = {"num_classes": num_classes, "task": task}
        # self.image_callback = ImageCallback(
        #     metric = TASK_METRIC[self.task],
        #     **kwargs
        # )
        ModleeModel.__init__(self, modality="image", task=task, kwargs_cache=vars_cache, *args, **kwargs)

    def configure_callbacks(self):
        """ 
        Configure image-specific callbacks.
        """
        return super().configure_callbacks()
    
        # Assuming the 
        base_callbacks = ModleeModel.configure_callbacks(self)
        # save accuracy
        # image_callback = self.image_callback
        # image_callback = ImageCallback(self.num_classes)
        # save image-specific datastats
        image_datastats_callback = DataMetafeaturesCallback(
            DataMetafeatures=dmf.ImageDataMetafeatures,
        )
        return [*base_callbacks, image_datastats_callback]

class ImageClassificationModleeModel(ImageModleeModel):
    def __init__(self, *args, **kwargs):
        super().__init__(task='classification', *args, **kwargs)

    def configure_callbacks(self):
        base_callbacks = ImageModleeModel.configure_callbacks(self)
        image_model_mf_callback = ModelMetafeaturesCallback(
            ModelMetafeatures=mmf.ImageClassificationMetafeatures
        )
        return [*base_callbacks, image_model_mf_callback]

class ImageSegmentationModleeModel(ImageModleeModel):
    def __init__(self, *args, **kwargs):
        super().__init__(task='segmentation', *args, **kwargs)

    def configure_callbacks(self):
        base_callbacks = ImageModleeModel.configure_callbacks(self)
        image_model_mf_callback = ModelMetafeaturesCallback(
            ModelMetafeatures=mmf.ImageSegmentationMetafeatures
        )
        return [*base_callbacks, image_model_mf_callback]


