from modlee.data_metafeatures import DataMetafeatures
from .recommender import Recommender
from modlee.model import RecommendedModel
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import modlee
from modlee.converter import Converter
from modlee.utils import get_model_size, typewriter_print

modlee_converter = Converter()


class ImageRecommender(Recommender):
    """
    Recommender for image models.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.modality = "image"
        self.MetafeatureClass = modlee.data_metafeatures.ImageDataMetafeatures

    def calculate_metafeatures(self, dataloader, *args, **kwargs):
        return super().calculate_metafeatures(
            dataloader,
            data_metafeature_cls=modlee.data_metafeatures.ImageDataMetafeatures,
        )

    def fit(self, dataloader, *args, **kwargs):
        """
        Fit the recommended to an image dataloader.

        :param dataloader: The dataloader, should contain images as the first batch element.
        """
        super().fit(dataloader, *args, **kwargs)
        assert self.metafeatures is not None
        if hasattr(self, 'num_classes'):
            self.metafeatures.update({"num_classes": self.num_classes})

        try:            
            self.model_text = self._get_model_text(self.metafeatures)

            if not isinstance(self.model_text, str):
                self.model_text = self.model_text.decode("utf-8")
            model = modlee_converter.onnx_text2torch(self.model_text)
            for param in model.parameters():
                try:
                    torch.nn.init.xavier_normal_(param, 1.0)
                except:
                    torch.nn.init.normal_(param)

            self.model = RecommendedModel(model, loss_fn=self.loss_fn, modality=self.modality)

            self.code_text = self.get_code_text()
            self.model_code = modlee_converter.onnx_text2code(self.model_text)

            self.write_file(self.model_text, "./model.txt")
            self.write_file(self.model_code, "./model.py")

        except:
            logging.error(
                f"ImageReccomender.fit failed, could  not return a recommended model, defaulting model to None"
            )
            self.model = None


class ImageClassificationRecommender(ImageRecommender):
    """
    Recommender for image classification tasks.
    Uses cross-entropy loss.
    """

    def __init__(self, num_classes=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task = "classification"
        self.loss_fn = F.cross_entropy
        # Check if num_classes is set, raise ValueError with a professional error message
        if num_classes is None:
            raise ValueError("recommender.fit: num_classes must be provided when using for modality='image', task='classification'.")
        self.num_classes = num_classes
