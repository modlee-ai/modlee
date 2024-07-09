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
        
    def calculate_metafeatures(self, dataloader, *args, **kwargs):
        return super().calculate_metafeatures(dataloader, data_metafeature_cls=modlee.data_metafeatures.ImageDataMetafeatures)

    def fit(self, dataloader, *args, **kwargs):
        """ 
        Fit the recommended to an image dataloader.
        
        :param dataloader: The dataloader, should contain images as the first batch element.
        """
        super().fit(dataloader, *args, **kwargs)
        assert self.metafeatures is not None
        if hasattr(dataloader.dataset, "classes"):
            num_classes = len(dataloader.dataset.classes)
        else:
            # Try to get all unique values
            # Assumes all classes will be represented in several batches
            unique_labels = set()
            n_samples = 0
            for d in dataloader.dataset:
                tgt = d[-1]
                unique_labels.update(list(tgt.unique().cpu().numpy()))
                n_samples += len(tgt)
            num_classes = len(unique_labels)
        self.metafeatures.update({"num_classes": num_classes})

        try:
            self.model_text = self._get_model_text(self.metafeatures)
            model = modlee_converter.onnx_text2torch(self.model_text)
            for param in model.parameters():
                try:
                    torch.nn.init.xavier_normal_(param, 1.0)
                except:
                    torch.nn.init.normal_(param)
            self.model = RecommendedModel(model, loss_fn=self.loss_fn)

            self.code_text = self.get_code_text()
            self.model_code = modlee_converter.onnx_text2code(self.model_text)
            self.model_text = self.model_text.decode("utf-8")
            clean_model_text = ">".join(self.model_text.split(">")[1:])
            
            self.write_file(self.model_text, "./model.txt")
            self.write_file(self.model_code, "./model.py")
            logging.info(f"The model is available at the recommender object's `model` attribute.")
        except:
            print(
                "Could not retrieve model, could not access server or data features may be malformed."
            )
            self.model = None

class ImageClassificationRecommender(ImageRecommender):
    """ 
    Recommender for image classification tasks.
    Uses cross-entropy loss.
    """
    def __init__(
        self,
       *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.task = "classification"
        self.loss_fn = F.cross_entropy

class ImageSegmentationRecommender(ImageRecommender):
    """ 
    Recommender for image segmentation tasks.
    Uses cross entropy loss.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task = "segmentation"
        self.loss_fn = torch.nn.CrossEntropyLoss()

        def squeeze_entropy_loss(x, *args, **kwargs):
            return torch.nn.CrossEntropyLoss()(x.squeeze)