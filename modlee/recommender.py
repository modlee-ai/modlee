
from torchvision.models import \
    resnet34, ResNet34_Weights, \
    resnet18, ResNet18_Weights, \
    resnet152, ResNet152_Weights
import torchvision
from torch.nn import functional as F
from torch import nn
import torch
import logging
import modlee
from modlee.converter import Converter
modlee_converter = Converter()
logging.basicConfig(level=logging.INFO)


class Recommender(object):
    """
    Recommends models given a dataset

    Args:
        object (_type_): _description_
    """

    def __init__(self) -> None:
        self._model = None
        self.meta_features = None

    def __call__(self, *args, **kwargs):
        """
        Wrapper to analyze
        """
        self.analyze(*args, **kwargs)

    def analyze(self, dataloader, *args, **kwargs):
        """
        Set dataloader and calculate meta features

        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader
        """
        self.dataloader = dataloader
        self.meta_features = self.calculate_meta_features(dataloader)

    def calculate_meta_features(self, dataloader):
        if modlee.data_stats.module_available:
            print("Data fingerprinting...")
            return modlee.data_stats.DataStats(dataloader, testing=True).stats_rep
        else:
            print("Could not fingerprint data (check access to server)")
            return {}

    @property
    def model(self):
        if self._model is None:
            logging.info(
                'No model recommendation, call .analyze on a dataloader first.')
        return self._model

    @model.setter
    def model(self, model):
        self._model = model


class DumbRecommender(Recommender):
    def __init__(self) -> None:
        super().__init__()


class ActualRecommender(Recommender):
    def __init__(self) -> None:
        super().__init__()


class ImageRecommender(Recommender):
    def __init__(self):
        super().__init__()


class ImageClassificationRecommender(ImageRecommender):
    def __init__(self):
        super().__init__()

    def analyze(self, dataloader, *args, **kwargs):
        """
        Returns a ready-to-train Lightning model given a dataloader

        Args:
            dataloader (_type_): The dataloader to analyze

        Returns:
            _type_: The Lightning model
        """
        super().analyze(dataloader, *args, **kwargs)
        num_classes = len(dataloader.dataset.classes)
        self.meta_features.update({
            'num_classes': num_classes
        })
        rec_model = self._get_model(self.meta_features)
        self.model_code = modlee_converter.torch2code(rec_model)
        self.model = RecommendedModel(
            modlee_converter.code2torch(self.model_code))

    def _get_model(self, meta_features):
        """
        Recommend a model based on meta-features

        Args:
            meta_features (_type_): A dictionary of meta-features

        Returns:
            torch.nn.Module: The recommended model
        """
        num_classes = meta_features['num_classes']

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torchvision.models.resnet18(
                    weights=ResNet18_Weights.IMAGENET1K_V1,
                    progress=False,
                )
                self.model_clf_layer = nn.Linear(1000, num_classes)

            def forward(self, x):
                x = self.model(x)
                x = self.model_clf_layer(x)
                return x
        return Model()


class RecommendedModel(modlee.modlee_model.ModleeModel):
    """
    A ready-to-train ModleeModel that wraps around a recommended model
    Defines a basic training pipeline

    Args:
        modlee (_type_): _description_
    """
    def __init__(self, model, loss_fn=F.cross_entropy, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.loss_fn = loss_fn

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_out = self(x)
        loss = self.loss_fn(y_out, y)
        return {'loss': loss}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_out = self(x)
        loss = self.loss_fn(y_out, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=0.001,
            momentum=0.9
        )
        return optimizer
