""" 
Recommender for image models.
"""
from .recommender import Recommender
from modlee.model import RecommendedModel

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# from torchvision.models import \
#     resnet34, ResNet34_Weights, \
#     resnet18, ResNet18_Weights, \
#     resnet152, ResNet152_Weights


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

    def _append_classifier_to_model(self, model, num_classes):
        """ 
        Helper function to append a classifier to a given model (deprecated?).
        
        :param model: The model on which to append a classifier.
        :param num_classes: The number of classes.
        :return: A tuple of the model object and an executable code string to rebuild the model.
        """
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = model
                self.model_clf_layer = nn.Linear(1000, num_classes)

            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        num_layers = 1
        num_channels = 8

        ret_model = VariableConvNet(
            num_layers, num_channels, self.input_sizes, self.num_classes
        )
        model_str = "VariableConvNet({},{},{},{})".format(
            num_layers, num_channels, self.input_sizes, self.num_classes
        )

        for i in range(10):

            model = VariableConvNet(
                int(num_layers), int(num_channels), self.input_sizes, self.num_classes
            )

            if get_model_size(model) < self.max_model_size_MB:
                ret_model = model
                num_layers += 1
                num_channels = num_channels * 2
            else:
                break

        return ret_model, model_str

    def fit(self, dataloader, *args, **kwargs):
        """ 
        Fit the recommended to an image dataloader.
        
        :param dataloader: The dataloader, should contain images as the first batch element.
        """
        super().fit(dataloader, *args, **kwargs)
        assert self.metafeatures is not None
        # num_classes = len(dataloader.dataset.classes)
        if hasattr(dataloader.dataset, "classes"):
            num_classes = len(dataloader.dataset.classes)
        else:
            # try to get all unique values
            # assumes all classes will be represented in several batches
            unique_labels = set()
            n_samples = 0
            # while n_samples < 200:
            for d in dataloader.dataset:
                tgt = d[-1]
                # img,tgt = next(iter(dataloader))
                unique_labels.update(list(tgt.unique().cpu().numpy()))
                n_samples += len(tgt)
                # num_classes = len(tgt.unique())
            num_classes = len(unique_labels)
            # num_classes = 21
            # print(f'{unique_labels = }')
        self.metafeatures.update({"num_classes": num_classes})
        try:
        # if 1:
            self.model_text = self._get_model_text(self.metafeatures)
            # breakpoint()
            model = modlee_converter.onnx_text2torch(self.model_text)
            for param in model.parameters():
                # torch.nn.init.constant_(param,0.001)
                try:
                    torch.nn.init.xavier_normal_(param, 1.0)
                except:
                    torch.nn.init.normal_(param)
            self.model = RecommendedModel(model, loss_fn=self.loss_fn)

            self.code_text = self.get_code_text()
            self.model_code = modlee_converter.onnx_text2code(self.model_text)
            self.model_text = self.model_text.decode("utf-8")
            # breakpoint()
            clean_model_text = ">".join(self.model_text.split(">")[1:])
            # typewriter_print(clean_model_onnx_text,sleep_time=0.005)
            # self.write_files()
            self.write_file(self.model_text, "./model.txt")
            self.write_file(self.model_code, "./model.py")
            typewriter_print(f"The model is available at the recommender object's `model` attribute.")

        except:
        # else:
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
        # self.loss_fn = F.cross_entropy
        self.loss_fn = torch.nn.CrossEntropyLoss()

        def squeeze_entropy_loss(x, *args, **kwargs):
            return torch.nn.CrossEntropyLoss()(x.squeeze)

