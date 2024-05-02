import pytest
from . import conftest

import torch
from torch import nn
import torchvision
from torchvision import models as tvm
from torchtext import models as ttm

import modlee
from modlee import model_metafeatures as mmf

class NeuralNetwork(nn.Module):
    def __init__(self, input_size=10, hidden_size=10, output_size=10):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

MODEL = NeuralNetwork()
IMAGE_MODELS = [
    tvm.resnet18(),
    tvm.resnet50(),
    tvm.resnet152(),
    tvm.googlenet(),
]
TEXT_MODELS = [
    # ttm.RobertaBundle(ttm.RobertaEncoderConf(vocab_size=1000)),
    # ttm.XLMR_BASE_ENCODER(),
    # ttm.XLMR_LARGE_ENCODER()
]
# breakpoint()
@pytest.mark.experimental
class TestModelMetafeatures:
    
    def test_model_metafeatures(self):
        model_mf = mmf.ModelMetafeatures(
            MODEL
        ) 
        pass
    
    @pytest.mark.parametrize("image_model", IMAGE_MODELS)
    def test_image_model_metafeatures(self, image_model):
        image_mf = mmf.ImageModelMetafeatures(
            image_model
        )
        # breakpoint()
        self._check_has_metafeatures(image_mf)
        return image_mf
        pass
    
    def test_text_model_metafeatures(self):
        pass

    def _check_has_metafeatures(self, mf):
        conftest._check_has_metafeatures(
            metafeature_types={
                'embedding', 
                'properties',        
            }
        )
        pass