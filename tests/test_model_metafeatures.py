import pytest
# from . import conftest
try:
    import conftest
except:
    from . import conftest

IMAGE_MODELS = conftest.IMAGE_MODELS

import torch
from torch import nn
import torchvision
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
# breakpoint()
@pytest.mark.experimental
class TestModelMetafeatures:
    
    def test_model_metafeatures(self):
        model_mf = mmf.ModelMetafeatures(
            MODEL
        ) 
        pass
    
    @pytest.mark.parametrize("image_model", conftest.IMAGE_MODELS)
    def test_image_model_metafeatures(self, image_model):
        image_mf = mmf.ImageModelMetafeatures(
            image_model
        )
        # breakpoint() 
        self._check_has_metafeatures(image_mf)
        return image_mf
        pass
    
    @pytest.mark.parametrize("image_model", conftest.IMAGE_SEGMENTATION_MODELS)
    def test_image_segmentation_model_metafeatures(self, image_model):
        image_mf = mmf.ImageSegmentationModelMetafeatures(
            image_model
        )
        # breakpoint() 
        self._check_has_metafeatures(image_mf)
        return image_mf
        pass
    
    @pytest.mark.parametrize("text_model", conftest.TEXT_MODELS[:2])
    def test_text_model_metafeatures(self, text_model):
        text_mf = mmf.TextModelMetafeatures(
            text_model
        )
        # breakpoint()
        pass

    def _check_has_metafeatures(self, mf):
        conftest._check_has_metafeatures(
            mf,
            metafeature_types={
                'embedding', 
                'properties',        
            }
        )
        pass

if __name__=="__main__":
    breakpoint()
    print('hello')