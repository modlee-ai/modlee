import modlee
from modlee.converter import Converter
modlee_converter = Converter()

import torchvision
from torchvision.models import \
    resnet34, ResNet34_Weights, \
    resnet152, ResNet152_Weights
torchvision.models.resnet.ResNet()
test_models = {
    'resnet34':resnet34(weights=ResNet34_Weights.IMAGENET1K_V1,),
    'resnet152':resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
}

class Recommender(object):
    def __init__(self) -> None:
        pass
    
    def __call__(self, dataloader, task):
        self.dataloader = dataloader
        self.task = task
        self.meta_features = self.calculate_meta_features(dataloader)
        
        pass
    
    def fit(self,*args,**kwargs):
        self(*args, **kwargs)
    
    def calculate_meta_features(self, dataloader):
        if modlee.data_stats.module_available:
            return modlee.data_stats.DataStats(dataloader).stats_rep
        else:
            return {}
    
    
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
        
    def predict(self):
        model = torchvision.models.ResNet34_Weights(
               weights=ResNet34_Weights.IMAGENET1K_V1,
        )
        onnx_model = modlee_converter.torch2code