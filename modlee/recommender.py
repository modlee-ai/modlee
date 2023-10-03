
import modlee
from modlee.converter import Converter
modlee_converter = Converter()
import logging
logging.basicConfig(level=logging.INFO)

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision.models import \
    resnet34, ResNet34_Weights, \
    resnet18, ResNet18_Weights, \
    resnet152, ResNet152_Weights
# test_models = {
#     'resnet34':resnet34(weights=ResNet34_Weights.IMAGENET1K_V1,),
#     'resnet152':resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
# }

class Recommender(object):
    def __init__(self) -> None:
        self._model = None
        self.meta_features = None
        pass
    
    def __call__(self, dataloader):
        self.dataloader = dataloader
        self.meta_features = self.calculate_meta_features(dataloader)
        
        pass
    
    def analyze(self,*args,**kwargs):
        self(*args, **kwargs)
    
    def calculate_meta_features(self, dataloader):
        print("Data fingerprinting...")
        if modlee.data_stats.module_available:
            return modlee.data_stats.DataStats(dataloader,testing=True).stats_rep
        else:
            return {}
        
    @property
    def model(self):
        if self._model is None:
            logging.info('No model recommendation, call .fit on a dataloader first.')            
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
        
    def __call__(self,dataloader,*args,**kwargs):
        super().__call__(dataloader,*args,**kwargs)
        num_classes = len(dataloader.dataset.classes)
        self.meta_features.update({
            'num_classes':num_classes
        })
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torchvision.models.resnet18(
                    weights=ResNet18_Weights.IMAGENET1K_V1,
                    progress=False,
                )
                self.model_clf_layer = nn.Linear(1000,num_classes)
            def forward(self,x):
                x = self.model(x)
                x = self.model_clf_layer(x)
                return x
        model = Model()
        # model = modlee_converter.torch2torch(model)
        self.model_code = modlee_converter.torch2code(model)
        # print(f"model code: {self.model_code}")
        self.model = RecommendedModel(
            modlee_converter.code2torch(self.model_code))
        
        
class RecommendedModel(modlee.modlee_model.ModleeModel):
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
