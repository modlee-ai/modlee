""" 
Modlee model for images. 
"""
from modlee.model import ModleeModel
from pytorch_tabular.models.tabnet import TabNetModel
from pytorch_tabular.models.category_embedding import CategoryEmbeddingModel

class TabularModleeModel(ModleeModel):

    def __init__(self, config=None, inferred_config=None, task="classification", *args, **kwargs):
        self.config = config
        self.inferred_config = inferred_config
        ModleeModel.__init__(self, modality="tabular", task=task, *args, **kwargs)


class TabNetModleeModel(TabularModleeModel):
    def __init__(self, config=None, inferred_config=None, *args, **kwargs):
        super().__init__(config=config, inferred_config=inferred_config, task='classification', *args, **kwargs)
        self.tabnet_model = TabNetModel(config=config, inferred_config=inferred_config)
    
    def get_model(self):
        return self.tabnet_model



class CategoryEmbeddingModleeModel(TabularModleeModel):
    def __init__(self, config=None, inferred_config=None, *args, **kwargs):
        super().__init__(config=config, inferred_config=inferred_config, task='classification', *args, **kwargs)
        self.category_embedding_model = CategoryEmbeddingModel(config=config, inferred_config=inferred_config)

    def get_model(self):
        return self.category_embedding_model

