""" 
Modlee model for images. 
"""
from modlee.model import ModleeModel
from pytorch_tabular.models.tabnet import TabNetModel
from pytorch_tabular.models.category_embedding import CategoryEmbeddingModel
from pytorch_tabular.models.gandalf import GANDALFModel
from pytorch_tabular.models.tab_transformer import TabTransformerModel
import torch
import torch.nn as nn

class TabularModleeModel(ModleeModel):

    def __init__(self, config=None, inferred_config=None, task="classification", *args, **kwargs):
        self.config = config
        self.inferred_config = inferred_config
        ModleeModel.__init__(self,
            # modality="tabular", task=task,
            *args, **kwargs)

class TabularClassificationModleeModel(TabularModleeModel):
    pass


class TabNetModleeModel(TabularClassificationModleeModel):
    def __init__(self, config=None, inferred_config=None, *args, **kwargs):
        super().__init__(config=config, inferred_config=inferred_config, task='classification', *args, **kwargs)
        self.tabnet_model = TabNetModel(config=config, inferred_config=inferred_config)
    
    def get_model(self):
        return self.tabnet_model



class CategoryEmbeddingModleeModel(TabularClassificationModleeModel):
    def __init__(self, config=None, inferred_config=None, *args, **kwargs):
        super().__init__(config=config, inferred_config=inferred_config, task='classification', *args, **kwargs)
        self.category_embedding_model = CategoryEmbeddingModel(config=config, inferred_config=inferred_config)

    def get_model(self):
        return self.category_embedding_model
    
class GANDALFModleeModel(TabularClassificationModleeModel):
    def __init__(self, config=None, inferred_config=None, *args, **kwargs):
        super().__init__(config=config, inferred_config=inferred_config, task='classification', *args, **kwargs)
        self.gandalf_model = GANDALFModel(config=config, inferred_config=inferred_config)

    def get_model(self):
        return self.gandalf_model

class DANetModleeModel(TabularClassificationModleeModel):
    def __init__(self, config=None, inferred_config=None, *args, **kwargs):
        super().__init__(config=config, inferred_config=inferred_config, task='classification', *args, **kwargs)
        self.danet_model = DANetModel(config=config, inferred_config=inferred_config)

    def get_model(self):
        return self.danet_model


class TabTransformerModleeModel(TabularClassificationModleeModel):
    def __init__(self, config=None, inferred_config=None, *args, **kwargs):
        super().__init__(config=config, inferred_config=inferred_config, task='classification', *args, **kwargs)
        self.tab_transformer_model = TabTransformerModel(config=config, inferred_config=inferred_config)

    def get_model(self):
        return self.tab_transformer_model

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(0.3)
        
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        
        x = self.fc4(x)
        return x

class SNN(nn.Module):
    def __init__(self, input_dim):
        super(SNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.AlphaDropout(0.1)
        
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.AlphaDropout(0.1)
        
        self.fc3 = nn.Linear(64, 32)
        self.dropout3 = nn.AlphaDropout(0.1)
        
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.selu(self.fc1(x))
        x = self.dropout1(x)
        
        x = torch.selu(self.fc2(x))
        x = self.dropout2(x)
        
        x = torch.selu(self.fc3(x))
        x = self.dropout3(x)
        
        x = self.fc4(x)
        return x

class EnhancedTabNet(nn.Module):
    def __init__(self, input_dim):
        super(EnhancedTabNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        
        self.attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.Sigmoid()
        )
        
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        
        att = self.attention(x)
        x = x * att
        
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class EnhancedResNet(nn.Module):
    def __init__(self, input_dim):
        super(EnhancedResNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 2)
        
        self.shortcut = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128)
        )

    def forward(self, x):
        if self.training and x.size(0) == 1:
            self.bn1.eval()
            self.bn2.eval()
            self.bn3.eval()

        residual = self.shortcut(x)
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        
        x += residual

        x = torch.relu(self.bn3(self.fc3(x)))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x
