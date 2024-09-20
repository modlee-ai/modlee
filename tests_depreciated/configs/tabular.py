import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch
import pandas as pd

from modlee.utils import tabular_loaders

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


input_dim = 10
mlp_model = MLP(input_dim)
snn_model = SNN(input_dim)
tabnet_model = EnhancedTabNet(input_dim)
resnet_model = EnhancedResNet(input_dim)


TABULAR_MODELS = [
    TabularClassificationModleeModel(
        _type=_type, 
        input_dim=input_dim
    ) for _type in MODEL_TYPES.keys()
]
TABULAR_MODALITY_TASK_KWARGS = [
    ("tabular", "classification", {"_type":"MLP"})
]
TABULAR_MODALITY_TASK_MODEL = [
    ("tabular", "classification", model) for model in TABULAR_MODELS
]
TABULAR_SUBMODELS = TABULAR_MODALITY_TASK_MODEL


# TABULAR_LOADERS = {
#     "housing_dataset": get_housing_dataloader,
#     "adult_dataset": get_adult_dataloader,
#     "diabetes_dataset": get_diabetes_dataloader,
# }
# tabular_loaders = list(TABULAR_LOADERS.values())

