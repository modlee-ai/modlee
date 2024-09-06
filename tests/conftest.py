""" 
Configure pytest.
"""
import torch
import pandas as pd
import pytest
import inspect
from torchvision import datasets as tv_datasets
from torchvision import models as tvm
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision import models as tvm
from torchtext import models as ttm
from pytorch_forecasting import NBeats, AutoRegressiveBaseModel
from modlee.timeseries_dataloader import TimeSeriesDataset



def NbeatsInit():
    data = pd.read_csv('data/HDFCBANK.csv')
    data.drop(columns=['Series', 'Symbol','Trades', 'Deliverable Volume', 'Deliverble'], inplace=True)
    encoder_column = data.columns.tolist()
    dataset = TimeSeriesDataset(data=data, target = 'Close', time_column='Date',
                                                       encoder_column=encoder_column, input_seq=2,
                                                       output_seq=1)
    model = NBeats.from_dataset(
        dataset=dataset.get_dataset()
    )
    return model


def makeDataloader():
    data = pd.read_csv('data/HDFCBANK.csv')
    data.drop(columns=['Series', 'Symbol','Trades', 'Deliverable Volume', 'Deliverble'], inplace=True)
    encoder_column = data.columns.tolist()
    dataset = TimeSeriesDataset(data=data, target = 'Close', time_column='Date',
                                                       encoder_column=encoder_column, input_seq=2,
                                                       output_seq=1).to_dataloader(batch_size=1)
    
    return dataset

IMAGE_MODELS = [
    tvm.resnet18(weights="DEFAULT"),
    tvm.resnet18(),
    tvm.resnet50(),
    tvm.resnet152(),
    tvm.googlenet(),
]

IMAGE_MODELS = [
    tvm.resnet18(weights="DEFAULT"),
    tvm.resnet18(),
    tvm.resnet50(),
    tvm.resnet152(),
    # tvm.alexnet(),
    tvm.googlenet(),
]

# IMAGE_MODELS = []
# for attr in dir(tvm):
#     tvm_attr = getattr(tvm, attr)
#     if not callable(tvm_attr) or isinstance(tvm_attr, type):
#         continue
#     try:
#         inspect.signature(tvm_attr).bind()
#     except TypeError:
#         continue
#     tvm_attr_ret = tvm_attr()
#     if 'forward' in dir(tvm_attr_ret):
#         print(f"Adding {tvm_attr}")
#         IMAGE_MODELS.append(tvm_attr_ret)
        
# breakpoint()

IMAGE_SEGMENTATION_MODELS = [
    tvm.segmentation.fcn_resnet50(),
    tvm.segmentation.fcn_resnet101(),
    # tvm.segmentation.lraspp(),
    tvm.segmentation.lraspp_mobilenet_v3_large(),
    tvm.segmentation.deeplabv3_resnet50(),
    tvm.segmentation.deeplabv3_resnet101(),
]
TEXT_MODELS = [
    # ttm.FLAN_T5_BASE,
    # ttm.FLAN_T5_BASE_ENCODER,
    # ttm.FLAN_T5_BASE_GENERATION,
    ttm.ROBERTA_BASE_ENCODER,
    ttm.ROBERTA_DISTILLED_ENCODER,
    # ttm.T5_BASE,
    # ttm.T5_BASE_ENCODER,
    # ttm.T5_SMALL,
    # ttm.T5_SMALL_ENCODER,
    # ttm.T5_SMALL_GENERATION,
    ttm.XLMR_BASE_ENCODER,
    # ttm.XLMR_LARGE_ENCODER, # Too large for M2 MacBook Air?
]

class simpleModel(torch.nn.Module):
    def __init__(self):
        super(simpleModel, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1)
        )

    def forward(self, x):
        # Assuming x is a dictionary with a key 'x' that holds the tensor
        #x = x['x']
        #print(x.shape)
        return self.model(x)
    
class mlp(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(mlp, self).__init__()
        self.hidden_layers = torch.nn.ModuleList()
        self.hidden_layers.append(torch.nn.Linear(input_size, hidden_size[0]))
        for i in range(1, len(hidden_size)):
            self.hidden_layers.append(torch.nn.Linear(hidden_size[i-1], hidden_size[i]))
        self.output_layer = torch.nn.Linear(hidden_size[-1], output_size)
    def forward(self, x, input_2 = None):
        for layer in self.hidden_layers:
            x = torch.nn.functional.relu(layer(x))
        x = self.output_layer(x)
        return x

class TransformerModel(torch.nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()
        self.model = torch.nn.Transformer(d_model=10, nhead=2, num_encoder_layers=6, num_decoder_layers=6)
        self.fc = torch.nn.Linear(10, 1)

    def forward(self, x):
        tgt_dummy = x
        x = self.model(x, tgt_dummy)
        x = self.fc(x)
        return x


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=100, input_channels=10, sequence_length=20):
        self.num_samples = num_samples
        self.input_channels = input_channels
        self.sequence_length = sequence_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random input data
        x = torch.randn(self.input_channels, self.sequence_length)  # For conv1dModel
        # Generate random target data
        y = torch.randn(1)
        return {'x': x}, y

dummy_data = DummyDataset(num_samples=10, input_channels=10, sequence_length=10)

def dummyDataloader(dummy_data1=dummy_data):
    return DataLoader(dummy_data1, batch_size=1, shuffle=False)

# Usage for simpleModel
def get_input_for_simple_model():
    dataloader = dummyDataloader()
    for batch in dataloader:
        x, y = batch
        x = x['x']
        x = x.reshape(1, -1)
    return x, y

# Usage for conv1dModel
def get_input_for_conv1d_model():
    dataloader = dummyDataloader()
    for batch in dataloader:
        x, y = batch
        x = x['x']
        x = x.unsqueeze(0)
    return x, y

class simpleLSTM(torch.nn.Module):
    def __init__(self):
        super(simpleLSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=10, hidden_size=10)
        self.fc = torch.nn.Linear(10, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

class conv1dModel(torch.nn.Module):
    def __init__(self):
        super(conv1dModel, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=10, out_channels=10, kernel_size=3)
        self.conv2 = torch.nn.Conv1d(in_channels=10, out_channels=1, kernel_size=3)
        self.relu = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten()
        
    def forward(self, x):
        #x = x['x']
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.flatten(x)
        return x

TIMESERIES_MODELS = [
    #pfm.TemporalFusionTransformer(),
    #pfm.LSTM(input_size=1, hidden_size=10),
    #pfm.GRU(input_size=1, hidden_size=10, num_layers=1),
    #pfm.RecurrentNetwork()
    #simpleLSTM(),
    simpleModel(),
    conv1dModel(),
    transformerModel(),
    #mlp(input_size=10, output_size=10, hidden_size=[64, 128, 64])
    #pfm.AutoRegressiveBaseModel(),
    #NbeatsInit(),
]
DATALOADER = [
    get_input_for_simple_model(),
    get_input_for_conv1d_model(),
    makeDataloader(),

]

@pytest.fixture()
def dataloaders(batch_size=64):
    training_loader = DataLoader(
        tv_datasets.CIFAR10(
            root="data", train=True, download=True, transform=ToTensor()
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        tv_datasets.CIFAR10(
            root="data", train=False, download=True, transform=ToTensor()
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    return training_loader, test_loader

def _check_has_metafeatures(mf, metafeature_types): 

    features = {}
    for metafeature_type in metafeature_types:
        assert hasattr(mf, metafeature_type), f"{mf} has no attribute {metafeature_type}"
        assert isinstance(getattr(mf, metafeature_type), dict), f"{mf} {metafeature_type} is not dictionary"
        # Assert that the attribute is a flat dictionary
        assert not any([isinstance(v,dict) for v in getattr(mf, metafeature_type).values()]), f"{mf} {metafeature_type} not a flat dictionary"
        features.update(getattr(mf, metafeature_type))

def _check_metafeatures_timesseries(mf, metafeature_types):
    for metafeature_type in metafeature_types:
        assert metafeature_type in mf, f"{mf} has no key {metafeature_type}"