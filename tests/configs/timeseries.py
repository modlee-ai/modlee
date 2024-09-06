import torch
from torch.utils.data import DataLoader
from modlee.timeseries_dataloader import TimeseriesDataset
import pandas as pd
from modlee.utils import timeseries_loaders
from modlee. model.timeseries_model import TimeseriesClassificationModleeModel


def makeDataloader():
    data = pd.read_csv("data/HDFCBANK.csv")
    data.drop(
        columns=["Series", "Symbol", "Trades", "Deliverable Volume", "Deliverble"],
        inplace=True,
    )
    encoder_column = data.columns.tolist()
    dataset = TimeseriesDataset(
        data=data,
        target="Close",
        time_column="Date",
        encoder_column=encoder_column,
        input_seq=2,
        output_seq=1,
    ).to_dataloader(batch_size=1)

    return dataset


# TODO - is this definition deprecated?
# TIMESERIES_MODELS = [
#     pfm.TemporalFusionTransformer(),
#     pfm.DeepAR(),
#     pfm.LSTM(),
#     pfm.GRU(),
# ]

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
        return {"x": x}, y


dummy_data = DummyDataset(num_samples=10, input_channels=10, sequence_length=10)


def dummyDataloader(dummy_data1=dummy_data):
    return DataLoader(dummy_data1, batch_size=1, shuffle=False)


# Usage for simpleModel
def get_input_for_simple_model():
    dataloader = dummyDataloader()
    for batch in dataloader:
        x, y = batch
        x = x["x"]
        x = x.reshape(1, -1)
    return x, y


# Usage for conv1dModel
def get_input_for_conv1d_model():
    dataloader = dummyDataloader()
    for batch in dataloader:
        x, y = batch
        x = x["x"]
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

simpleModel = TimeseriesClassificationModleeModel("simple")
conv1dModel = TimeseriesClassificationModleeModel("conv1d")
TIMESERIES_MODELS = [
    # pfm.TemporalFusionTransformer(),
    # pfm.LSTM(input_size=1, hidden_size=10),
    # pfm.GRU(input_size=1, hidden_size=10, num_layers=1),
    # pfm.RecurrentNetwork()
    # simpleLSTM(),
    simpleModel,
    conv1dModel,
    # pfm.AutoRegressiveBaseModel(),
    # NbeatsInit(),
]


# DATALOADER = timeseries_loaders = [
#     get_input_for_simple_model(),
#     get_input_for_conv1d_model(),
#     makeDataloader(),
# ]

TIMESERIES_MODALITY_TASK_MODEL = [    
    ("timeseries", "classification", simpleModel),
    ("timeseries", "classification", conv1dModel),
]

TIMESERIES_MODALITY_TASK_KWARGS = [
    ("timeseries", "classification", {"_type": "simple"}),
    ("timeseries", "classification", {"_type": "conv"}),
]

